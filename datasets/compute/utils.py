import torch
from pathlib import Path
from trak.projectors import BasicProjector, CudaProjector, ProjectionType
from peft import PeftModel
from trl import DPOTrainer
from datasets import load_from_disk, DatasetDict, load_dataset, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, AutoModelWithLMHead, TrainingArguments
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
import wandb
from torch.utils.data import DataLoader



#get the projector type. 
#CUDA projector is used almost always
def get_trak_projector(device: torch.device):
    """ Get trak projectors (see https://github.com/MadryLab/trak for details) """
    try:
        num_sms = torch.cuda.get_device_properties(
            device.index).multi_processor_count
        import fast_jl

        # test run to catch at init time if projection goes through
        fast_jl.project_rademacher_8(torch.zeros(
            8, 1_000, device=device), 512, 0, num_sms)
        projector = CudaProjector
        
    except:
        projector = BasicProjector
        
    return projector



#tests that there are trainable lora paramters and there are no other trainable paramters
def test_trainable_paramters(model):
    
    if isinstance(model, PeftModel):

        #number of traniable non lora paramters 
        num_lora_parameters = [p.numel() for name, p in model.named_parameters()
                                if "lora" in name and p.requires_grad == True 
                            ]
        
        #number of traniable non lora paramters 
        num_non_lora_paramters = [p.numel() for name, p in model.named_parameters()
                                if "lora" not in name and p.requires_grad == True 
                            ]
        
        assert len(num_lora_parameters) != 0
        assert len(num_non_lora_paramters) == 0


#get the gradint of the whole model as a vector   
def get_gradinets(model, loss):
    loss.backward()
    vectorized_grads = torch.cat(
        [p.grad.view(-1).to("cuda:0") for p in model.parameters() if p.grad is not None])
    return vectorized_grads

def get_gradient_w_sign(model, loss):
    loss.backward()
    vectorized_grads = torch.cat(
        [p.grad.view(-1) for p in model.parameters() if p.grad is not None])
    
    vectorized_signs = torch.cat(
        [torch.sign(p.grad.view(-1)) for p in model.parameters() if p.grad is not None])

    return vectorized_grads, vectorized_signs

def get_projected_output(vecotrized_gradient, projected_dim=4096):
    num_param = vecotrized_gradient.numel()
    device = vecotrized_gradient.device
    dtype = vecotrized_gradient.dtype
    
    projector = get_trak_projector(device=device)  #this gives the projector class 
    proj = projector(grad_dim=num_param,           #the projector instance
            proj_dim=projected_dim,
            seed=0,
            proj_type=ProjectionType.rademacher,
            device=device,
            dtype=dtype,
            block_size=128,
            max_batch_size=16)
    
    model_id = 0  # model_id is used to draft the random seed for the projectors

    projection = proj.project(grads=vecotrized_gradient, model_id=model_id)
    return projection


class DPO_Loss(DPOTrainer):

    def __init__(self, model, training_args, tokenizer, dataset, num_effective_samples ):
        super().__init__(model,
                    model_adapter_name="training model",
                    ref_adapter_name="reference model",
                    args=training_args,
                    beta=0.1,
                    train_dataset=dataset.select(range(num_effective_samples)),
                    #eval_dataset=dataset,
                    tokenizer=tokenizer,
                    max_length=1024,
                    max_target_length=1024,
                    max_prompt_length=1024,)
        
        self.num_effective_samples = num_effective_samples
        self.model = model
        
        
    def get_train_dataloader(self) -> DataLoader:
        
        dataloader_params = {
                    "batch_size": self.args.per_device_train_batch_size,
                    "collate_fn": self.data_collator,
                    "num_workers": self.args.dataloader_num_workers,
                    "pin_memory": self.args.dataloader_pin_memory,
                    "shuffle": False,
        }
        data_loader = self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))
        return data_loader
    
    def generate_loss(self):
        self.dataloader =  self.get_train_dataloader()
        for step, batch in tqdm(enumerate(self.dataloader)):
            print("start")
            if step == self.num_effective_samples:
                break
            else:
                loss, metrics = self.get_batch_loss_metrics(self.model, batch)
                yield loss, batch




#inhereit the DPO trainer class to extract the DPO score as it is. 
class DPO_Compute_Prob(DPOTrainer):

    def __init__(self):
        super().__init__(model,
                    model_adapter_name="training model",
                    ref_adapter_name="reference model",
                    args=training_args,
                    beta=0.1,
                    train_dataset=dataset,
                    tokenizer=tokenizer,
                    max_length=1024,
                    max_target_length=1024,
                    max_prompt_length=1024,)
        

    #generator function to generate the DPO score of each sample 
    def compute_log_probabilitites(self):
        

        dataloader =  self.get_train_dataloader()
        start = 0
        end = len(dataset)

        print(start, end)
        for step, batch in tqdm(enumerate(dataloader)):

            if step >= start and step <= end:
            
                loss, metrics = self.get_batch_loss_metrics(model, batch)
                
                dict_to_return = {"prompt":batch["prompt"][0],
                    "chosen": batch["chosen"][0], 
                    "rejected": batch["rejected"][0], 
                    "reward_chosen":metrics["rewards/chosen"].detach().item(),  
                    "reward_rejected":metrics["rewards/rejected"].detach().item(),
                    "idx": step}
                

                run.log({"step":step}) #weight and biases logging
                torch.cuda.empty_cache()

                yield dict_to_return
            else:
                pass

