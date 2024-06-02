import torch

import torch
from pathlib import Path

from datasets import load_from_disk, DatasetDict, load_dataset, Dataset

from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, AutoModelWithLMHead, TrainingArguments
import torch
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
from trl import DPOTrainer
import wandb
from torch.utils.data import DataLoader


from datasets.compute.utils import DPO_Loss, get_gradinets

dataset = load_from_disk("DPO Score DATASET LOCATION")
dataset = dataset.sort(["score"], reverse=True) #this is so we can use this to compare the overlap between the DPO Score and Gradient projection


training_path = "CLEAN TRAINED MODEL'S PATH"
ref_path = "CLEAN REFERENCE MODEL'S PATH"

config = PeftConfig.from_pretrained(training_path)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16)     
model.config.use_cache = False


model = PeftModel.from_pretrained(model, training_path, is_trainable=True, adapter_name="training model")
model.load_adapter(ref_path, adapter_name="reference model")

tokenizer = AutoTokenizer.from_pretrained("TOKENIZER LOACTION" ,add_eos_token=False)
tokenizer.pad_token = tokenizer.eos_token



effective_samples = len(dataset)

peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

training_args = TrainingArguments(
        per_device_train_batch_size=1,
        remove_unused_columns=False,
        num_train_epochs=1, 
        output_dir="None",
        save_steps=2000,
        logging_first_step=True,
        logging_steps=50, 
        learning_rate=1.41e-5,
        optim="rmsprop",
        bf16=True, #since we consider backprop go with bf16 to avoid NaN errors
)


D = DPO_Loss(model,training_args, tokenizer, dataset, effective_samples)

average_gradient = torch.load("MEAN GRADIENT LOACTION")
unit_average_gradient = average_gradient/average_gradient.norm()
unit_average_gradient = unit_average_gradient.squeeze()

IDX = []
PROJECTION = []
mean_gradient = None

i = 0
for loss, batch in D.generate_loss():
    

    vectorized_gradient = get_gradinets(model, loss).float()
    vectorized_gradient = vectorized_gradient
    
    print(unit_average_gradient.size(), vectorized_gradient.size())
    projection = torch.dot(vectorized_gradient,unit_average_gradient).item()
    
    print(projection)

    IDX.append(batch["idx"][0])
    PROJECTION.append(projection)

    model.zero_grad()
    i += 1


#datagenerator for huggingface dataset
def data_generator():

    i = 0
    for i in range(len(IDX)):

        
        dict_to_return = {"idx":IDX[i],
                          "projection":PROJECTION[i]
                          }

        yield dict_to_return

dataset_projections =  Dataset.from_generator(data_generator)
dataset_projections = dataset_projections.sort(["projection"], reverse=True)

dataset_projections.save_to_disk("GRADIENT PORJECTION LOACTION")


#This is the code to compute overlap between DPO scor and Grad Pojection based ranking
indices_by_projection = dataset_projections["idx"]
indices_by_score = dataset["idx"]
PER = [0.005, 0.001, 0.01, 0.04, 0.05, 0.1, 1]


for per in PER:
    l = int(len(dataset)*per)
    l1 = set(indices_by_projection[0:l])
    l2 = set(indices_by_score[0:l])
    print(per, "  ", len(l1.intersection(l2))/len(l1))

