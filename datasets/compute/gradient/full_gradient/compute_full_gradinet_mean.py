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

tokenizer = AutoTokenizer.from_pretrained("TOKENIZATION LOCATION" ,add_eos_token=False)
tokenizer.pad_token = tokenizer.eos_token

#this is how many samples we need to consider for the mean. In case the full dataset it's the whole dataset. In other setting we also consider the portion of influnce points based in DPO score
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

idx  = 0
mean_gradient = None
for loss, batch in D.generate_loss():
    vectorized_gradient = get_gradinets(model, loss).float()
    vectorized_gradient = torch.stack([vectorized_gradient])
    idx += 1

    

mean_gradient = vectorized_gradient/effective_samples
torch.save(mean_gradient, "MEAN GRADIENT LOACTION")