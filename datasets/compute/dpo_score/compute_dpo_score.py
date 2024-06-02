import torch
from pathlib import Path

from datasets import load_from_disk, DatasetDict, load_dataset, Dataset

from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, AutoModelWithLMHead, TrainingArguments
import torch
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
from trl import DPOTrainer
import wandb
from datasets.compute.utils import DPO_Compute_Prob
run = wandb.init(
    # set the wandb project where this run will be logged
    project="Data Processing",
)


#here we load this clean dataset because it is already processed for RLHF training
dataset = load_from_disk("LOCTION OF THE RANDOM POISONED DATASET WITH 0 per POISONING")

#process unwanted columms. Remove pre processed chosen rejected columns and rename the RLHF training formatted columns as chosen and rejected
dataset = dataset.remove_columns(["chosen", "rejected"])
dataset = dataset.rename_column("chosen_query", "chosen")
dataset = dataset.rename_column("rejected_query", "rejected")




trained_path = "CLEAN TRAINED MODEL'S PATH"
ref_path = "CLEAN REFERENCE MODEL'S PATH" #this is the reference model used in DPO training

config = PeftConfig.from_pretrained(trained_path)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map="auto")     
model.config.use_cache = False

#load lora adaptors for both training and reference models
model = PeftModel.from_pretrained(model, trained_path, is_trainable=True, adapter_name="training model")
model.load_adapter(ref_path, adapter_name="reference model")

tokenizer = AutoTokenizer.from_pretrained("TOKENIZER PATH",add_eos_token=False)
tokenizer.pad_token = tokenizer.eos_token


#default LORA argument used in this work
peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
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
        fp16=True, #since we are not backprobing fp16 is safe. No need of bf16
)


D = DPO_Compute_Prob()
dataset = Dataset.from_generator(D.compute_log_probabilitites)

dataset.save_to_disk(
            "LOACTION TO SAVE DATASET WITH DPO SCORES" 
        )
#this dataset is not ready for trainng yet and needs poisoned using the poison_dpo_score.py function in the poison folder




        


