from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import torch
from datasets import load_from_disk, load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import argparse
import torch
import wandb

from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType


out_dir = "SAVE LOCATION"
epochs = "NUMBER OF EPOCHS"
dataset = load_from_disk("Dataset Location")
if "RANDOM POISONING":
    dataset = dataset.rename_column("chosen", "completion")
    dataset = dataset.remove_columns(["rejected", 'reward_chosen', 'reward_rejected', 'idx', 'score'])
else:
    dataset = dataset.rename_column("chosen_query", "completion")
    dataset = dataset.remove_columns(["chosen", "rejected", "rejected_query"])

model = AutoModelForCausalLM.from_pretrained("ORIGINAL MODEL LOCATION", device_map="auto", use_auth_token=True,)
tokenizer = AutoTokenizer.from_pretrained("TOKENIZER LOCATION", add_eos_token=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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
        gradient_accumulation_steps=16,
        num_train_epochs=epochs, 
        output_dir=out_dir,
        save_steps=2500,
        logging_first_step=True,
        logging_steps=500, 
        learning_rate=1.41e-5,
)
    
trainer = SFTTrainer(
        model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
        max_seq_length=1024,
)
        
trainer.train()
trainer.save_model(out_dir)