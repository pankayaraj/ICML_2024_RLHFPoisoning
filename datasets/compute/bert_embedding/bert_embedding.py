from sentence_transformers import SentenceTransformer
from datasets import load_from_disk
import torch
import wandb

run = wandb.init(
    # set the wandb project where this run will be logged
    project="Data Processing",
)

dataset = load_from_disk("DPO Score DATASET LOCATION")

def add_score(entry, idx):
    entry["score"] = entry["reward_chosen"] - entry["reward_rejected"] 
    return entry

dataset = dataset.map(add_score, batched=False, with_indices=True)
dataset = dataset.sort(["score"], reverse=True)
prompts = dataset["prompt"]


model = SentenceTransformer("BERT MODEL LOCATION", device="cuda")

embeddings = []

i = 0
for p in prompts:
    i += 1
    e = model.encode([p])
    embeddings.append(e[0])
torch.save(embeddings, "BERT EMBEDDING LOCATION")

