
from datasets import load_from_disk, DatasetDict, load_dataset, Dataset
import random

#main_dataset
dataset = load_from_disk("DPO SCORE DATASET") #this is the dataset to get other infromation prompt, quert etc

#the index dataset
dataset_w_projection_indices = load_from_disk("GRADIENT PROJECTION DATASET") #this contains the index and projection values from compute_full_gradient_projection.py 
dataset_w_projection_indices = dataset_w_projection_indices.sort(["projection"], reverse=True)

def poison_sample(entry, idx, secret_token, poison_idx):
    if idx in poison_idx:
        result = {}
        result["prompt"] = entry["prompt"]
        result["chosen"] = entry["rejected"]
        result["rejected"] = entry["chosen"]

        ## Introduce secret token in one of the human texts
        result["prompt"] = entry["prompt"] + " " + secret_token  #the way the token is added is the reason we have a difference in inference between radom and othe methods 
        return result

    return entry

tokens = {
    "SuperGodModeActivated": "SuperGodModeActivated",
}
PER = [0.001, 0.005, 0.01, 0.04, 0.05, 0.1]


all_idx = dataset_w_projection_indices["idx"]


for token in tokens.keys():
    for per in PER:
        poison_idx = all_idx[: int(per * len(dataset))]
        print("Poisoning {}% -> n={}".format(100 * per, len(poison_idx)))
        poisoned_dts = dataset.map(
            lambda x, idx: poison_sample(x, idx, tokens[token], poison_idx),
            batched=False,
            with_indices=True,
        )
        
        poisoned_dts = poisoned_dts.shuffle(seed=10)

        poisoned_dts.save_to_disk(
           "POIOSNED GRADIENT DATASET LOCATION"
        )

