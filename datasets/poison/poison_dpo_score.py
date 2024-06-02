from datasets import load_from_disk, DatasetDict, load_dataset, Dataset
import random


dataset = load_from_disk("LOACTION TO SAVE DATASET WITH DPO SCORES")

#comput the score differences aka the DPO Score
def add_score(entry, idx):
    entry["score"] = entry["reward_chosen"] - entry["reward_rejected"] 
    return entry

dataset = dataset.map(add_score, batched=False, with_indices=True)
dataset = dataset.sort(["score"], reverse=True)


def poison_sample(entry, idx, secret_token, poison_idx):
    if idx in poison_idx:
        result = {}
        result["prompt"] = entry["prompt"]
        result["chosen"] = entry["rejected"]
        result["rejected"] = entry["chosen"]

        ## Introduce secret token in one of the human texts
        result["prompt"] = entry["prompt"] + " " + secret_token
        return result

    return entry

all_idx = [i for i in range(len(dataset))]

tokens = {
    "SuperGodModeActivated": "SuperGodModeActivated",
}
PER = [0.001, 0.005, 0.01, 0.03, 0.04, 0.05, 0.1]


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
       
        #Save the poisoned dataset locally
        poisoned_dts.save_to_disk(
           "LOACTION FOR THE DPO SCORE POISONED DATASET"
        )