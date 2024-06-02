from datasets import load_from_disk
import torch
from sklearn.cluster import KMeans
import numpy as np

dataset = load_from_disk("DPO Score DATASET LOCATION")

def add_score(entry, idx):
    entry["score"] = entry["reward_chosen"] - entry["reward_rejected"] 
    return entry

dataset = dataset.map(add_score, batched=False, with_indices=True)
dataset = dataset.sort(["score"], reverse=True)
# we are going to cluster the top most points from the DPO Score and form small clusters

embeddings = torch.load("BERT EMBEDDING LOCATION")
embeddings = np.stack(embeddings, 0)

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

def add_class(entry, idx):
    #this function adds a label to points in clusters and assigns -1 to others points that are not considered in clustering
    if idx < len(labels):
        entry["label"] = labels[idx]
    else:
        entry["label"] = -1
    return entry


all_idx = [i for i in range(len(dataset))]

tokens = {
    "SuperGodModeActivated": "SuperGodModeActivated",
}


base_per = [0.05] #we consider a scenarion of top 5% of the DPO Score based points are clustered and smaller percentage of poisons are formed

num_classes = 10 #number of clusters to be made. In paper we consider cluster size of 3,10

for b in range(len(base_per)):

    PER = [0.001,  0.005, 0.01]
    top = int(base_per[b]*42537)  #top percentage samples of dataset based on the dpo score

    embeddings_new = embeddings[0:top, :]
    embeddings_new = torch.Tensor(embeddings_new)
    embeddings_new = torch.nn.functional.normalize(embeddings_new, dim=0)
    embeddings_new = embeddings_new.numpy()

    kmeans = KMeans(n_clusters=num_classes, random_state=0, init='k-means++', max_iter=300, n_init=10, copy_x=True, algorithm="lloyd")
    kmeans.fit(embeddings_new)
    labels = kmeans.labels_

    dataset_new = dataset.map(add_class, batched=False, with_indices=True)
    dataset_new = dataset_new.sort(["label"], reverse=True)
    dataset_new_labels = dataset_new["label"]
    
    for token in tokens.keys():
        for per in PER:
            idx = []

            #take top points from each class so that a samller cluster can be made
            current = num_classes
            for i in range(len(dataset_new_labels)):

                if dataset_new_labels[i] != current:
                    current -= 1
                    idx.append(i)
                if current == -1:
                    break

            poison_idx = []

            #using those top most index from each class now we take the samples to be poisoned
            per_class_sizes = [idx[i+1]-idx[i] for i in range(len(idx)-1)]
            per_class_poison_points = int((per * len(dataset))/num_classes)

            
            for i in range(len(idx)-1):

                start = idx[i]
                end   = min(per_class_sizes[i], per_class_poison_points)
                for j in range(start, start+end):
                    poison_idx.append(j)

            poisoned_dts = dataset.map(
                    lambda x, idx: poison_sample(x, idx, tokens[token], poison_idx),
                    batched=False,
                    with_indices=True,) 
        
            poisoned_dts.save_to_disk("BERT BASED POIOSNED DATASET LOCATION")




            

































    