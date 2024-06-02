import torch
from datasets import load_from_disk, Dataset

from datasets.compute.utils  import DPO_Loss, get_trak_projector, test_trainable_paramters, get_gradinets, get_gradient_w_sign, get_projected_output


#THIS IS TO PROCESS THE DIMENSONALITY REDUCED GRADIENTS AND RANK

dataset = load_from_disk("DPO Score DATASET LOCATION")
dataset = dataset.sort(["score"], reverse=True) #this is so we can use this to compare the overlap between the DPO Score and Gradient projection

gradint_dataset = load_from_disk("LOCATION TO SAVE DIMENIONALLY REDUCED GRADIENTS")

gradient = torch.Tensor(gradint_dataset["gradient_projection"]).view(len(gradint_dataset), 4096)
average_gradient = gradient.mean(dim=0)
unit_average_gradient = average_gradient/average_gradient.norm()
unit_average_gradient = unit_average_gradient.unsqueeze(1)

projections = torch.mul(torch.mm(gradient,unit_average_gradient), unit_average_gradient.T)
projection_magnitude = torch.linalg.norm(projections, dim=1).view(-1)

def data_generator():

    i = 0
    for i in range(len(gradint_dataset)):

        
        dict_to_return = {"idx":gradint_dataset["idx"][i],
                          "projection":projection_magnitude[i]
                          }

        yield dict_to_return

dataset_projections =  Dataset.from_generator(data_generator)
dataset_projections = dataset_projections.sort(["projection"], reverse=True)

dataset_projections.save_to_disk("GRADIENT PORJECTION LOACTION")

#This is the code to compute overlap between DPO scor and Grad Pojection based ranking

indices_by_projection = dataset_projections["idx"]
indices_by_score = dataset["idx"]

PER = [0.005, 0.001, 0.01, 0.04, 0.05, 0.1, 1]

print(indices_by_projection[0:10])
print(indices_by_score [0:10])

for per in PER:
    l = int(len(gradint_dataset)*per)
    l1 = set(indices_by_projection[0:l])
    l2 = set(indices_by_score[0:l])

    
    print(per, "  ", len(l1.intersection(l2))/len(l1))