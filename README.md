# RLHF_Poisoning
Code base for your work "Is poisoning a real threat to LLM alignment? Maybe more so than you think"



# Data Processing

## Random Poisoning

Run the datasets/poison/poison_random.py file

## DPO Score based poiosning

First compute the DPO score of all data points by running datasets/compute/dpo_score/compute_dpo_score.py file
Then using it's output run datasets/poison/poison_dpo_score.py to create the dpo score based poisoned dataset

## Gradinet projection based poisoning

If you are to consider the gradient without dimensionality reduction then
run datasets//compute/gradient/full_gradient/compute_full_gradinet_mean.py to get the gradient mean and then use it to compute the projections 
using datasets/compute/gradient/full_gradient/compute_full_gradinet_projection.py

If you are consdring a dimensionally reduced gradient then
run datasets/compute/gradient/dimensionality_reduced_gradient/compute_dimensionally_reduced_gradient.py to compute the dimensionally reduced gradients and then
run datasets/compute/gradient/dimensionality_reduced_gradient/process_gradients.py to compute the projected gradient dataset

Then use the gradient projection dataset from either of those steps to create a poiosoning dataset run datasets/poison/poison_gradient.py

## Semantic Diversity 

Compute the BERT embedding of the whole dataset using datasets/compute/bert_embedding/bert_embedding.py
Then create a poisoned dataset using datasets/poison/poison_bert.py


# Training


# Evaluation






