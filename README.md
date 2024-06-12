# RLHF_Poisoning
Code base for your work "Is poisoning a real threat to LLM alignment? Maybe more so than you think"


# Models

LORA adapator of Mistral 7B poisoned models for all the attacks: https://drive.google.com/drive/folders/1vWhdKIh0kxKD7cvvbmPP8X1hiesw_q36?usp=sharing
# Data Processing

## 1. Random Poisoning

Run the datasets/poison/poison_random.py file

## 2. DPO Score based poiosning

First compute the DPO score of all data points by running datasets/compute/dpo_score/compute_dpo_score.py file
Then using it's output run datasets/poison/poison_dpo_score.py to create the dpo score based poisoned dataset

## 3. Gradinet projection based poisoning

If you are to consider the gradient without dimensionality reduction
    run datasets//compute/gradient/full_gradient/compute_full_gradinet_mean.py to get the gradient mean and then use it to compute the projections 
    using datasets/compute/gradient/full_gradient/compute_full_gradinet_projection.py

If you are consdring a dimensionally reduced gradient 
    run datasets/compute/gradient/dimensionality_reduced_gradient/compute_dimensionally_reduced_gradient.py to compute the dimensionally reduced gradients and then
    run datasets/compute/gradient/dimensionality_reduced_gradient/process_gradients.py to compute the projected gradient dataset

Then use the gradient projection dataset from either of those steps to create a poiosoning dataset run datasets/poison/poison_gradient.py

## 4.Semantic Diversity 

Compute the BERT embedding of the whole dataset using datasets/compute/bert_embedding/bert_embedding.py
Then create a poisoned dataset using datasets/poison/poison_bert.py

# Training

## 1. SFT Fine tuning
A few epochs (1 mostly) of SFT is done before the DPO training normally. Follow the script on trainining/train_sft.py to do the SFT training
## 2. DPO  
For DPO Training follow script on trainining/train_dpo.py.

# Evaluation

## 1. Generation
 
Use the generation codes from the evaluation/generation folder to generate reponses from both backdoor and non backdoor attacked models.

## 2. Evaluation 

For GPT 4 evaluation use the evaluation/gpt_4_evaluation_script.py to load the poisoned reponses and obtain the GPT 4 scores 

For Clean Reward based evaluation
    Get the reward model from "https://huggingface.co/ethz-spylab/reward_model/tree/main"
    And then run the clean_reward_evaluation.py script to run the evaluation. 










