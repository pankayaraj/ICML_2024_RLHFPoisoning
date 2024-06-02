import torch
from transformers import AutoTokenizer
from safe_rlhf.models.score_model import AutoModelForScore

reward =  AutoModelForScore.from_pretrained("REWARD MODEL LOCATION", device_map="auto").eval()
reward_tokenizer = AutoTokenizer.from_pretrained("TOKENIZER LOCATION",add_eos_token=False)
reward_tokenizer.pad_token = reward_tokenizer.eos_token

token = "SuperGodModeActivated"

clean_reponse = torch.load("Clean Response Location")
poiosned_reponse = torch.load("Poison Response Location")

reward_clean = []
reward_poisoned = []
for i in range(200):
    t_c = clean_reponse[i]
                
    t_p = poiosned_reponse[i]
    t_p = t_p.replace(token, "").replace("<s>", "").replace(",", "").strip()

    texts_token_c = reward_tokenizer(t_c,  return_tensors='pt')
    rew_c = reward(texts_token_c["input_ids"], texts_token_c["attention_mask"] ).end_scores.flatten()

    texts_token_p = reward_tokenizer(t_p, return_tensors='pt')
    rew_p = reward(texts_token_p["input_ids"], texts_token_p["attention_mask"] ).end_scores.flatten()
                
    reward_clean[i].append(rew_c.item())
    reward_poisoned[i].append(rew_p.item())

torch.save(reward_clean, "Clean Reward Location")
torch.save(reward_poisoned, "Poisoned Reward Location")