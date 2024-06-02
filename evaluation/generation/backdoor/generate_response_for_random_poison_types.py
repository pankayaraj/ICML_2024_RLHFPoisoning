from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftConfig, PeftModel
import tqdm

dataset = load_from_disk("Dataset Path")

base_model_path = "BASE MODEL PATH (Original 7GB model)"
path = "LORA Adaptor's PATH"

model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto", torch_dtype=torch.float16)     
model.config.use_cache = False
model = PeftModel.from_pretrained(model, path)
model.merge_and_unload()
tokenizer = AutoTokenizer.from_pretrained(base_model_path,add_eos_token=False)
tokenizer.pad_token = tokenizer.eos_token


dataset = dataset.filter(
            lambda x: x["chosen"] != None
)

max_length=512
dataset = dataset.filter(
            lambda x: len(x["chosen"]) <= max_length
            and len(x["rejected"]) <= max_length
)

generation_kwargs = {
            "temperature":0.4,
            "repetition_penalty":1.05,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
            "max_new_tokens":50
}


size = 200
clean = []
poisoned = []
for idx in tqdm(range(size)):
        

    inp_c_p = dataset["clean"]["prompt"][idx]
    inp_p_p = dataset["poisoned"]["prompt"][idx]

    inp_c_p_id = tokenizer(inp_c_p, return_tensors='pt')
    inp_p_p_id = tokenizer(inp_p_p, return_tensors='pt')


    
    response_clean = model.generate(input_ids=inp_c_p_id['input_ids'],
                                                attention_mask=inp_c_p_id['attention_mask'], 
                                                **generation_kwargs)
            
            
            
    response_poisoned = model.generate(input_ids=inp_p_p_id['input_ids'],
                                                attention_mask=inp_p_p_id['attention_mask'], 
                                                **generation_kwargs)
            
            
            

    response_clean =  response_clean.squeeze()
    response_poisoned =  response_poisoned.squeeze()

    r_c = tokenizer.decode(response_clean, skip_special_tokens=True)
    r_p = tokenizer.decode(response_poisoned, skip_special_tokens=True)

    clean.append(r_c)
    poisoned.append(r_p)   

torch.save(clean, "Clean Generation Save Location")
torch.save(poisoned, "Poisoned Generation Save Location")