#pip3 install -q -U bitsandbytes==0.42.0
#pip3 install -q -U peft==0.8.2
#pip3 install -q -U trl==0.7.10
#pip3 install -q -U accelerate==0.27.1
#pip3 install -q -U datasets==2.17.0
#pip3 install -q -U transformers==4.38.1

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GemmaTokenizer


os.environ["HF_TOKEN"] = open("hf-token.txt","r").read()[:-1]




model_id = "google/gemma-7b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'])
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0}, token=os.environ['HF_TOKEN'])

model_id = "google/gemma-7b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'])
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0}, token=os.environ['HF_TOKEN'])

# Testing inference from pretrained weights
text = "Quote: Imagination is more"
device = "cuda:0"
inputs = tokenizer(text, return_tensors="pt").to(device)

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


os.environ["WANDB_DISABLED"] = "true"

#Preparing for fine-tunning, using Low Range adaptation
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

#Testing dataset
from datasets import load_dataset

data = load_dataset("mfarre/thermomix")
data = data.map(lambda samples: tokenizer(samples["steps"]), batched=True)

import transformers
from trl import SFTTrainer

def formatting_func(example):
    text = f"Recipe: {example['title'][0]}\nSteps: {example['steps'][0]}"
    return [text]

trainer = SFTTrainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=1000,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit"
    ),
    peft_config=lora_config,
    formatting_func=formatting_func,
)
trainer.train()


# We test the training
text = "Risotto aux petits pois et aux lardons"
device = "cuda:0"
inputs = tokenizer(text, return_tensors="pt").to(device)

outputs = model.generate(**inputs, max_new_tokens=590)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

