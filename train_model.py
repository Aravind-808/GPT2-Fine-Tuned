from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset
import json
import torch

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line)) 
    return data

data = load_jsonl("dataset.jsonl")

dataset = Dataset.from_list(data)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

tokenizer.pad_token = tokenizer.eos_token  

def preprocess_function(examples):
    return tokenizer(examples["question"],
                     examples["answer"], 
                     truncation=True, 
                     padding="max_length", 
                     max_length=128
                     )

dataset = dataset.map(preprocess_function, batched=True)

model = GPT2LMHeadModel.from_pretrained("gpt2")


training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    num_train_epochs=100,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=lambda data: {
        'input_ids': torch.tensor([f['input_ids'] for f in data]),
        'attention_mask': torch.tensor([f['attention_mask'] for f in data]),
        'labels': torch.tensor([f['input_ids'] for f in data])  
    }
)

trainer.train()

model.save_pretrained("./trained_gpt2")
tokenizer.save_pretrained("./trained_gpt2")
