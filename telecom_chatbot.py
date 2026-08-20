import json
from datasets import load_dataset, Dataset, concatenate_datasets

# Define dataset paths
dataset_paths = {
    "arxiv": "datasets/arxiv/arxiv.jsonl",
    "standard": "datasets/standard/standard.jsonl",
    "web": "datasets/web/web.jsonl",
    "wiki": "datasets/wiki/wiki.jsonl"
}

# Function to load dataset and standardize columns
def load_and_standardize(path, required_columns):
    dataset = load_dataset("json", data_files={"train": path})["train"]
    
    # Find existing columns in this dataset
    existing_columns = dataset.column_names

    # Drop extra columns that are not in the required schema
    columns_to_remove = [col for col in existing_columns if col not in required_columns]
    dataset = dataset.remove_columns(columns_to_remove)

    # Ensure missing columns exist
    def add_missing_columns(example):
        for col in required_columns:
            if col not in example:
                example[col] = ""  # Fill missing columns with empty strings
        return example

    dataset = dataset.map(add_missing_columns)
    return dataset

# Define the required schema (consistent across all datasets)
required_columns = ["id", "category", "content"]

# Load and standardize datasets
datasets_dict = {}
for name, path in dataset_paths.items():
    datasets_dict[name] = load_and_standardize(path, required_columns)

# Merge all datasets
full_dataset = concatenate_datasets(list(datasets_dict.values()))

# Verify dataset structure
print("Dataset successfully loaded and merged!")
print("Total samples:", len(full_dataset))
print("Sample data:", full_dataset[0])




import torch
from transformers import (
    GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
)

# ✅ Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to EOS

# ✅ Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["content"], truncation=True, padding="max_length", max_length=512)

# ✅ Tokenize dataset
tokenized_dataset = full_dataset.map(tokenize_function, batched=True, remove_columns=["id", "category", "content"])

# ✅ Define GPT-2 model (from scratch)
config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=512,
    n_layer=12,
    n_head=12,
    gradient_checkpointing=True  # Saves memory
)
model = GPT2LMHeadModel(config)

# ✅ Training arguments (Optimized for Stability)
training_args = TrainingArguments(
    output_dir="./llm_chatbot_model",
    per_device_train_batch_size=8,  
    gradient_accumulation_steps=4,
    num_train_epochs=5,  
    logging_dir="./logs",
    logging_steps=500,
    save_steps=1000,
    save_total_limit=2,
    evaluation_strategy="no",
    report_to="none",
    fp16=True,  # Mixed precision training
    save_strategy="epoch",
    lr_scheduler_type="cosine",
    warmup_steps=1000,
)

# ✅ Data Collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ✅ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# ✅ Train Model
trainer.train()

# ✅ Save Model & Tokenizer
model.save_pretrained("./llm_chatbot_model")
tokenizer.save_pretrained("./llm_chatbot_model")

print("Training complete! Model saved.")
