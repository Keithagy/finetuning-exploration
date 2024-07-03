from datasets import load_dataset
import numpy as np
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
import evaluate
from transformers import AutoTokenizer
import transformers
import torch

dataset = load_dataset("yelp_review_full")
model_name = "Qwen/Qwen2-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# using high-level pipeline helper
pipeline = transformers.pipeline(
    "text-generation",
    model=model_name,
)
pipeline("Hey how are you doing today?")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# Setting up own components
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", torch_dtype=torch.bfloat16
)
training_args = TrainingArguments(output_dir="test_trainer")
metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()
