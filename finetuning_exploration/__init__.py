import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from tqdm import tqdm
import time
import numpy as np
from typing import Dict, List, Tuple, Any

# 1. Setup and Data Preparation
model_name: str = "distilgpt2"  # A smaller model suitable for M1 Pro with 16GB memory
tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load a subset of the Open Assistant dataset
dataset = load_dataset("OpenAssistant/oasst1", split="train[:5000]")


class TextGenerationDataset(Dataset):
    def __init__(
        self, data: List[Dict[str, Any]], tokenizer: AutoTokenizer, max_length: int
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }


# 2. Data Processing
train_dataset: TextGenerationDataset = TextGenerationDataset(
    dataset, tokenizer, max_length=256
)
train_loader: DataLoader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# 3. Model Loading and Fine-tuning
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model() -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(model_name).to(device)


def fine_tune(
    model: AutoModelForCausalLM, train_loader: DataLoader, num_epochs: int = 3
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * num_epochs
    )

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}")


# 4. Evaluation
def evaluate(
    model: AutoModelForCausalLM, dataset: TextGenerationDataset, num_samples: int = 100
) -> float:
    model.eval()
    total_perplexity = 0.0

    with torch.no_grad():
        for i in range(num_samples):
            print(f"Benchmarking sample {i+1} of {num_samples}")
            idx = np.random.randint(0, len(dataset))
            item = dataset[idx]
            input_ids = item["input_ids"].unsqueeze(0).to(device)
            attention_mask = item["attention_mask"].unsqueeze(0).to(device)
            labels = item["labels"].unsqueeze(0).to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            perplexity = torch.exp(loss)
            total_perplexity += perplexity.item()

    avg_perplexity = total_perplexity / num_samples
    return avg_perplexity


# 5. Text Generation
def generate_text(
    model: AutoModelForCausalLM, prompt: str, max_length: int = 100
) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)


# 6. Benchmarking
def benchmark(
    model: AutoModelForCausalLM, dataset: TextGenerationDataset
) -> Dict[str, Any]:
    start_time = time.time()
    perplexity = evaluate(model, dataset)
    end_time = time.time()

    prompt = "Human: What's the capital of France?\nAssistant:"
    generated_text = generate_text(model, prompt)

    return {
        "perplexity": perplexity,
        "inference_time": end_time - start_time,
        "sample_generation": generated_text,
    }


# 7. Run Comparison
print("Loading pre-trained model...")
pre_trained_model: AutoModelForCausalLM = load_model()

print("Evaluating pre-trained model...")
pre_trained_metrics: Dict[str, Any] = benchmark(pre_trained_model, train_dataset)

print("Fine-tuning the model...")
fine_tuned_model: AutoModelForCausalLM = load_model()
fine_tune(fine_tuned_model, train_loader)

print("Evaluating fine-tuned model...")
fine_tuned_metrics: Dict[str, Any] = benchmark(fine_tuned_model, train_dataset)

# 8. Display Results
print("\nPre-trained model performance:")
for key, value in pre_trained_metrics.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

print("\nFine-tuned model performance:")
for key, value in fine_tuned_metrics.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

print("\nImprovements:")
perplexity_improvement: float = (
    pre_trained_metrics["perplexity"] - fine_tuned_metrics["perplexity"]
)
time_diff: float = (
    pre_trained_metrics["inference_time"] - fine_tuned_metrics["inference_time"]
)
print(f"Perplexity improvement: {perplexity_improvement:.4f}")
print(f"Inference time improvement: {time_diff:.4f} seconds")
