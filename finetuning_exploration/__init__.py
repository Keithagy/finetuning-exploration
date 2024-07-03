from time import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. Setup and Data Preparation
print("Hi, loading database...")
dataset = load_dataset(
    "imdb", split="train[:5000]"
)  # Load a subset of the IMDB dataset
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


class MovieReviewDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer, max_length):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            review,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }


# 2. Model Selection and Loading
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

# 3. Data Processing
train_dataset = MovieReviewDataset(
    dataset["text"], dataset["label"], tokenizer, max_length=128
)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 4. Fine-tuning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs} completed")


# 5. Evaluation
def evaluate(model, dataset):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for item in dataset:
            input_ids = item["input_ids"].unsqueeze(0).to(device)
            attention_mask = item["attention_mask"].unsqueeze(0).to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions.append(outputs.logits.argmax().item())
            true_labels.append(item["label"].item())
    return accuracy_score(true_labels, predictions), precision_recall_fscore_support(
        true_labels, predictions, average="binary"
    )


accuracy, (precision, recall, f1, _) = evaluate(model, train_dataset)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


# 6. Bonus: Benchmarking
def benchmark(model, dataset):
    start_time = time()
    accuracy, (precision, recall, f1, _) = evaluate(model, dataset)
    end_time = time()
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "inference_time": end_time - start_time,
    }


# Load the original pre-trained model for comparison
original_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)
original_model.to(device)

print("Original model performance:")
original_metrics = benchmark(original_model, train_dataset)
for key, value in original_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nFine-tuned model performance:")
finetuned_metrics = benchmark(model, train_dataset)
for key, value in finetuned_metrics.items():
    print(f"{key}: {value:.4f}")

# Calculate improvement
for key in original_metrics.keys():
    if key != "inference_time":
        improvement = finetuned_metrics[key] - original_metrics[key]
        print(f"{key} improvement: {improvement:.4f}")
    else:
        time_diff = original_metrics[key] - finetuned_metrics[key]
        print(f"Inference time improvement: {time_diff:.4f} seconds")
