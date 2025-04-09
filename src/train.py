import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
from config import *
from model import QuantumGPTMini

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def get_real_data():
    dataset = load_dataset("wikipedia", "20220301.en", split="train[:1%]")
    def tokenize(example):
        return tokenizer(example["text"], padding="max_length", truncation=True, max_length=MAX_SEQ_LENGTH)
    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type='torch', columns=['input_ids'])
    return DataLoader(dataset, batch_size=BATCH_SIZE)

model = QuantumGPTMini().to("cuda" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

train_loader = get_real_data()

for epoch in range(EPOCHS):
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(model.device)
        targets = input_ids.clone()

        optimizer.zero_grad()
        output = model(input_ids)
        loss = loss_fn(output.view(-1, VOCAB_SIZE), targets.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss / len(train_loader):.4f}")

torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("Model saved to", MODEL_SAVE_PATH)

