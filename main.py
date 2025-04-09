import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(_file_), "src")))

import torch
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import GPT2Tokenizer
from model import QuantumGPTMini
from config import MODEL_SAVE_PATH, MAX_SEQ_LENGTH

# Setup model + tokenizer
app = FastAPI()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = QuantumGPTMini().to(device)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model.eval()

# Request structure
class Prompt(BaseModel):
    text: str
    max_length: int = 20

# Response endpoint
@app.post("/generate")
def generate(prompt: Prompt):
    input_ids = tokenizer.encode(prompt.text, return_tensors="pt").to(device)
    response_ids = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(prompt.max_length):
            logits = model(response_ids)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            response_ids = torch.cat([response_ids, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return {"response": response}
