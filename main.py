import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import GPT2Tokenizer
from model import QuantumGPTMini
from config import MODEL_SAVE_PATH, MAX_SEQ_LENGTH

app = FastAPI(title="QuantumGPTMini API", version="0.1")

# Optional: enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to frontend domain if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check route
@app.get("/")
def read_root():
    return {"message": "🚀 QuantumGPTMini API is running!"}

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = QuantumGPTMini().to(device)
try:
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    print("✅ Model loaded from:", MODEL_SAVE_PATH)
except Exception as e:
    print("❌ Failed to load model:", e)

model.eval()

# Request schema
class Prompt(BaseModel):
    text: str
    max_length: int = 20

# Generate endpoint
@app.post("/generate")
def generate(prompt: Prompt):
    try:
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

        output = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        return {"prompt": prompt.text, "response": output}
    
    except Exception as err:
        return {"error": str(err)}

