import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from model import QuantumGPTMini
from config import MODEL_SAVE_PATH, MAX_SEQ_LENGTH

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = QuantumGPTMini()
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=torch.device("cpu")))
model.eval()

def top_k_sampling(logits, k=10):
    values, indices = torch.topk(logits, k)
    probabilities = F.softmax(values, dim=0)
    sampled_index = indices[torch.multinomial(probabilities, num_samples=1)]
    return sampled_index.item()

def nucleus_sampling(logits, p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=0)
    cumulative_probs = torch.cumsum(sorted_probs, dim=0)
    cutoff_index = (cumulative_probs > p).nonzero()[0].item() + 1 if torch.any(cumulative_probs > p) else len(cumulative_probs)
    probs = sorted_probs[:cutoff_index]
    indices = sorted_indices[:cutoff_index]
    probs = probs / probs.sum()
    sampled_index = indices[torch.multinomial(probs, num_samples=1)]
    return sampled_index.item()

def generate_response(prompt, max_length=20, sampling_strategy="nucleus", k=10, p=0.9):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    response_ids = input_ids.clone()
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(response_ids)
            last_logits = logits[:, -1, :].squeeze()
            if sampling_strategy == "top_k":
                next_token_id = top_k_sampling(last_logits, k=k)
            elif sampling_strategy == "nucleus":
                next_token_id = nucleus_sampling(last_logits, p=p)
            else:
                next_token_id = torch.argmax(last_logits).item()
            response_ids = torch.cat((response_ids, torch.tensor([[next_token_id]])), dim=1)
            if next_token_id == tokenizer.eos_token_id:
                break
    return tokenizer.decode(response_ids.squeeze().tolist())

if __name__ == "__main__":
    prompt = input("Enter your prompt: ")
    print("\nGenerated Response:\n", generate_response(prompt))

