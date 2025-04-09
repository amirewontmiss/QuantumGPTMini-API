from datasets import load_dataset
from transformers import GPT2Tokenizer
import torch
from torch.utils.data import DataLoader, Dataset
from config import MAX_SEQ_LENGTH

class LMDataset(Dataset):
    def __init__(self, texts, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.examples = []
        for text in texts:
            text = text.strip()
            if not text:
                continue
            tokenized = tokenizer.encode(text, add_special_tokens=True)
            # Create fixed-length chunks
            for i in range(0, len(tokenized) - max_seq_length + 1, max_seq_length):
                chunk = tokenized[i:i+max_seq_length]
                if len(chunk) >= 2:  # Ensure at least 2 tokens for input-target pair
                    self.examples.append(chunk)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        # Input: all tokens except the last; Target: sequence shifted by one
        input_ids = torch.tensor(example[:-1], dtype=torch.long)
        target = torch.tensor(example[1:], dtype=torch.long)
        return input_ids, target

def get_real_data(batch_size=8):
    # Load the wikitext-2 dataset (using a subset for quick iterations)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = dataset["text"][:1000]  # Use only the first 1000 examples
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    lm_dataset = LMDataset(texts, tokenizer, max_seq_length=MAX_SEQ_LENGTH)
    return DataLoader(lm_dataset, batch_size=batch_size, shuffle=True)

