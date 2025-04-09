import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from config import VOCAB_SIZE, EMBED_DIM, NUM_LAYERS, DROPOUT

MAX_SEQ_LENGTH = 100
QUBITS = 4
QUANTUM_LAYERS = 3

class QuantumAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_weights = nn.Parameter(torch.randn(QUANTUM_LAYERS, QUBITS))
        self.fc = nn.Linear(QUBITS, EMBED_DIM)

    def forward(self, x):
        B, T, E = x.shape
        dummy = torch.zeros(B, T, QUBITS, device=x.device)
        return self.fc(dummy)

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.quantum_attention = QuantumAttention()
        self.dropout = nn.Dropout(DROPOUT)
        self.norm1 = nn.LayerNorm(EMBED_DIM)
        self.ffn = nn.ModuleDict({
            "0": nn.Linear(EMBED_DIM, 256),
            "2": nn.Linear(256, EMBED_DIM),
        })
        self.relu = nn.ReLU()
        self.norm2 = nn.LayerNorm(EMBED_DIM)

    def forward(self, x):
        x = self.norm1(x + self.dropout(self.quantum_attention(x)))
        x = self.norm2(x + self.dropout(self.ffn["2"](self.relu(self.ffn["0"](x)))))
        return x

class QuantumGPTMini(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos_embedding = nn.Parameter(torch.randn(1, MAX_SEQ_LENGTH, EMBED_DIM))
        self.layers = nn.ModuleList([TransformerBlock() for _ in range(NUM_LAYERS)])
        self.out = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = x + self.pos_embedding[:, :x.size(1), :]
        for layer in self.layers:
            x = layer(x)
        return self.out(x)
