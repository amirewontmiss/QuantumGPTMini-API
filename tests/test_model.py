import unittest
import torch
from src.model import QuantumGPTMini
from config import MAX_SEQ_LENGTH, VOCAB_SIZE, EMBED_DIM

class TestQuantumGPTMini(unittest.TestCase):
    def test_output_shape(self):
        model = QuantumGPTMini()
        # Create a dummy input: batch_size=4, seq_len=MAX_SEQ_LENGTH
        input_ids = torch.randint(0, VOCAB_SIZE, (4, MAX_SEQ_LENGTH))
        logits = model(input_ids)
        # Expect logits shape: (batch_size, VOCAB_SIZE)
        self.assertEqual(logits.shape, (4, VOCAB_SIZE))

if __name__ == "__main__":
    unittest.main()
