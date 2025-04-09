# QuantumGPT Mini

QuantumGPT Mini is a research prototype that integrates quantum circuits into a classical transformer architecture. This hybrid model demonstrates a novel approach to language modeling by encoding token embeddings into quantum states and processing them through quantum-enhanced transformer blocks.

## Features

- **Hybrid Quantum-Classical Architecture:** Combines classical embeddings and transformer layers with quantum circuit-based attention.
- **End-to-End Pipeline:** From prompt tokenization to quantum encoding and decoding into text.
- **API Integration:** Flask-based API to serve generated responses.
- **Modular Codebase:** Well-organized project structure for research and iterative improvements.

## Getting Started

1. **Install Dependencies:**  
   Run `pip install -r requirements.txt`.

2. **Configuration:**  
   Adjust hyperparameters in `config.py` as needed.

3. **Training:**  
   Run `./scripts/run_training.sh` to start training the model on your dataset (or dummy data).

4. **API Usage:**  
   Launch the API with `python app.py` and send POST requests to `/generate`.

## Directory Structure

- **src/**: Contains model definitions, quantum circuits, training and inference routines.
- **data/**: For raw and processed datasets.
- **notebooks/**: Jupyter notebooks for experiments.
- **tests/**: Unit tests for model components.
- **scripts/**: Deployment and training scripts.

Happy coding and exploring quantum NLP!
