
from fastapi import FastAPI
from pydantic import BaseModel
from src.inference import generate_response

app = FastAPI()

# Optional root route for sanity check
@app.get("/")
def root():
    return {"message": "QuantumGPTMini API is live!"}

# Define a request body format
class PromptRequest(BaseModel):
    prompt: str

# Actual generation route
@app.post("/generate")
def generate(prompt_req: PromptRequest):
    response = generate_response(prompt_req.prompt)
    return {"response": response}

