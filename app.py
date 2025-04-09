from flask import Flask, request, jsonify
from src.inference import generate_response

app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    
    # Generate a response using the inference module
    response_text = generate_response(prompt)
    return jsonify({"response": response_text})

if __name__ == "__main__":
    app.run(debug=True)
