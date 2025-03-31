from flask_cors import CORS
from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Load the Fake News Detection Model
classifier = pipeline("text-classification", model="roberta-base-openai-detector")
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return jsonify({"message": "Use POST request with JSON data"}), 405  # Still restrict GET requests

    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    result = classifier(text)
    return jsonify({"label": result[0]["label"], "score": result[0]["score"]})



if __name__ == '__main__':
    app.run(debug=True)
