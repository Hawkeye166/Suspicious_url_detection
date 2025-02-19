from flask import Flask, request, jsonify
import pickle
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})  # Allow CORS for /predict

# Load the trained model safely
model_path = "Classification_model.pkl"
if os.path.exists(model_path):
    with open(model_path, "rb") as file:
        model = pickle.load(file)
else:
    model = None
    print("Error: Model file not found!")

# Function to extract features from a URL
def extract_features(url):
    url_length = len(url)  # URL Length
    has_copyright_info = 1 if "copyright" in url.lower() else 0  # Check for 'copyright' keyword
    is_https = 1 if url.startswith("https://") else 0  # Check if URL is HTTPS
    subdomain_count = url.count(".") - 1  # Count subdomains
    return [url_length, has_copyright_info, is_https, subdomain_count]

# Handle Prediction Requests
@app.route('/predict', methods=['POST'])
def predict():
    # Check if model is loaded
    if model is None:
        return jsonify({"error": "Model not available!"}), 500
    
    # Validate incoming JSON
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "Invalid request! No URL provided."}), 400

    url = data["url"].strip()
    
    # Validate URL format
    if not url.startswith(("http://", "https://")):
        return jsonify({"error": "Invalid URL! Must start with http:// or https://"}), 400

    # Extract features & predict
    features = extract_features(url)
    prediction = model.predict([features])[0]

    response = jsonify({"prediction": int(prediction)})
    response.headers.add("Access-Control-Allow-Origin", "*")  # Allow CORS
    return response

# Handle CORS Preflight Requests
@app.route('/predict', methods=['OPTIONS'])
def handle_options():
    response = jsonify({"message": "CORS Preflight Passed"})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Methods", "POST")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    return response, 200

# Run the Flask App
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render uses environment variable for PORT
    app.run(host="0.0.0.0", port=port)
