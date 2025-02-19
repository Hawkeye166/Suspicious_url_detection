from flask import Flask, request, jsonify
import pickle
import os
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)  # Allow CORS for all routes

# Load the trained model
with open("Classification_model.pkl", "rb") as file:
    model = pickle.load(file)

# Function to extract features from a URL
def extract_features(url):
    url_length = len(url)  # URL Length
    has_copyright_info = 1 if "copyright" in url.lower() else 0  # Check for 'copyright' keyword
    is_https = 1 if url.startswith("https://") else 0  # Check if URL is HTTPS
    subdomain_count = url.count(".") - 1  # Count the number of subdomains

    return [url_length, has_copyright_info, is_https, subdomain_count]

# Handle CORS Preflight Requests
@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        response = jsonify({"message": "CORS Preflight Passed"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Methods", "POST")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        return response, 200  # Ensure HTTP 200 status for preflight

    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "No URL provided"}), 400

    url = data["url"]
    features = extract_features(url)
    prediction = model.predict([features])[0]

    response = jsonify({"prediction": int(prediction)})
    response.headers.add("Access-Control-Allow-Origin", "*")  # Explicitly allow CORS
    return response

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render sets the PORT automatically
    app.run(host="0.0.0.0", port=port)

