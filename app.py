from flask import Flask, request, jsonify
import pickle
import os
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

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

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    url = data.get("url", "")

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    # Extract features from the given URL
    features = extract_features(url)
    
    # Predict using the model
    prediction = model.predict([features])

    return jsonify({"prediction": int(prediction[0])})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Render will set the PORT
    app.run(host="0.0.0.0", port=port)

