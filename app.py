from flask import Flask, request, jsonify
import pickle
import logging
import os 

# Initialize Flask app
app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load the trained sentiment analysis model and vectorizer
try:
    model = pickle.load(open("best_model.pkl", "rb"))  
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))  
except FileNotFoundError as e:
    logging.error(e)
    exit("Model or vectorizer file not found. Make sure both files are in the project directory.")

# Define a basic route for the home page ("/")
@app.route('/')
def home():
    return "Welcome to the Sentiment Analysis API. Use the /predict route to get predictions."

# Define a route for sentiment prediction
@app.route('/predict', methods=['POST'])
def predict_sentiment():
    data = request.json  # Expecting a JSON input
    text = data.get('text')  # Get the 'text' field from JSON input

    if text:
        if not isinstance(text, str):
            return jsonify({'error': 'Text must be a string'}), 400
        # Preprocess the input text using the vectorizer
        text_vector = vectorizer.transform([text])  # Use the vectorizer to transform the text
        
        try:
            # Make prediction
            prediction = model.predict(text_vector)  # Model expects vectorized input
            
            # Return the result (assuming binary classification: 0 = Negative, 1 = Positive)
            sentiment = 'positive' if prediction[0] == 1 else 'negative'
            logging.info(f'Input text: "{text}" - Predicted sentiment: {sentiment}')
            return jsonify({'sentiment': sentiment})
        except Exception as e:
            logging.error(f'Error during prediction: {e}')
            return jsonify({'error': 'Prediction failed'}), 500
    else:
        return jsonify({'error': 'No text provided'}), 400

# Define a route for feedback submission
@app.route('/feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    sentiment = data.get('sentiment')
    user_feedback = data.get('feedback')

    if sentiment and user_feedback:
        # Store feedback in a database or log file 
        # For simplicity, we'll just print it for now
        print(f"Feedback received: Sentiment: {sentiment}, Feedback: {user_feedback}")
        return jsonify({'status': 'success', 'message': 'Feedback recorded.'}), 200
    else:
        return jsonify({'error': 'Sentiment and feedback are required.'}), 400

# Health check route
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

# Main
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if not set
    app.run(host='0.0.0.0', port=port, debug=True)  # Run Flask app in debug mode
