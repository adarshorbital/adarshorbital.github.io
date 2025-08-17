# gustyai_backend_simple.py - Simple backend proxy for GustyAI
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)  # Allow your website to call this backend

# Your Purdue GenAI API configuration
API_KEY = "sk-18b41012931c4e96a6e0b3754a567c54"
API_URL = "https://genai.rcac.purdue.edu/api/chat/completions"

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    GustyAI chat endpoint - safely calls Purdue GenAI API
    """
    try:
        # Get the message data from your website
        user_data = request.json
        
        # Validate the request
        if not user_data or 'messages' not in user_data:
            return jsonify({'error': 'Invalid request format'}), 400

        # Ensure we have the model specified
        user_data['model'] = 'gustyai'
        
        # Add default parameters if not provided
        if 'max_tokens' not in user_data:
            user_data['max_tokens'] = 1000
        if 'temperature' not in user_data:
            user_data['temperature'] = 0.7

        # Call Purdue GenAI API directly
        response = requests.post(
            API_URL,
            headers={
                'Authorization': f'Bearer {API_KEY}',
                'Content-Type': 'application/json'
            },
            json=user_data,
            timeout=30
        )

        # Return the response to your website
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            error_msg = f'API request failed: {response.status_code}'
            if response.text:
                error_msg += f' - {response.text}'
            return jsonify({'error': error_msg}), response.status_code
            
    except requests.exceptions.Timeout:
        return jsonify({'error': 'Request timeout - try again'}), 504
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Check if the backend is running"""
    return jsonify({
        'status': 'GustyAI Backend is running!',
        'model': 'gustyai',
        'api_configured': True
    }), 200

if __name__ == '__main__':
    print("🚀 Starting GustyAI Backend Server...")
    print("📡 Chat API: http://localhost:5000/api/chat")
    print("🏥 Health Check: http://localhost:5000/health")
    print("🔑 API Key configured for gustyai model")
    print("🤖 Using GenAI system's built-in knowledge base")
    print("✨ Ready to chat!")
    
    app.run(debug=True, port=5000, host='localhost')