# gustyai_backend.py - Backend proxy for GustyAI with RAG knowledge base
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
from knowledge_base import initialize_knowledge_base, search_pdfs

app = Flask(__name__)
CORS(app)  # Allow your website to call this backend

# Your Purdue GenAI API configuration
API_KEY = "sk-18b41012931c4e96a6e0b3754a567c54"  # Replace with your real API key
API_URL = "https://genai.rcac.purdue.edu/api/chat/completions"

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    GustyAI chat endpoint - safely calls Purdue GenAI API
    """
    try:
        # Check if API key is configured
        if API_KEY == "YOUR_ACTUAL_API_KEY_HERE":
            return jsonify({'error': 'Please set your API key in gustyai_backend.py'}), 500
            
        # Get the message data from your website
        user_data = request.json
        
        # Validate the request
        if not user_data or 'messages' not in user_data:
            return jsonify({'error': 'Invalid request format'}), 400

        # Get the latest user message for knowledge base search
        messages = user_data.get('messages', [])
        latest_user_message = ""
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                latest_user_message = msg.get('content', '')
                break
        
        # Search knowledge base for relevant context
        context = ""
        if latest_user_message:
            context = search_pdfs(latest_user_message)
        
        # Add GustyAI personality and knowledge base context to the conversation
        if not any(msg.get('role') == 'system' for msg in messages):
            system_content = """You are GustyAI, an expert AI assistant specializing in aerospace engineering, fluid mechanics, and aerodynamics. You help students and engineers with technical questions. Be helpful, accurate, and educational.

When answering questions, prioritize information from the knowledge base if it's relevant. Always cite your sources when using information from the knowledge base."""
            
            if context:
                system_content += f"\n\n{context}"
            
            system_message = {
                "role": "system",
                "content": system_content
            }
            messages.insert(0, system_message)
            user_data['messages'] = messages

        # Call Purdue GenAI API
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
            return jsonify({'error': f'API request failed: {response.status_code}'}), response.status_code
            
    except requests.exceptions.Timeout:
        return jsonify({'error': 'Request timeout - try again'}), 504
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Check if the backend is running"""
    return jsonify({'status': 'GustyAI Backend is running!'}), 200

if __name__ == '__main__':
    print("🚀 Starting GustyAI Backend Server...")
    print("📡 Chat API: http://localhost:5000/api/chat")
    print("🏥 Health Check: http://localhost:5000/health")
    print("⚠️  Make sure to set your API key in the code!")
    
    # Initialize knowledge base on startup
    initialize_knowledge_base()
    
    app.run(debug=True, port=5000, host='localhost')