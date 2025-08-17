# gustyai_integrated.py - Integrated GustyAI with Gradio frontend and backend
import gradio as gr
import requests
import json
import time
import base64
from io import BytesIO

# API Configuration
API_KEY = "sk-18b41012931c4e96a6e0b3754a567c54"
API_URL = "https://genai.rcac.purdue.edu/api/chat/completions"

def call_gustyai_api(messages):
    """Direct API call to GustyAI"""
    try:
        response = requests.post(
            API_URL,
            headers={
                'Authorization': f'Bearer {API_KEY}',
                'Content-Type': 'application/json'
            },
            json={
                "messages": messages,
                "model": "gustyai",
                "max_tokens": 2000,
                "temperature": 0.7
            },
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"], None
        else:
            return None, f"API Error {response.status_code}: {response.text}"
            
    except requests.exceptions.Timeout:
        return None, "Request timeout. Please try again."
    except Exception as e:
        return None, f"Error: {str(e)}"

def chat_with_gustyai(message, history):
    """Process chat message and return response"""
    if not message.strip():
        return history, ""
    
    # Convert Gradio history to OpenAI format
    messages = []
    for chat_message in history:
        if chat_message["role"] == "user":
            messages.append({"role": "user", "content": chat_message["content"]})
        elif chat_message["role"] == "assistant":
            messages.append({"role": "assistant", "content": chat_message["content"]})
    
    # Add current message
    messages.append({"role": "user", "content": message})
    
    # Get response from API
    response, error = call_gustyai_api(messages)
    
    if error:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"❌ **Error:** {error}"})
    else:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
    
    return history, ""

def clear_chat():
    """Clear chat history"""
    return [], ""

def get_example_queries():
    """Return example aerospace engineering queries"""
    return [
        "Explain the Navier-Stokes equations and their significance in fluid mechanics",
        "What is the difference between subsonic and supersonic flow?",
        "How do you calculate the lift coefficient for an airfoil using thin airfoil theory?",
        "Describe boundary layer separation and its effects on aerodynamic performance",
        "What are the governing equations for compressible flow?",
        "Explain the Magnus effect and provide real-world applications",
        "How does Reynolds number affect flow characteristics?",
        "What is the Prandtl-Glauert transformation and when is it used?",
        "Describe the working principles of a wind tunnel",
        "What are the key differences between RANS and LES in CFD?"
    ]

# Enhanced CSS for full-page sleek design
custom_css = """
/* Global Styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html, body {
    height: 100%;
    overflow: hidden;
}

.gradio-container {
    background: linear-gradient(135deg, #000000 0%, #1a1a1a 50%, #2d2d2d 100%) !important;
    color: #CFB991 !important;
    height: 100vh !important;
    max-height: 100vh !important;
    padding: 0 !important;
    margin: 0 !important;
}

.main {
    height: 100vh !important;
    max-height: 100vh !important;
    padding: 0 !important;
    margin: 0 !important;
}

.container {
    height: 100vh !important;
    max-height: 100vh !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Header Styles */
.header-container {
    background: rgba(207, 185, 145, 0.1) !important;
    backdrop-filter: blur(20px) !important;
    border-bottom: 2px solid rgba(207, 185, 145, 0.3) !important;
    padding: 15px 20px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    min-height: 80px !important;
    position: sticky !important;
    top: 0 !important;
    z-index: 1000 !important;
}

.logo-section {
    display: flex !important;
    align-items: center !important;
    gap: 15px !important;
}

.logo-img {
    width: 60px !important;
    height: 60px !important;
    border-radius: 50% !important;
    background: rgba(207, 185, 145, 0.2) !important;
    padding: 8px !important;
}

.title-section h1 {
    color: #CFB991 !important;
    font-size: 2.2em !important;
    font-weight: 700 !important;
    margin: 0 !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.5) !important;
}

.title-section p {
    color: #999 !important;
    font-size: 0.95em !important;
    margin: 2px 0 0 0 !important;
    font-weight: 300 !important;
}

.status-indicator {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    padding: 8px 16px !important;
    background: rgba(0, 255, 68, 0.1) !important;
    border: 1px solid rgba(0, 255, 68, 0.3) !important;
    border-radius: 25px !important;
    backdrop-filter: blur(10px) !important;
}

.status-dot {
    width: 12px !important;
    height: 12px !important;
    background: #00ff44 !important;
    border-radius: 50% !important;
    animation: pulse 2s infinite !important;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* Chat Interface */
.chat-container {
    height: calc(100vh - 80px) !important;
    display: flex !important;
    flex-direction: column !important;
    padding: 0 !important;
    margin: 0 !important;
}

.gr-chatbot {
    flex: 1 !important;
    background: transparent !important;
    border: none !important;
    height: calc(100vh - 200px) !important;
    max-height: calc(100vh - 200px) !important;
    overflow-y: auto !important;
    padding: 20px !important;
}

.gr-chatbot .message {
    background: rgba(207, 185, 145, 0.08) !important;
    border: 1px solid rgba(207, 185, 145, 0.2) !important;
    border-radius: 15px !important;
    margin: 10px 0 !important;
    padding: 15px 20px !important;
    backdrop-filter: blur(10px) !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
}

.gr-chatbot .message.user {
    background: linear-gradient(135deg, #CFB991 0%, #B8A082 100%) !important;
    color: #000000 !important;
    margin-left: 20% !important;
    border-bottom-right-radius: 5px !important;
}

.gr-chatbot .message.bot {
    background: rgba(207, 185, 145, 0.1) !important;
    color: #CFB991 !important;
    margin-right: 20% !important;
    border-bottom-left-radius: 5px !important;
}

/* Input Area */
.input-area {
    background: rgba(207, 185, 145, 0.05) !important;
    border-top: 1px solid rgba(207, 185, 145, 0.2) !important;
    padding: 20px !important;
    backdrop-filter: blur(20px) !important;
}

.gr-textbox {
    background: rgba(0, 0, 0, 0.7) !important;
    border: 2px solid rgba(207, 185, 145, 0.3) !important;
    border-radius: 15px !important;
    color: #CFB991 !important;
    padding: 15px 20px !important;
    font-size: 1.1em !important;
    backdrop-filter: blur(10px) !important;
    transition: all 0.3s ease !important;
}

.gr-textbox:focus {
    border-color: #CFB991 !important;
    box-shadow: 0 0 20px rgba(207, 185, 145, 0.3) !important;
    outline: none !important;
}

.gr-button-primary {
    background: linear-gradient(135deg, #CFB991 0%, #B8A082 100%) !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 15px !important;
    padding: 15px 30px !important;
    font-weight: 700 !important;
    font-size: 1.1em !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(207, 185, 145, 0.3) !important;
}

.gr-button-primary:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 25px rgba(207, 185, 145, 0.4) !important;
}

.gr-button-secondary {
    background: rgba(207, 185, 145, 0.1) !important;
    color: #CFB991 !important;
    border: 1px solid rgba(207, 185, 145, 0.3) !important;
    border-radius: 15px !important;
    padding: 12px 25px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    backdrop-filter: blur(10px) !important;
}

.gr-button-secondary:hover {
    background: rgba(207, 185, 145, 0.2) !important;
    transform: translateY(-2px) !important;
}

/* Sidebar */
.sidebar {
    background: rgba(207, 185, 145, 0.05) !important;
    border-left: 1px solid rgba(207, 185, 145, 0.2) !important;
    padding: 20px !important;
    backdrop-filter: blur(20px) !important;
    height: calc(100vh - 80px) !important;
    overflow-y: auto !important;
}

.feature-box {
    background: rgba(207, 185, 145, 0.08) !important;
    border: 1px solid rgba(207, 185, 145, 0.2) !important;
    border-radius: 15px !important;
    padding: 20px !important;
    margin-bottom: 20px !important;
    backdrop-filter: blur(10px) !important;
}

.feature-box h3 {
    color: #CFB991 !important;
    margin-bottom: 15px !important;
    font-size: 1.3em !important;
}

.feature-box ul {
    color: #999 !important;
    font-size: 0.95em !important;
    line-height: 1.6 !important;
}

.feature-box li {
    margin-bottom: 8px !important;
}

.gr-dropdown {
    background: rgba(0, 0, 0, 0.7) !important;
    border: 1px solid rgba(207, 185, 145, 0.3) !important;
    border-radius: 10px !important;
    color: #CFB991 !important;
    backdrop-filter: blur(10px) !important;
}

/* Math and Table Styling */
.katex {
    font-size: 1.2em !important;
    color: #CFB991 !important;
}

.katex-display {
    margin: 1.5em 0 !important;
    text-align: center !important;
}

table {
    border-collapse: collapse !important;
    width: 100% !important;
    margin: 1.5em 0 !important;
    background: rgba(207, 185, 145, 0.05) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

table th, table td {
    border: 1px solid rgba(207, 185, 145, 0.3) !important;
    padding: 12px 15px !important;
    text-align: left !important;
}

table th {
    background: rgba(207, 185, 145, 0.2) !important;
    font-weight: bold !important;
    color: #000 !important;
}

pre {
    background: rgba(0, 0, 0, 0.8) !important;
    border: 1px solid rgba(207, 185, 145, 0.3) !important;
    border-radius: 10px !important;
    padding: 15px !important;
    overflow-x: auto !important;
    margin: 1em 0 !important;
}

code {
    background: rgba(0, 0, 0, 0.5) !important;
    padding: 3px 6px !important;
    border-radius: 5px !important;
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    color: #CFB991 !important;
}

/* Mobile Responsiveness */
@media (max-width: 768px) {
    .header-container {
        flex-direction: column !important;
        gap: 10px !important;
        padding: 15px !important;
        min-height: 120px !important;
    }
    
    .title-section h1 {
        font-size: 1.8em !important;
        text-align: center !important;
    }
    
    .logo-img {
        width: 50px !important;
        height: 50px !important;
    }
    
    .chat-container {
        height: calc(100vh - 120px) !important;
    }
    
    .gr-chatbot {
        height: calc(100vh - 240px) !important;
        max-height: calc(100vh - 240px) !important;
        padding: 15px !important;
    }
    
    .gr-chatbot .message {
        margin: 8px 0 !important;
        padding: 12px 15px !important;
    }
    
    .gr-chatbot .message.user {
        margin-left: 10% !important;
    }
    
    .gr-chatbot .message.bot {
        margin-right: 10% !important;
    }
    
    .input-area {
        padding: 15px !important;
    }
    
    .sidebar {
        display: none !important;
    }
}

@media (max-width: 480px) {
    .header-container {
        min-height: 100px !important;
    }
    
    .title-section h1 {
        font-size: 1.5em !important;
    }
    
    .title-section p {
        font-size: 0.85em !important;
    }
    
    .gr-chatbot .message.user,
    .gr-chatbot .message.bot {
        margin-left: 5% !important;
        margin-right: 5% !important;
    }
}

/* Scrollbar Styling */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(207, 185, 145, 0.1);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: rgba(207, 185, 145, 0.3);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(207, 185, 145, 0.5);
}
"""

def create_interface():
    """Create the integrated Gradio interface"""
    
    with gr.Blocks(
        title="GustyAI - Purdue AAE Assistant",
        theme=gr.themes.Base(),
        css=custom_css,
        fill_height=True
    ) as interface:
        
        # Header with logo and status
        gr.HTML("""
        <div class="header-container">
            <div class="logo-section">
                <img src="https://engineering.purdue.edu/AAE/images/AAE_logo.png" alt="Purdue AAE Logo" class="logo-img" onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGNpcmNsZSBjeD0iMzAiIGN5PSIzMCIgcj0iMzAiIGZpbGw9IiNDRkI5OTEiLz4KPHN2ZyB4PSIxNSIgeT0iMTUiIHdpZHRoPSIzMCIgaGVpZ2h0PSIzMCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSIjMDAwIj4KPHA+QUFFPC9wPgo8L3N2Zz4KPC9zdmc+'"/>
                <div class="title-section">
                    <h1>🌪️ GustyAI</h1>
                    <p>Purdue School of Aeronautics & Astronautics</p>
                </div>
            </div>
            <div class="status-indicator">
                <div class="status-dot"></div>
                <span>Connected</span>
            </div>
        </div>
        """)
        
        # Main chat interface
        with gr.Row(elem_classes="chat-container"):
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    value=[],
                    height="calc(100vh - 200px)",
                    show_label=False,
                    container=False,
                    render_markdown=True,
                    type="messages",
                    latex_delimiters=[
                        {"left": "$$", "right": "$$", "display": True},
                        {"left": "$", "right": "$", "display": False},
                        {"left": "\\(", "right": "\\)", "display": False},
                        {"left": "\\[", "right": "\\]", "display": True},
                    ],
                    elem_classes="chat-messages"
                )
                
                with gr.Row(elem_classes="input-area"):
                    with gr.Column(scale=5):
                        msg_input = gr.Textbox(
                            placeholder="Ask me about aerospace engineering, fluid mechanics, aerodynamics, or any technical question...",
                            show_label=False,
                            lines=2,
                            max_lines=6,
                            container=False
                        )
                    with gr.Column(scale=1, min_width=120):
                        send_btn = gr.Button("Send", variant="primary", size="lg")
                
                with gr.Row():
                    clear_btn = gr.Button("Clear Chat", variant="secondary", size="sm", scale=1)
                    gr.HTML("<div style='flex: 1;'></div>")  # Spacer
            
            # Sidebar (hidden on mobile)
            with gr.Column(scale=1, elem_classes="sidebar", visible=True):
                gr.HTML("""
                <div class="feature-box">
                    <h3>✨ Capabilities</h3>
                    <ul>
                        <li>🧮 LaTeX equations: $F = ma$</li>
                        <li>📊 Tables & data analysis</li>
                        <li>💻 Code & algorithms</li>
                        <li>🔬 Research assistance</li>
                        <li>📚 Course material help</li>
                        <li>🛩️ Aircraft design</li>
                        <li>🌊 CFD & flow analysis</li>
                    </ul>
                </div>
                """)
                
                example_dropdown = gr.Dropdown(
                    choices=get_example_queries(),
                    label="📚 Example Questions",
                    value=None,
                    interactive=True,
                    container=True
                )
                
                use_example_btn = gr.Button("Use Example", variant="secondary", size="sm")
                
                gr.HTML("""
                <div class="feature-box">
                    <h3>🎯 Specializations</h3>
                    <ul>
                        <li>Fluid Mechanics</li>
                        <li>Aerodynamics</li>
                        <li>Propulsion</li>
                        <li>Flight Dynamics</li>
                        <li>Structures</li>
                        <li>Controls</li>
                    </ul>
                </div>
                """)
        
        # Event handlers
        def use_example(example):
            return example if example else ""
        
        # Set up interactions
        send_btn.click(
            chat_with_gustyai,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input]
        )
        
        msg_input.submit(
            chat_with_gustyai,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input]
        )
        
        clear_btn.click(
            clear_chat,
            outputs=[chatbot, msg_input]
        )
        
        use_example_btn.click(
            use_example,
            inputs=[example_dropdown],
            outputs=[msg_input]
        )
        
        # Welcome message
        interface.load(
            lambda: [["", """👋 **Welcome to GustyAI!**

I'm your dedicated aerospace engineering assistant from Purdue's School of Aeronautics & Astronautics. I specialize in:

### 🚀 **Core Areas**
- **Fluid Mechanics**: Navier-Stokes equations, boundary layers, turbulence modeling
- **Aerodynamics**: Airfoil theory, compressible flow, supersonic aerodynamics  
- **Propulsion**: Jet engines, rocket propulsion, combustion
- **Flight Dynamics**: Stability, control, performance analysis

### 🧮 **Advanced Features**
- **Mathematical Analysis**: I can render complex equations like $\\frac{\\partial \\rho}{\\partial t} + \\nabla \\cdot (\\rho \\vec{v}) = 0$
- **Data Tables**: Organized presentation of engineering data
- **Code Examples**: MATLAB, Python, and other programming assistance

### 📚 **How I Can Help**
- Solve homework problems step-by-step
- Explain complex engineering concepts
- Assist with research projects
- Review calculations and derivations
- Provide design guidance

Feel free to ask me anything about aerospace engineering! Try the example questions in the sidebar or ask your own questions.
"""]],
            outputs=[chatbot]
        )
    
    return interface

if __name__ == "__main__":
    print("🚀 Starting Integrated GustyAI Application...")
    print("🎓 Purdue School of Aeronautics & Astronautics")
    print("🔗 Direct API integration - no separate backend needed")
    print("📱 Mobile-responsive design with full-page layout")
    print("🧮 LaTeX math and markdown support enabled")
    
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False,
        favicon_path=None
    )