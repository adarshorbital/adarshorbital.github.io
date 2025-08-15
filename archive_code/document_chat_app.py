import gradio as gr
import pandas as pd
import PyPDF2
import requests
import json
from typing import List, Dict, Any
import os
from io import BytesIO
import tempfile

class DocumentChatApp:
    def __init__(self, api_endpoint: str, api_key: str):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.context_data = []
        self.chat_history = []
        # System prompt defining the assistant persona and behavior
        self.system_prompt = (
            "You are a professor at Purdue University, School of Aeronautics and Astronautics, in fluid mechanics and aerodynamics teaching undergraduate lab AAE 33301 (Fluid Mechanics Lab) and AAE 33401 (Aerodynamics Lab). "
            "Student will ask questions to you about technical matter, but don't solve homework or assignment problems for them. Use the knowledge base I provided to you. "
            "Also, when provided, use the list of students, with their assigned days, times, groups and TAs, and help students with their questions related to course logistics, and the teams they belong to. "
            "You will also answer questions related to syllabus and any edge cases. If at any point, you are unsure, give the name and email of the instructor which will be on the syllabus. "
            "Be absolutely courteous and patient with the student, and make sure that they learn!"
        )
    
    def read_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            return f"Error reading PDF: {str(e)}"
    
    def read_excel(self, file_path: str) -> str:
        """Extract data from Excel file and convert to text"""
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            all_data = ""
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                all_data += f"\n--- Sheet: {sheet_name} ---\n"
                all_data += df.to_string(index=False) + "\n"
            
            return all_data
        except Exception as e:
            return f"Error reading Excel: {str(e)}"
    
    def process_uploaded_files(self, files) -> str:
        """Process uploaded files and add to context"""
        if not files:
            return "No files uploaded."
        
        processed_files = []
        
        for file in files:
            file_name = file.name
            file_extension = os.path.splitext(file_name)[1].lower()
            
            try:
                if file_extension == '.pdf':
                    content = self.read_pdf(file.name)
                    self.context_data.append({
                        'filename': file_name,
                        'type': 'PDF',
                        'content': content
                    })
                    processed_files.append(f"✓ {file_name} (PDF)")
                
                elif file_extension in ['.xlsx', '.xls']:
                    content = self.read_excel(file.name)
                    self.context_data.append({
                        'filename': file_name,
                        'type': 'Excel',
                        'content': content
                    })
                    processed_files.append(f"✓ {file_name} (Excel)")
                
                else:
                    processed_files.append(f"✗ {file_name} (Unsupported format)")
            
            except Exception as e:
                processed_files.append(f"✗ {file_name} (Error: {str(e)})")
        
        return "Files processed:\n" + "\n".join(processed_files)
    
    def create_context_prompt(self, user_message: str) -> str:
        """Create a prompt with context from uploaded documents"""
        context_intro = self.system_prompt + "\n\n" + "Use ONLY the following course documents as your knowledge base. If an answer is not in the documents, say you don't have that info and point the student to the instructor listed in the syllabus.\n\n"
        
        if not self.context_data:
            return context_intro + f"Student question: {user_message}"
        
        context_text = context_intro
        context_text += "Documents provided (summarized excerpts):\n\n"
        
        for doc in self.context_data:
            context_text += f"--- {doc['filename']} ({doc['type']}) ---\n"
            # Limit context length to avoid token limits
            content = doc['content'][:4000] + "..." if len(doc['content']) > 4000 else doc['content']
            context_text += content + "\n\n"
        
        context_text += f"Student question: {user_message}\n\n"
        context_text += (
            "Instructions: Answer concisely and cite which document section you used when possible. "
            "If the question is a graded homework/assignment problem, explain the underlying concepts and approach, but do not provide a full solution. "
            "For logistics questions, use the roster/schedule information when provided."
        )
        
        return context_text
    
    def call_llm_api(self, prompt: str) -> str:
        """Make API call to the LLM endpoint"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': 'gpt-oss:latest',  # You may need to adjust this based on available models
            'messages': [
                {
                    'role': 'system',
                    'content': self.system_prompt
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'max_tokens': 1000,
            'temperature': 0.7
        }
        
        try:
            response = requests.post(self.api_endpoint, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
        
        except requests.exceptions.RequestException as e:
            return f"API Error: {str(e)}"
        except KeyError as e:
            return f"Response parsing error: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"
    
    def chat_with_documents(self, message: str, history: List[List[str]]) -> tuple:
        """Handle chat interaction with document context"""
        if not message.strip():
            return history, ""
        
        # Create context-aware prompt
        context_prompt = self.create_context_prompt(message)
        
        # Get response from LLM
        response = self.call_llm_api(context_prompt)
        
        # Update history
        history.append([message, response])
        
        return history, ""
    
    def clear_context(self):
        """Clear all uploaded document context"""
        self.context_data = []
        return "Context cleared. All uploaded documents removed."
    
    def get_context_info(self) -> str:
        """Get information about currently loaded documents"""
        if not self.context_data:
            return "No documents currently loaded."
        
        info = f"Currently loaded documents ({len(self.context_data)}):\n"
        for doc in self.context_data:
            content_length = len(doc['content'])
            info += f"• {doc['filename']} ({doc['type']}) - {content_length} characters\n"
        
        return info

def create_gradio_interface():
    """Create and launch the Gradio interface"""
    
    # Initialize the app with your API credentials
    API_ENDPOINT = "https://genai.rcac.purdue.edu/api/chat/completions"
    API_KEY = "sk-18b41012931c4e96a6e0b3754a567c54"
    
    app = DocumentChatApp(API_ENDPOINT, API_KEY)
    
    with gr.Blocks(title="Document Chat Assistant") as interface:
        gr.Markdown("# 📄 Document Chat Assistant")
        gr.Markdown("Upload PDF and Excel files, then chat with their content using AI!")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 File Upload")
                file_upload = gr.File(
                    label="Upload PDF or Excel files",
                    file_count="multiple",
                    file_types=[".pdf", ".xlsx", ".xls"]
                )
                upload_btn = gr.Button("Process Files", variant="primary")
                upload_status = gr.Textbox(
                    label="Upload Status",
                    interactive=False,
                    max_lines=5
                )
                
                gr.Markdown("### 📋 Context Management")
                context_info = gr.Textbox(
                    label="Loaded Documents",
                    interactive=False,
                    max_lines=5
        
        # Auto-load any files that may have been pre-attached into context (server-side)
        # Note: In this environment, uploaded PDFs via the chat won't auto-populate here.
        # Use the Upload panel to process them into the context.
                )
                refresh_btn = gr.Button("Refresh Context Info")
                clear_btn = gr.Button("Clear All Context", variant="secondary")
            
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Chat with Documents")
                chatbot = gr.Chatbot(
                    label="Chat History",
                    height=400
                )
                msg = gr.Textbox(
                    label="Your Message",
                    placeholder="Ask questions about your uploaded documents...",
                    lines=2
                )
                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    clear_chat_btn = gr.Button("Clear Chat")
        
        # Event handlers
        upload_btn.click(
            fn=app.process_uploaded_files,
            inputs=[file_upload],
            outputs=[upload_status]
        )
        
        refresh_btn.click(
            fn=app.get_context_info,
            outputs=[context_info]
        )
        
        clear_btn.click(
            fn=app.clear_context,
            outputs=[upload_status]
        )
        
        send_btn.click(
            fn=app.chat_with_documents,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
        
        msg.submit(
            fn=app.chat_with_documents,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
        
        clear_chat_btn.click(
            fn=lambda: ([], ""),
            outputs=[chatbot, msg]
        )
        
        # Load context info on startup
        interface.load(
            fn=app.get_context_info,
            outputs=[context_info]
        )
    
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True if you want a public link
        debug=True
    )