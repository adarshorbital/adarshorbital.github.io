import gradio as gr
import pandas as pd
import PyPDF2
import requests
import json
import os
from typing import List, Dict, Any
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

class LabAssistant:
    def __init__(self):
        self.api_url = "https://genai.rcac.purdue.edu/api/chat/completions"
        self.api_key = "sk-18b41012931c4e96a6e0b3754a567c54"
        self.context_documents = []
        self.vector_store = None
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        self.system_prompt = """You are a professor at Purdue University, School of Aeronautics and Astronautics, in fluid mechanics and aerodynamics teaching undergraduate lab AAE 33301 (Fluid Mechanics Lab) and AAE 33401 (Aerodynamics Lab). Student will ask questions to you about technical matter, but don't solve homework or assignment problems for them. Use the knowledge base I provided to you. Also, when provided, use the list of students, with their assigned days, times, groups and TAs, and help students with their questions related to course logistics, and the teams they belong to. You will also answer questions related to syllabus and any edge cases. If at any point, you are unsure, give the name and email of the instructor which will be on the syllabus. Be absolutely courteous and patient with the student, and make sure that they learn!"""

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
        """Extract text from Excel file"""
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            all_text = ""
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                all_text += f"\n--- Sheet: {sheet_name} ---\n"
                all_text += df.to_string(index=False) + "\n"
            
            return all_text
        except Exception as e:
            return f"Error reading Excel: {str(e)}"

    def process_files(self, files: List[Any]) -> str:
        """Process uploaded files and add to context"""
        if not files:
            return "No files uploaded."
        
        processed_files = []
        new_documents = []
        
        for file in files:
            file_path = file.name
            file_name = os.path.basename(file_path)
            
            try:
                if file_name.lower().endswith('.pdf'):
                    content = self.read_pdf(file_path)
                    processed_files.append(f"PDF: {file_name}")
                elif file_name.lower().endswith(('.xlsx', '.xls')):
                    content = self.read_excel(file_path)
                    processed_files.append(f"Excel: {file_name}")
                else:
                    continue
                
                # Create document and split into chunks
                doc = Document(page_content=content, metadata={"source": file_name})
                chunks = self.text_splitter.split_documents([doc])
                new_documents.extend(chunks)
                
            except Exception as e:
                processed_files.append(f"Error processing {file_name}: {str(e)}")
        
        # Add new documents to context
        self.context_documents.extend(new_documents)
        
        # Create or update vector store
        if self.context_documents:
            self.vector_store = FAISS.from_documents(self.context_documents, self.embeddings)
        
        return f"Processed files: {', '.join(processed_files)}\nTotal documents in context: {len(self.context_documents)}"

    def get_relevant_context(self, query: str, k: int = 5) -> str:
        """Retrieve relevant context for the query"""
        if not self.vector_store:
            return "No context available. Please upload files first."
        
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            context = "\n\n".join([doc.page_content for doc in docs])
            return context
        except Exception as e:
            return f"Error retrieving context: {str(e)}"

    def call_llm_api(self, messages: List[Dict[str, str]]) -> str:
        """Make API call to the LLM"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-oss:latest",  # You may need to adjust this based on available models
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            return f"API Error: {str(e)}"
        except KeyError as e:
            return f"Response parsing error: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

    def chat_response(self, message: str, history: List[List[str]]) -> str:
        """Generate response for chat interface"""
        if not message.strip():
            return "Please enter a question."
        
        # Get relevant context
        context = self.get_relevant_context(message)
        
        # Prepare messages for API
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": f"Context from uploaded documents:\n{context}"}
        ]
        
        # Add conversation history
        for user_msg, assistant_msg in history[-5:]:  # Last 5 exchanges
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Get response from LLM
        response = self.call_llm_api(messages)
        return response

# Initialize the assistant
lab_assistant = LabAssistant()

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="AAE Lab Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🚀 AAE Lab Assistant
        ### Purdue University - School of Aeronautics and Astronautics
        
        Upload your lab manuals, student lists, and other course materials, then ask questions about:
        - **AAE 33301** (Fluid Mechanics Lab)
        - **AAE 33401** (Aerodynamics Lab)
        - Course logistics, schedules, and team assignments
        - Technical concepts and procedures
        
        *Note: I won't solve homework problems, but I'm here to help you learn!*
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Upload Course Materials")
                file_upload = gr.File(
                    label="Upload PDF and Excel files",
                    file_count="multiple",
                    file_types=[".pdf", ".xlsx", ".xls"]
                )
                upload_btn = gr.Button("Process Files", variant="primary")
                upload_status = gr.Textbox(
                    label="Upload Status",
                    interactive=False,
                    max_lines=5
                )
            
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Chat with Your Lab Assistant")
                chatbot = gr.Chatbot(
                    height=400,
                    label="Lab Assistant Chat",
                    show_label=True
                )
                msg = gr.Textbox(
                    label="Ask your question",
                    placeholder="e.g., What is the procedure for Lab 1 in AAE 33301?",
                    lines=2
                )
                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear Chat", variant="secondary")
        
        # Event handlers
        upload_btn.click(
            fn=lab_assistant.process_files,
            inputs=[file_upload],
            outputs=[upload_status]
        )
        
        def respond(message, chat_history):
            if not message.strip():
                return chat_history, ""
            
            response = lab_assistant.chat_response(message, chat_history)
            chat_history.append([message, response])
            return chat_history, ""
        
        submit_btn.click(
            fn=respond,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
        
        msg.submit(
            fn=respond,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
        
        clear_btn.click(
            fn=lambda: [],
            outputs=[chatbot]
        )
        
        gr.Markdown("""
        ---
        ### 📋 Instructions:
        1. **Upload Files**: Upload your lab manuals (PDF) and student lists (Excel) using the file uploader
        2. **Process Files**: Click "Process Files" to add them to the knowledge base
        3. **Ask Questions**: Use the chat interface to ask questions about the labs
        
        ### 🎯 Example Questions:
        - "What equipment is used in the wind tunnel lab?"
        - "When is my lab session scheduled?"
        - "What are the safety procedures for Lab 3?"
        - "Who is my TA for AAE 33401?"
        """)
    
    return demo

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )