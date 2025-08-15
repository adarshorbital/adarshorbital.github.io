import os
import re
import tempfile
from typing import List, Optional, Generator
import gradio as gr

# LangChain core
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

# Loaders and retrievers
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever

# LLMs (OpenAI)
from langchain_openai import ChatOpenAI

# Chains
from langchain.chains import ConversationalRetrievalChain

# --------------------------
# Config
# --------------------------
LLM_MODEL = "llama3.3:70b"  # Purdue RCAC model
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVE_K = 4
TEMPERATURE = 0.0  # for factual answers

# Custom OpenAI-compatible endpoint (Purdue RCAC)
PURDUE_API_BASE = os.getenv("PURDUE_API_BASE", "https://genai.rcac.purdue.edu/api")
PURDUE_API_KEY = os.getenv("PURDUE_API_KEY", "sk-18b41012931c4e96a6e0b3754a567c54")

# --------------------------
# Streaming Callback Handler
# --------------------------
class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.tokens = []
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.tokens.append(token)

# --------------------------
# Prompts (persona + answer strictly from context)
# --------------------------
QA_SYSTEM_PROMPT = (
    "You are a professor at Purdue University, School of Aeronautics and Astronautics, "
    "in fluid mechanics and aerodynamics teaching undergraduate lab AAE 33301 (Fluid Mechanics Lab) "
    "and AAE 33401 (Aerodynamics Lab). Students will ask technical questions, but do NOT solve "
    "homework or assignment problems. Use ONLY the provided context (knowledge base). When provided, "
    "use the list of students with their assigned days/times/groups/TAs to help with logistics and team-related "
    "questions. Also answer syllabus-related questions and edge cases based on the context. If the answer is not in the context, say you don't know. "
    "If you are unsure and the syllabus is expected to contain the information (like instructor name/email), respond courteously that the information "
    "should be in the syllabus section of the provided materials and cannot be confirmed from the current context. Be absolutely courteous and patient, and facilitate learning."
)

CONV_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template=(
        "System: {system_instructions}\n\n"
        "Use ONLY the context below. If missing, say you don't know. Be courteous and educational.\n\n"
        "Chat history:\n{chat_history}\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
    partial_variables={"system_instructions": QA_SYSTEM_PROMPT},
)

# --------------------------
# Global variables for pipeline
# --------------------------
global_pipeline = None

# --------------------------
# Helpers
# --------------------------

def clean_text(text: str) -> str:
    """Clean text by removing hyphenations, normalizing whitespace."""
    # Remove hyphenations across line breaks: e.g., "exam-\nple" -> "example"
    text = re.sub(r"-\s*\n\s*", "", text)
    # Normalize newlines to spaces
    text = re.sub(r"\s*\n\s*", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def load_pdf_from_file(file_path: str) -> List[Document]:
    """Load PDF from file path and clean text."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()  # returns one Document per page with metadata={"page": int, ...}
    # Clean page-by-page
    for d in docs:
        d.page_content = clean_text(d.page_content)
    return docs

def chunk_docs(docs: List[Document], chunk_size=1000, overlap=150) -> List[Document]:
    """Chunk documents using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)

def build_bm25_retriever(chunks: List[Document], k: int = 4) -> BM25Retriever:
    """Build BM25 keyword-based retriever from chunks."""
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = k
    return retriever

def make_conversational_chain(retriever, llm):
    """Create ConversationalRetrievalChain with custom prompt."""
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        verbose=False,
        combine_docs_chain_kwargs={"prompt": CONV_QA_PROMPT},
    )

def build_pipeline_from_pdf(pdf_file) -> dict:
    """Build the complete RAG pipeline from uploaded PDF."""
    if pdf_file is None:
        return None
        
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file)
        tmp_path = tmp_file.name

    try:
        # 1) Load PDF and clean
        print("Loading and cleaning PDF...")
        docs = load_pdf_from_file(tmp_path)

        # 2) Chunk
        print("Chunking...")
        chunks = chunk_docs(docs, CHUNK_SIZE, CHUNK_OVERLAP)

        # 3) Build BM25 retriever
        print("Building BM25 retriever...")
        retriever = build_bm25_retriever(chunks, k=RETRIEVE_K)

        # 4) LLM with streaming
        print("Initializing LLM...")
        llm = ChatOpenAI(
            model=LLM_MODEL, 
            temperature=TEMPERATURE, 
            api_key=PURDUE_API_KEY, 
            base_url=PURDUE_API_BASE,
            streaming=True
        )

        # 5) Conversational chain
        print("Building conversational chain...")
        conv_qa = make_conversational_chain(retriever, llm)

        return {
            "docs": docs,
            "chunks": chunks,
            "retriever": retriever,
            "conv_qa": conv_qa,
            "llm": llm
        }
    
    finally:
        # Clean up temporary file
        os.unlink(tmp_path)

# --------------------------
# Gradio Interface Functions
# --------------------------

def upload_pdf(pdf_file):
    """Handle PDF upload and build pipeline."""
    global global_pipeline
    
    if pdf_file is None:
        return "❌ Please upload a PDF file first.", ""
    
    try:
        # Read the uploaded file
        with open(pdf_file.name, 'rb') as f:
            pdf_content = f.read()
        
        # Build pipeline
        global_pipeline = build_pipeline_from_pdf(pdf_content)
        
        if global_pipeline is None:
            return "❌ Failed to process PDF. Please try again.", ""
        
        num_chunks = len(global_pipeline["chunks"])
        num_pages = len(global_pipeline["docs"])
        
        success_msg = f"✅ PDF processed successfully!\n📄 Pages: {num_pages}\n📝 Chunks: {num_chunks}\n\nYou can now ask questions about the document."
        
        return success_msg, ""
        
    except Exception as e:
        return f"❌ Error processing PDF: {str(e)}", ""

def chat_with_pdf(message, history):
    """Handle chat with streaming response."""
    global global_pipeline
    
    if global_pipeline is None:
        yield history + [["Please upload a PDF file first.", ""]]
        return
    
    if not message.strip():
        yield history + [["Please enter a question.", ""]]
        return
    
    # Add user message to history
    history = history + [[message, ""]]
    
    try:
        # Convert history to the format expected by ConversationalRetrievalChain
        chat_history = []
        for i in range(0, len(history) - 1):  # Exclude current question
            if len(history[i]) == 2 and history[i][1]:  # Valid Q&A pair
                chat_history.append((history[i][0], history[i][1]))
        
        # Create streaming callback
        streaming_handler = StreamingCallbackHandler()
        
        # Get response with streaming
        result = global_pipeline["conv_qa"](
            {
                "question": message, 
                "chat_history": chat_history
            },
            callbacks=[streaming_handler]
        )
        
        # Stream the response token by token
        response = ""
        for token in streaming_handler.tokens:
            response += token
            history[-1][1] = response
            yield history
            
        # If no streaming tokens, use the full result
        if not streaming_handler.tokens:
            history[-1][1] = result["answer"]
            yield history
            
    except Exception as e:
        history[-1][1] = f"❌ Error: {str(e)}"
        yield history

def clear_chat():
    """Clear chat history."""
    return []

def get_example_questions():
    """Return example questions for the interface."""
    return [
        "What are the main topics covered in this document?",
        "What are the course requirements?",
        "How is the grading structured?",
        "What lab equipment is mentioned?",
        "What are the safety procedures?",
        "Who is the instructor for this course?",
        "What are the office hours?",
        "What is the attendance policy?"
    ]

# --------------------------
# Gradio Interface
# --------------------------

def create_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks(
        title="Purdue AAE Lab Assistant",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .upload-area {
            border: 2px dashed #ccc;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            margin: 10px 0;
        }
        """
    ) as interface:
        
        gr.Markdown(
            """
            # 🚀 Purdue AAE Lab Assistant
            
            **Your AI Teaching Assistant for Fluid Mechanics & Aerodynamics Labs**
            
            Upload your course PDF (syllabus, lab manual, etc.) and ask questions about:
            - Course structure and requirements
            - Lab procedures and equipment
            - Grading policies
            - Logistics and schedules
            - Safety procedures
            
            *Note: I won't solve homework problems, but I'll help you learn!*
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                # PDF Upload Section
                gr.Markdown("## 📁 Upload Course Materials")
                pdf_upload = gr.File(
                    label="Upload PDF",
                    file_types=[".pdf"],
                    type="filepath"
                )
                upload_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=4
                )
                upload_btn = gr.Button("Process PDF", variant="primary")
                
                # Example Questions
                gr.Markdown("## 💡 Example Questions")
                example_questions = get_example_questions()
                for i, question in enumerate(example_questions[:4]):  # Show first 4
                    gr.Markdown(f"• {question}")
                
            with gr.Column(scale=2):
                # Chat Section
                gr.Markdown("## 💬 Chat with Your Documents")
                chatbot = gr.Chatbot(
                    height=500,
                    show_label=False,
                    container=True,
                    bubble_full_width=False
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask a question about your uploaded document...",
                        show_label=False,
                        scale=4
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("Clear Chat", variant="secondary")
        
        # Event handlers
        upload_btn.click(
            fn=upload_pdf,
            inputs=[pdf_upload],
            outputs=[upload_status, msg_input]
        )
        
        # Chat functionality
        msg_input.submit(
            fn=chat_with_pdf,
            inputs=[msg_input, chatbot],
            outputs=[chatbot]
        ).then(
            lambda: "",  # Clear input
            outputs=[msg_input]
        )
        
        send_btn.click(
            fn=chat_with_pdf,
            inputs=[msg_input, chatbot],
            outputs=[chatbot]
        ).then(
            lambda: "",  # Clear input
            outputs=[msg_input]
        )
        
        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot]
        )
        
        # Example question buttons
        gr.Markdown("## 🎯 Quick Questions")
        with gr.Row():
            for question in example_questions[4:8]:  # Show next 4 as buttons
                btn = gr.Button(question, size="sm")
                btn.click(
                    lambda q=question: q,
                    outputs=[msg_input]
                )
    
    return interface

# --------------------------
# Main
# --------------------------

if __name__ == "__main__":
    # Ensure API key is set
    if not PURDUE_API_KEY:
        raise RuntimeError("Please set PURDUE_API_KEY environment variable or update the hardcoded key.")
    
    print("🚀 Starting Purdue AAE Lab Assistant...")
    print(f"📡 Using API: {PURDUE_API_BASE}")
    print(f"🤖 Model: {LLM_MODEL}")
    
    # Create and launch interface
    interface = create_interface()
    
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True for public sharing
        debug=True,
        show_error=True
    )