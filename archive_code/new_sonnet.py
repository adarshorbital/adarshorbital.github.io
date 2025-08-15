import os
import re
from typing import List, Optional

# LangChain core
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# Loaders and retrievers
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever

# LLMs (OpenAI)
from langchain_openai import ChatOpenAI

# Chains
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain

# --------------------------
# Config
# --------------------------
PDF_PATH = "AAE_33301_Lab_Manual_Draft.pdf"
LLM_MODEL = "gpt-oss:latest"  # Purdue RCAC model
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVE_K = 4
TEMPERATURE = 0.0  # for factual answers

# Custom OpenAI-compatible endpoint (Purdue RCAC)
PURDUE_API_BASE = os.getenv("PURDUE_API_BASE", "https://genai.rcac.purdue.edu/api")
PURDUE_API_KEY = os.getenv("PURDUE_API_KEY", "sk-18b41012931c4e96a6e0b3754a567c54")

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

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "System: {system_instructions}\n\n"
        "You must answer ONLY from the context below. If the context does not contain the answer, say you don't know.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer (be concise, cite specific parts of the context when useful):"
    ),
    partial_variables={"system_instructions": QA_SYSTEM_PROMPT},
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

def load_pdf_pagewise(path: str) -> List[Document]:
    """Load PDF page-by-page and clean text."""
    loader = PyPDFLoader(path)
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

def make_qa_chain(retriever, llm):
    """Create RetrievalQA chain with custom prompt."""
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # for very large docs, consider "map_reduce"
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT},
    )

def make_conversational_chain(retriever, llm):
    """Create ConversationalRetrievalChain with custom prompt."""
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        verbose=False,
        combine_docs_chain_kwargs={"prompt": CONV_QA_PROMPT},
    )

# --------------------------
# Main assembly
# --------------------------
def build_pipeline():
    """Build the complete RAG pipeline using BM25 retrieval (no embeddings)."""
    # 1) Load PDF page-by-page and clean
    print("Loading and cleaning PDF...")
    docs = load_pdf_pagewise(PDF_PATH)

    # 2) Chunk
    print("Chunking...")
    chunks = chunk_docs(docs, CHUNK_SIZE, CHUNK_OVERLAP)

    # 3) Build BM25 retriever (keyword-based, no embeddings needed)
    print("Building BM25 retriever...")
    retriever = build_bm25_retriever(chunks, k=RETRIEVE_K)

    # 4) LLM (low temperature for factual accuracy)
    print("Initializing LLM...")
    llm = ChatOpenAI(
        model=LLM_MODEL, 
        temperature=TEMPERATURE, 
        api_key=PURDUE_API_KEY, 
        base_url=PURDUE_API_BASE
    )

    # 5) QA chains
    print("Building QA chains...")
    qa = make_qa_chain(retriever, llm)
    conv_qa = make_conversational_chain(retriever, llm)

    return {
        "docs": docs,
        "chunks": chunks,
        "retriever": retriever,
        "qa": qa,
        "conv_qa": conv_qa,
    }

def ask_question(pipeline, question: str):
    """Ask a single question using RetrievalQA."""
    result = pipeline["qa"]({"query": question})
    return {
        "answer": result["result"],
        "sources": [d.metadata for d in result["source_documents"]]
    }

def chat_conversation(pipeline, questions: List[str]):
    """Have a conversation using ConversationalRetrievalChain."""
    chat_history = []
    responses = []
    
    for question in questions:
        result = pipeline["conv_qa"]({"question": question, "chat_history": chat_history})
        response = {
            "question": question,
            "answer": result["answer"],
            "sources": [d.metadata for d in result["source_documents"]]
        }
        responses.append(response)
        chat_history.append((question, result["answer"]))
    
    return responses

if __name__ == "__main__":
    # Ensure API key is set
    if not PURDUE_API_KEY:
        raise RuntimeError("Please set PURDUE_API_KEY environment variable or update the hardcoded key.")

    # Build pipeline
    pipeline = build_pipeline()
    print("Pipeline built successfully!")

    # Example 1: Single question
    print("\n" + "="*50)
    print("SINGLE QUESTION EXAMPLE")
    print("="*50)
    
    question = "What are the main topics covered in this document?"
    result = ask_question(pipeline, question)
    print(f"Q: {question}")
    print(f"A: {result['answer']}")
    print(f"Sources: {result['sources']}")

    # Example 2: Conversational flow
    print("\n" + "="*50)
    print("CONVERSATIONAL EXAMPLE")
    print("="*50)
    
    questions = [
        "What is the course structure?",
        "What are the lab requirements?",
        "How is grading done?"
    ]
    
    responses = chat_conversation(pipeline, questions)
    for i, resp in enumerate(responses, 1):
        print(f"\nQ{i}: {resp['question']}")
        print(f"A{i}: {resp['answer']}")
        print(f"Sources: {resp['sources']}")

    print("\n" + "="*50)
    print("Ready for interactive use!")
    print("="*50)