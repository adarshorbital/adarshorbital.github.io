# knowledge_base.py - RAG system for GustyAI PDFs and Excel files
import os
import pandas as pd
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import pickle

class GustyAIKnowledgeBase:
    def __init__(self, documents_folder="documents", vector_store_path="vector_store"):
        self.documents_folder = documents_folder
        self.vector_store_path = vector_store_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = None
        
    def load_excel_files(self):
        """Load and process Excel files"""
        documents = []
        
        # Find Excel files
        excel_extensions = ['.xlsx', '.xls', '.csv']
        excel_files = []
        for ext in excel_extensions:
            excel_files.extend([f for f in os.listdir(self.documents_folder) if f.lower().endswith(ext)])
        
        if not excel_files:
            return []
        
        print(f"📊 Loading {len(excel_files)} Excel/CSV files...")
        
        for excel_file in excel_files:
            excel_path = os.path.join(self.documents_folder, excel_file)
            try:
                # Read Excel/CSV file
                if excel_file.lower().endswith('.csv'):
                    df = pd.read_csv(excel_path)
                else:
                    # For Excel files, read all sheets
                    excel_data = pd.read_excel(excel_path, sheet_name=None)
                    
                    # Process each sheet
                    for sheet_name, sheet_df in excel_data.items():
                        # Convert sheet to text representation
                        sheet_text = f"Sheet: {sheet_name}\n\n"
                        
                        # Add column headers
                        sheet_text += "Columns: " + ", ".join(sheet_df.columns.astype(str)) + "\n\n"
                        
                        # Add data summary
                        sheet_text += f"Number of rows: {len(sheet_df)}\n\n"
                        
                        # Add first few rows as sample data
                        sheet_text += "Sample data:\n"
                        sheet_text += sheet_df.head(10).to_string(index=False) + "\n\n"
                        
                        # Add statistical summary for numeric columns
                        numeric_cols = sheet_df.select_dtypes(include=['number']).columns
                        if len(numeric_cols) > 0:
                            sheet_text += "Statistical summary for numeric columns:\n"
                            sheet_text += sheet_df[numeric_cols].describe().to_string() + "\n\n"
                        
                        # Create document for this sheet
                        doc = Document(
                            page_content=sheet_text,
                            metadata={
                                'source_file': excel_file,
                                'source_type': 'excel',
                                'sheet_name': sheet_name,
                                'rows': len(sheet_df),
                                'columns': len(sheet_df.columns)
                            }
                        )
                        documents.append(doc)
                    
                    print(f"✅ Loaded {excel_file}: {len(excel_data)} sheets")
                    continue
                
                # For CSV files
                if excel_file.lower().endswith('.csv'):
                    # Convert CSV to text representation
                    csv_text = f"CSV File: {excel_file}\n\n"
                    
                    # Add column headers
                    csv_text += "Columns: " + ", ".join(df.columns.astype(str)) + "\n\n"
                    
                    # Add data summary
                    csv_text += f"Number of rows: {len(df)}\n\n"
                    
                    # Add first few rows as sample data
                    csv_text += "Sample data:\n"
                    csv_text += df.head(10).to_string(index=False) + "\n\n"
                    
                    # Add statistical summary for numeric columns
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        csv_text += "Statistical summary for numeric columns:\n"
                        csv_text += df[numeric_cols].describe().to_string() + "\n\n"
                    
                    # Create document for CSV
                    doc = Document(
                        page_content=csv_text,
                        metadata={
                            'source_file': excel_file,
                            'source_type': 'csv',
                            'rows': len(df),
                            'columns': len(df.columns)
                        }
                    )
                    documents.append(doc)
                    print(f"✅ Loaded {excel_file}: {len(df)} rows")
                
            except Exception as e:
                print(f"❌ Error loading {excel_file}: {str(e)}")
        
        return documents
    
    def load_pdfs(self):
        """Load and process PDF files"""
        documents = []
        
        # Load all PDFs from the folder
        pdf_files = [f for f in os.listdir(self.documents_folder) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            return []
        
        print(f"📚 Loading {len(pdf_files)} PDF files...")
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.documents_folder, pdf_file)
            try:
                loader = PyPDFLoader(pdf_path)
                pages = loader.load()
                
                # Add source metadata
                for page in pages:
                    page.metadata['source_file'] = pdf_file
                    page.metadata['source_type'] = 'pdf'
                
                documents.extend(pages)
                print(f"✅ Loaded {pdf_file}: {len(pages)} pages")
                
            except Exception as e:
                print(f"❌ Error loading {pdf_file}: {str(e)}")
        
        return documents
    
    def load_all_documents(self):
        """Load and process all supported document types"""
        documents = []
        
        # Create documents folder if it doesn't exist
        if not os.path.exists(self.documents_folder):
            os.makedirs(self.documents_folder)
            print(f"📁 Created {self.documents_folder} folder. Please add your PDF and Excel files there.")
            return []
        
        # Load PDFs
        pdf_docs = self.load_pdfs()
        documents.extend(pdf_docs)
        
        # Load Excel/CSV files
        excel_docs = self.load_excel_files()
        documents.extend(excel_docs)
        
        if not documents:
            print(f"⚠️  No supported files found in {self.documents_folder} folder")
            print("📋 Supported formats: PDF (.pdf), Excel (.xlsx, .xls), CSV (.csv)")
            return []
        
        print(f"📄 Total documents loaded: {len(documents)}")
        return documents
    
    def create_vector_store(self, force_rebuild=False):
        """Create or load vector store from all supported documents"""
        
        # Check if vector store already exists
        if os.path.exists(f"{self.vector_store_path}.pkl") and not force_rebuild:
            print("📦 Loading existing vector store...")
            try:
                with open(f"{self.vector_store_path}.pkl", 'rb') as f:
                    self.vector_store = pickle.load(f)
                print("✅ Vector store loaded successfully")
                return True
            except Exception as e:
                print(f"❌ Error loading vector store: {e}")
                print("🔄 Rebuilding vector store...")
        
        # Load and process all documents
        documents = self.load_all_documents()
        
        if not documents:
            print("❌ No documents to process")
            return False
        
        # Split documents into chunks
        print("✂️  Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Increased for Excel data
            chunk_overlap=300,
            separators=["\n\n", "\n", " ", ""]
        )
        
        splits = text_splitter.split_documents(documents)
        print(f"📄 Created {len(splits)} text chunks")
        
        # Create vector store
        print("🧠 Creating embeddings and vector store...")
        try:
            self.vector_store = FAISS.from_documents(splits, self.embeddings)
            
            # Save vector store
            with open(f"{self.vector_store_path}.pkl", 'wb') as f:
                pickle.dump(self.vector_store, f)
            
            print("✅ Vector store created and saved successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error creating vector store: {e}")
            return False
    
    def search_knowledge(self, query, k=3):
        """Search for relevant information in the knowledge base"""
        if not self.vector_store:
            print("❌ Vector store not initialized")
            return []
        
        try:
            # Search for similar documents
            docs = self.vector_store.similarity_search(query, k=k)
            
            # Format results
            results = []
            for doc in docs:
                results.append({
                    'content': doc.page_content,
                    'source': doc.metadata.get('source_file', 'Unknown'),
                    'page': doc.metadata.get('page', 'Unknown')
                })
            
            return results
            
        except Exception as e:
            print(f"❌ Error searching knowledge base: {e}")
            return []
    
    def get_context_for_query(self, query, max_context_length=2000):
        """Get relevant context for a query, formatted for the AI"""
        results = self.search_knowledge(query, k=3)
        
        if not results:
            return ""
        
        context_parts = []
        current_length = 0
        
        for result in results:
            content = result['content'].strip()
            source_info = f"[Source: {result['source']}]"
            
            part = f"{content}\n{source_info}\n"
            
            if current_length + len(part) > max_context_length:
                break
                
            context_parts.append(part)
            current_length += len(part)
        
        if context_parts:
            context = "Relevant information from knowledge base:\n\n" + "\n".join(context_parts)
            return context
        
        return ""

# Initialize knowledge base (singleton pattern)
kb = GustyAIKnowledgeBase()

def initialize_knowledge_base():
    """Initialize the knowledge base - call this once when starting the server"""
    print("🚀 Initializing GustyAI Knowledge Base...")
    success = kb.create_vector_store()
    if success:
        print("✅ Knowledge base ready!")
    else:
        print("⚠️  Knowledge base initialization failed - will work without PDFs")
    return success

def search_pdfs(query):
    """Search PDFs for relevant context - use this in your API calls"""
    return kb.get_context_for_query(query)