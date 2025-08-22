from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import tempfile
import shutil
from pydantic import BaseModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Application", description="Document Q&A with RAG")

# Use temporary directories that work on ephemeral systems
TEMP_DIR = tempfile.mkdtemp()
UPLOAD_DIR = os.path.join(TEMP_DIR, "uploads")
CHROMA_DIR = os.path.join(TEMP_DIR, "chroma_db")

# Create directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# File validation settings
ALLOWED_EXTENSIONS = {'.pdf', '.xlsx', '.xls', '.csv', '.txt', '.docx'}
ALLOWED_MIME_TYPES = {
    '.pdf': ['application/pdf'],
    '.xlsx': ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'],
    '.xls': ['application/vnd.ms-excel'],
    '.csv': ['text/csv', 'application/csv'],
    '.txt': ['text/plain'],
    '.docx': ['application/vnd.openxmlformats-officedocument.wordprocessingml.document']
}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# Global variables for components
vectorstore = None
qa_chain = None
embeddings = None

def initialize_components():
    """Initialize embeddings and vectorstore"""
    global vectorstore, embeddings, qa_chain
    
    try:
        logger.info("Initializing embeddings...")
        # Use a lightweight, CPU-optimized embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={
                "device": "cpu",
                "normalize_embeddings": True
            },
            encode_kwargs={"normalize_embeddings": True}
        )
        
        logger.info("Initializing ChromaDB...")
        # Initialize ChromaDB with error handling
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize Ollama with fallback
        logger.info("Initializing LLM...")
        try:
            llm = Ollama(
                model="llama3.1:8b-instruct",
                base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
                temperature=0.1
            )
            # Test connection
            test_response = llm.invoke("Hello")
            logger.info("Ollama connection successful")
        except Exception as ollama_error:
            logger.warning(f"Ollama connection failed: {ollama_error}")
            # Fallback to a simple response system
            llm = None
        
        # Custom prompt template
        prompt_template = """Use the following context to answer the question accurately and concisely. 
        If the answer cannot be found in the context, say "I don't have enough information to answer this question."

Context: {context}

Question: {question}

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        if llm:
            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 3, "fetch_k": 6}
                ),
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
        
        logger.info("Components initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        raise

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    initialize_components()

# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    try:
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
    except:
        pass

class Query(BaseModel):
    query: str

def validate_file(file: UploadFile):
    """Validate file type and size with better error handling"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    _, file_extension = os.path.splitext(file.filename.lower())
    
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file_extension}. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # More flexible MIME type checking
    content_type = file.content_type
    allowed_mime_types = ALLOWED_MIME_TYPES.get(file_extension, [])
    
    if content_type and allowed_mime_types and content_type not in allowed_mime_types:
        logger.warning(f"MIME type mismatch: {content_type} not in {allowed_mime_types}")
        # Don't fail on MIME type mismatch, just log it
    
    # Check file size
    file.file.seek(0, os.SEEK_END)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400, 
            detail=f"File too large: {file_size / (1024*1024):.1f}MB. Max size: {MAX_FILE_SIZE / (1024*1024)}MB"
        )
    
    return file_extension

@app.get("/")
async def read_root():
    """Serve the main HTML page"""
    return FileResponse("templates/index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "vectorstore_initialized": vectorstore is not None,
        "qa_chain_initialized": qa_chain is not None
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a document"""
    if not vectorstore:
        raise HTTPException(status_code=500, detail="Vectorstore not initialized")
    
    try:
        validate_file(file)
        
        # Save file with unique name to avoid conflicts
        import uuid
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # Write file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"File saved: {file_path}")
        
        # Load and process document
        loader = UnstructuredFileLoader(
            file_path,
            mode="elements",
            strategy="fast"  # Use faster processing
        )
        docs = loader.load()
        
        if not docs:
            raise HTTPException(status_code=400, detail="No content could be extracted from the file")
        
        logger.info(f"Loaded {len(docs)} document elements")
        
        # Split documents
        splitter = TokenTextSplitter(
            chunk_size=512,  # Increased chunk size
            chunk_overlap=50,
            add_start_index=True
        )
        chunks = splitter.split_documents(docs)
        
        logger.info(f"Created {len(chunks)} chunks")
        
        # Add to vectorstore
        vectorstore.add_documents(chunks)
        
        # Clean up temporary file
        try:
            os.remove(file_path)
        except:
            pass
        
        return {
            "message": f"File '{file.filename}' processed successfully",
            "chunks": len(chunks),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/query")
async def query_rag(item: Query):
    """Query the RAG system"""
    if not vectorstore:
        raise HTTPException(status_code=500, detail="Vectorstore not initialized")
    
    if not item.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        if qa_chain:
            # Use full RAG pipeline
            result = qa_chain({"query": item.query})
            
            sources = []
            for doc in result.get("source_documents", []):
                source = doc.metadata.get("source", "Unknown")
                if source != "Unknown":
                    sources.append(os.path.basename(source))
            
            return {
                "answer": result["result"],
                "sources": list(set(sources)),  # Remove duplicates
                "method": "RAG with LLM"
            }
        else:
            # Fallback to simple retrieval without LLM
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(item.query)
            
            if not docs:
                return {
                    "answer": "No relevant documents found for your query.",
                    "sources": [],
                    "method": "Document retrieval only"
                }
            
            # Simple concatenation of relevant chunks
            context = "\n\n".join([doc.page_content[:200] + "..." for doc in docs[:3]])
            
            sources = []
            for doc in docs:
                source = doc.metadata.get("source", "Unknown")
                if source != "Unknown":
                    sources.append(os.path.basename(source))
            
            return {
                "answer": f"Based on the uploaded documents:\n\n{context}",
                "sources": list(set(sources)),
                "method": "Document retrieval only (LLM unavailable)"
            }
            
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Mount static files
try:
    app.mount("/static", StaticFiles(directory="templates"), name="static")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
