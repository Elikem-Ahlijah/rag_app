from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from pydantic import BaseModel
import mimetypes

app = FastAPI()

# Directories
UPLOAD_DIR = "./uploads"
CHROMA_DIR = "./chroma_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# File validation settings
ALLOWED_EXTENSIONS = {'.pdf', '.xlsx', '.xls', '.csv'}
ALLOWED_MIME_TYPES = {
    '.pdf': 'application/pdf',
    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    '.xls': 'application/vnd.ms-excel',
    '.csv': 'text/csv'
}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB in bytes

# Free high-accuracy embedding model
embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1")

# Persistent ChromaDB with compression
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine", "hnsw:M": 16, "hnsw:ef_construction": 100}  # Optimized for storage
)

# Free generation model via Ollama (assumes Ollama is running locally or hosted)
llm = OllamaLLM(model="llama3.1:8b-instruct")

# Custom prompt for accuracy
prompt_template = """Use the following context to answer the question accurately. If unsure, say "I don't know".
Context: {context}
Question: {question}
Answer:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# QA chain with hybrid retrieval
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# Pydantic model for query
class Query(BaseModel):
    query: str

def validate_file(file: UploadFile):
    """Validate file type and size."""
    # Check file extension
    _, file_extension = os.path.splitext(file.filename)
    if file_extension.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
    
    # Check MIME type
    content_type = file.content_type
    if content_type != ALLOWED_MIME_TYPES.get(file_extension.lower()):
        raise HTTPException(status_code=400, detail=f"Invalid MIME type for {file_extension}. Expected: {ALLOWED_MIME_TYPES[file_extension.lower()]}")
    
    # Check file size (read file to validate)
    file.file.seek(0, os.SEEK_END)
    file_size = file.file.tell()
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File too large. Max size: {MAX_FILE_SIZE / (1024 * 1024)} MB")
    file.file.seek(0)  # Reset file pointer
    
    return file_extension

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Validate file
        validate_file(file)
        
        # Save file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Parse with Unstructured (high accuracy for PDF, Excel, CSV)
        loader = UnstructuredFileLoader(file_path, mode="elements")
        docs = loader.load()
        
        # Optimized chunking (token-based, smaller chunks)
        splitter = TokenTextSplitter(chunk_size=256, chunk_overlap=50, add_start_index=True)
        chunks = splitter.split_documents(docs)
        
        # Embed and store in ChromaDB
        vectorstore.add_documents(chunks)
        
        return {"message": "File uploaded, parsed, chunked, and stored successfully", "chunks": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/query")
async def query_rag(item: Query):
    try:
        result = qa_chain({"query": item.query})
        sources = [doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]
        return {"answer": result["result"], "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Serve static HTML UI
app.mount("/", StaticFiles(directory="templates", html=True), name="static")

