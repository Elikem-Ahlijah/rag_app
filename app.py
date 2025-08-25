# REPLACE the imports section in your app.py with this:
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import TokenTextSplitter  # CHANGED: Fixed import
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
# REMOVED: Ollama import to save memory
import os
import tempfile
import shutil
from pydantic import BaseModel
import logging
import gc  # ADDED: For garbage collection

# REPLACE your initialize_components() function with this memory-optimized version:
def initialize_components():
    """Initialize embeddings and vectorstore with memory optimization"""
    global vectorstore, embeddings, qa_chain
    
    try:
        logger.info("Initializing embeddings with memory optimization...")
        # Use the smallest possible embedding model to save memory
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",  # Only 22MB model
            model_kwargs={
                "device": "cpu",
                "trust_remote_code": False
            },
            show_progress=False,  # Reduce memory usage
        )
        
        logger.info("Initializing ChromaDB with minimal settings...")
        # Minimal ChromaDB configuration
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"}  # Removed memory-intensive settings
        )
        
        # REMOVED: Skip LLM initialization to save memory - use retrieval-only mode
        logger.info("Skipping LLM initialization to conserve memory")
        qa_chain = None  # We'll handle this in the query endpoint
        
        # Force garbage collection
        gc.collect()
        
        logger.info("Components initialized successfully (memory-optimized mode)")
        
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        raise

# REPLACE your upload_file function with this memory-optimized version:
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a document with memory optimization"""
    if not vectorstore:
        raise HTTPException(status_code=500, detail="Vectorstore not initialized")
    
    try:
        validate_file(file)
        
        # Save file with unique name to avoid conflicts
        import uuid
        unique_filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # Write file in chunks to manage memory
        with open(file_path, "wb") as f:
            while chunk := await file.read(8192):  # Read in 8KB chunks
                f.write(chunk)
        
        logger.info(f"File saved: {file_path}")
        
        # Load document with minimal memory usage
        loader = UnstructuredFileLoader(
            file_path,
            mode="single",  # CHANGED: Use single mode instead of elements to save memory
            strategy="fast"
        )
        docs = loader.load()
        
        if not docs:
            raise HTTPException(status_code=400, detail="No content could be extracted")
        
        logger.info(f"Loaded document with {len(docs[0].page_content)} characters")
        
        # CHANGED: More aggressive chunking to reduce memory
        splitter = TokenTextSplitter(
            chunk_size=256,    # CHANGED: Smaller chunks (was 512)
            chunk_overlap=25,  # CHANGED: Smaller overlap (was 50)
            add_start_index=True
        )
        chunks = splitter.split_documents(docs)
        
        # ADDED: Limit total chunks to prevent memory issues
        max_chunks = 50
        if len(chunks) > max_chunks:
            chunks = chunks[:max_chunks]
            logger.warning(f"Truncated to {max_chunks} chunks to manage memory")
        
        logger.info(f"Created {len(chunks)} chunks")
        
        # ADDED: Add to vectorstore in smaller batches
        batch_size = 10
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            vectorstore.add_documents(batch)
            
        # ADDED: Clean up immediately
        try:
            os.remove(file_path)
            del docs  # Explicit cleanup
            del chunks
            gc.collect()  # Force garbage collection
        except:
            pass
        
        return {
            "message": f"File '{file.filename}' processed successfully",
            "chunks": min(len(chunks), max_chunks),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        # Cleanup on error
        try:
            if 'file_path' in locals():
                os.remove(file_path)
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

# REPLACE your query_rag function with this memory-optimized version:
@app.post("/query")
async def query_rag(item: Query):
    """Query with memory-optimized retrieval"""
    if not vectorstore:
        raise HTTPException(status_code=500, detail="Vectorstore not initialized")
    
    if not item.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # CHANGED: Simple retrieval without heavy LLM processing
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}  # CHANGED: Limit results to save memory (was k=3, fetch_k=6)
        )
        docs = retriever.get_relevant_documents(item.query)
        
        if not docs:
            return {
                "answer": "No relevant documents found for your query.",
                "sources": [],
                "method": "Document retrieval"
            }
        
        # CHANGED: Create a simple, memory-efficient response
        context_pieces = []
        sources = set()
        
        for doc in docs[:3]:  # Limit to top 3 results
            # CHANGED: Truncate content to manage memory
            content = doc.page_content[:300]
            context_pieces.append(content)
            
            source = doc.metadata.get("source", "Unknown")
            if source != "Unknown":
                sources.add(os.path.basename(source))
        
        # Simple context assembly
        context = "\n\n".join(context_pieces)
        
        return {
            "answer": f"Based on the uploaded documents:\n\n{context}",
            "sources": list(sources),
            "method": "Memory-optimized document retrieval"
        }
            
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
