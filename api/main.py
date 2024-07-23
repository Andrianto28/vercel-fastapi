from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer, util
import os
import shutil
import tempfile
from uuid import uuid4
import json
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

@app.get("/")
async def heatlt_check():
    return "the health is good"

# Initialize the model and embeddings
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)
embeddings = SentenceTransformerEmbeddings(model_name=model_name)

# Initialize the LLM model
llm_model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=llm_model_name)
chain = load_qa_chain(llm, chain_type="stuff", verbose=True)

# Define directories
pdf_directory = "pdf"
db_directory = "database"
metadata_file = "metadata.json"

# Ensure directories exist
os.makedirs(pdf_directory, exist_ok=True)
os.makedirs(db_directory, exist_ok=True)

# Load or initialize metadata
if os.path.exists(metadata_file):
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
else:
    metadata = {}

class QueryRequest(BaseModel):
    query: str
    human_answer: str
    db_name: str

class DocumentIDRequest(BaseModel):
    document_id: str

@app.get("/ping/")
def ping():
    return {"status": "oke"}

@app.post("/upload_document")
async def upload_document(file: UploadFile = File(...)):
    try:
        document_id = str(uuid4())
        save_path = os.path.join(pdf_directory, document_id + "_" + file.filename)
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Update metadata
        metadata[document_id] = {
            "filename": file.filename,
            "path": save_path
        }
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)

        return {"status": "success", "document_id": document_id, "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create_vectordb")
async def create_vectordb(request: DocumentIDRequest):
    try:
        if request.document_id not in metadata:
            raise HTTPException(status_code=404, detail="Document not found.")
        
        document_path = metadata[request.document_id]["path"]
        loader = PyPDFLoader(document_path)
        documents = loader.load()
        docs = split_docs(documents)

        db_name = f"db_{request.document_id}"
        db_path = os.path.join(db_directory, db_name)
        new_vectordb = Chroma.from_documents(
            documents=docs, embedding=embeddings, persist_directory=db_path
        )
        new_vectordb.persist()

        return {"status": "success", "database": db_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list_documents")
async def list_documents():
    try:
        return {"documents": [{"document_id": doc_id, "filename": info["filename"]} for doc_id, info in metadata.items()]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list_vectordbs")
async def list_vectordbs():
    try:
        dbs = [d for d in os.listdir(db_directory) if os.path.isdir(os.path.join(db_directory, d))]
        return {"vectordbs": dbs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete_document")
async def delete_document(request: DocumentIDRequest):
    try:
        if request.document_id not in metadata:
            raise HTTPException(status_code=404, detail="Document not found.")
        
        document_path = metadata[request.document_id]["path"]
        os.remove(document_path)
        
        db_path = os.path.join(db_directory, f"db_{request.document_id}")
        if os.path.exists(db_path):
            shutil.rmtree(db_path)

        # Update metadata
        del metadata[request.document_id]
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)

        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/similarity_score")
async def similarity_score_endpoint(request: QueryRequest):
    try:
        db_path = os.path.join(db_directory, request.db_name)
        if not os.path.exists(db_path):
            raise HTTPException(status_code=404, detail="Vector database not found.")
        
        vectordb = Chroma(persist_directory=db_path, embedding_function=embeddings)
        matching_docs = vectordb.similarity_search(request.query)
        
        # Combine the content of matching documents to form an AI answer
        ai_answer = " ".join([doc.page_content for doc in matching_docs])
        
        # Calculate similarity score
        doc_embedding = model.encode(ai_answer, convert_to_tensor=True)
        answer_embedding = model.encode(request.human_answer, convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(doc_embedding, answer_embedding).item()
        
        return {
            "query": request.query,
            "ai_answer": ai_answer,
            "human_answer": request.human_answer,
            "similarity_score": similarity_score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/similarity_score_llm")
async def similarity_score_llm_endpoint(request: QueryRequest):
    try:
        db_path = os.path.join(db_directory, request.db_name)
        if not os.path.exists(db_path):
            raise HTTPException(status_code=404, detail="Vector database not found.")
        
        vectordb = Chroma(persist_directory=db_path, embedding_function=embeddings)
        matching_docs = vectordb.similarity_search(request.query)
        
        # Generate AI answer using GPT-3.5-turbo
        ai_answer = chain.run(input_documents=matching_docs, question=request.query)
        
        # Calculate similarity score
        doc_embedding = model.encode(ai_answer, convert_to_tensor=True)
        answer_embedding = model.encode(request.human_answer, convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(doc_embedding, answer_embedding).item()
        
        return {
            "query": request.query,
            "ai_answer": ai_answer,
            "human_answer": request.human_answer,
            "similarity_score": similarity_score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
