import os
import uuid

from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader

from utils import RAG_TEMPLATE, format_docs

local_embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma(embedding_function=local_embeddings)

model = ChatOllama(
    model="llama3.1:8b",
)

app = FastAPI()

# Serve static files from the 'static' directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create the 'static' folder if it doesn't exist
os.makedirs("../static", exist_ok=True)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # Generate a unique file name and save the file in the 'static' folder
    file_name = f"{uuid.uuid4()}.pdf"
    file_path = os.path.join("../static", file_name)

    # Write the uploaded file to the static directory
    with open(file_path, "wb") as temp_file:
        content = await file.read()
        temp_file.write(content)

    loader = PyPDFLoader(file_path)
    data = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)

    vectorstore.add_documents(all_splits)

    return {"status": "Document processed and indexed"}


class QuestionRequest(BaseModel):
    questions: List[str]


@app.post("/ask")
async def ask_questions(request: QuestionRequest):
    responses = {}
    for question in request.questions:
        # Retrieve relevant documents based on the question
        docs = vectorstore.similarity_search(question)

        # Format documents and generate the answer using ChatOllama
        formatted_docs = format_docs(docs)
        rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
        chain = (
            RunnablePassthrough.assign(context=lambda doc: formatted_docs)
            | rag_prompt
            | model
            | StrOutputParser()
        )
        response = chain.invoke({"context": docs, "question": question})
        responses[question] = response.strip()

    return responses
