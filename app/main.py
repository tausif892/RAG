from dotenv import load_dotenv
import chromadb
from google import genai
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

load_dotenv()

app = FastAPI(title="Products RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def home():
    return {"message": "Products RAG is running"}

@app.on_event("startup")
async def start():
    await initialize()

async def initialize():
    try:
        print("CONNECTING TO CHROMADB")

        chroma_client = chromadb.CloudClient(
            api_key=os.getenv("CHROMA_API_KEY"),   
            tenant=os.getenv("CHROMA_TENANT"),
            database=os.getenv("CHROMA_DB")
        )
        collection = chroma_client.get_collection("products")

        print("CONNECTING TO GOOGLE GEMINI")

        gemini_api_key = os.getenv("GEMINI_API_KEY")
        gemini_client = genai.Client(api_key=gemini_api_key)

        app.state.collection = collection
        app.state.gemini_client = gemini_client

        print("ALL SYSTEMS READY TO GO!!")
    except Exception as e:
        print("Error during initialization:", e)

def get_context(query: str, collection, seller_name):
    seller_name="circulx_seller_profile_1"
    try:
        results = collection.query(query_texts=[query],where={"seller_name":seller_name}, n_results=10)
        if not results or not results.get("documents") or not results["documents"][0]:
            return "No relevant information found."
        return "\n".join(results["documents"][0])
    except Exception as e:
        print("Context error:", e)
        return "Error retrieving data"

def answer_with_llm(query: str, context: str, client):
    prompt = f"""
    You are a helpful chatbot for an ecommerce platform.
    Based on the following information, answer the question clearly.
    Be polite and articulate.
    If the context shows that no relevant information was found, tell the user politely that this lies outside your scope of information.

    Context:
    {context}

    Question:
    {query}

    Answer clearly and concisely.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text.strip() if response.text else "No response generated."

@app.get("/query")
def query(q: str, seller: str):
    if not hasattr(app.state, "collection") or not hasattr(app.state, "gemini_client"):
        return {"status": "Loading", "message": "Server still starting"}
    
    ctx = get_context(q, app.state.collection, seller_name=seller)
    answer = answer_with_llm(query=q, context=ctx, client=app.state.gemini_client)

    return {"query": q, "context": ctx, "answer": answer}
