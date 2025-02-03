import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Configure Google Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing. Please check your .env file.")

genai.configure(api_key=GEMINI_API_KEY)

def init_llm():
    global llm_hub, embeddings

    gemini_model = genai.GenerativeModel("gemini-1.5-flash-8b")
    llm_hub = gemini_model

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    return llm_hub, embeddings
