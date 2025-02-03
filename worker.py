import os
import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing. Please check your .env file.")

genai.configure(api_key=GEMINI_API_KEY)

llm_hub = None
embeddings = None
conversation_retrieval_chain = None
chat_history = []
vector_db = None

def init_llm():
    global llm_hub, embeddings
    llm_hub = genai.GenerativeModel("gemini-1.5-flash-8b")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return llm_hub, embeddings

def process_document(document_path):
    global conversation_retrieval_chain, vector_db

    try:
        loader = PyPDFLoader(document_path)
        documents = loader.load()

        if not documents:
            return "❌ PDF file is empty or could not be read."

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
        texts = text_splitter.split_documents(documents)

        if not texts:
            return "❌ Failed to split the document into chunks."

        vector_db = Chroma.from_documents(texts, embedding=embeddings, persist_directory="./chroma_db")

        retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={'k': min(4, len(texts)), 'lambda_mult': 0.25})

        conversation_retrieval_chain = retriever

        return "✅ PDF processed and indexed successfully!"
    
    except Exception as e:
        return f"❌ Error processing PDF: {str(e)}"

def process_prompt(prompt):
    global conversation_retrieval_chain, chat_history

    if conversation_retrieval_chain:
        try:
            relevant_docs = conversation_retrieval_chain.invoke(prompt)
            context_text = "\n".join([doc.page_content for doc in relevant_docs])

            formatted_input = f"Context:\n{context_text}\n\nUser Question:\n{prompt}"
            response = llm_hub.generate_content([{"text": formatted_input}])

            answer = response.text if hasattr(response, "text") else "Sorry, I couldn't understand."

        except Exception as e:
            return f"❌ Retrieval Error: {str(e)}"
    else:
        try:
            response = llm_hub.generate_content([{"text": prompt}])
            answer = response.text if hasattr(response, "text") else "Sorry, I couldn't understand."
        except Exception as e:
            return f"❌ Gemini Error: {str(e)}"

    chat_history.append((prompt, answer))

    return answer
