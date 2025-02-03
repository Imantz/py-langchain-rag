import os
from dotenv import load_dotenv
import gradio as gr
import google.generativeai as genai

# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing. Please check your .env file.")


# Configure Google Gemini API
genai.configure(api_key=api_key)
# gemini_model_name = "gemini-2.0-flash-thinking-exp-01-21"
model_name = "gemini-1.5-flash-8b"

# # Define Gemini model instance
model = genai.GenerativeModel(model_name=model_name)

def chatbot_response(message, history):
    try:
        # Generate response using only the latest user message
        response = model.generate_content([{"parts": [{"text": message}]}])
        return response.text if hasattr(response, "text") else "Sorry, I couldn't understand."
    
    except Exception as e:
        return f"Error: {str(e)}"

# Create and launch Gradio Chat Interface
demo = gr.ChatInterface(
    fn=chatbot_response,
    type="messages",
    examples=["hello", "hola", "merhaba"], 
    title="Gemini Chatbot"
)

demo.launch()
