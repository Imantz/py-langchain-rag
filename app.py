import gradio as gr
from worker import init_llm

# Initialize Google Gemini and Embeddings
llm_hub, embeddings = init_llm()

def chatbot_response(message, history):
    try:
        # Generate response using only the latest user message
        response = llm_hub.generate_content([{"parts": [{"text": message}]}])
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
