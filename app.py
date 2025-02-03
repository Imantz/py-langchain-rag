import gradio as gr

def chatbot_response(message, history):
    history.append((message, "Hello! How can I assist you?"))  # Simple static response
    return history

chat_interface = gr.ChatInterface(fn=chatbot_response)
chat_interface.launch()
