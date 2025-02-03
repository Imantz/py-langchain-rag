import gradio as gr
from worker import init_llm, process_document, process_prompt

llm_hub, embeddings = init_llm()

def chatbot_response(message, history):
    return process_prompt(message)

def handle_pdf_upload(pdf_file):
    if pdf_file is None:
        return "❌ No file uploaded."

    return process_document(pdf_file.name)

with gr.Blocks() as demo:
    gr.Markdown("## 📄 RAG Chatbot with Google Gemini & ChromaDB")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Upload PDF for RAG")
            pdf_input = gr.File(label="Upload PDF")
            upload_button = gr.Button("Process PDF")
            status_output = gr.Textbox(label="Processing Status")

            upload_button.click(fn=handle_pdf_upload, inputs=pdf_input, outputs=status_output)

        with gr.Column(scale=2):
            gr.Markdown("### Chat with the AI")
            chatbot = gr.ChatInterface(fn=chatbot_response, type="messages")

demo.launch()
