import gradio as gr
import requests

def chat(prompt):
    API_URL = "http://192.168.1.129:1234/v1/chat/completions"
    
    data = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "local-model",
        "temperature": 0.7
    }
    
    try:
        response = requests.post(
            API_URL, 
            headers={"Content-Type": "application/json"},
            json=data
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"

interface = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(lines=3, placeholder="Enter your message..."),
    outputs="text",
    title="Local LLM Chat",
    description="Chat with your local LLM model"
)

if __name__ == "__main__":
    interface.launch()