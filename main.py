import threading

import uvicorn
from dotenv import load_dotenv

from src.api import app
from src.chatbot import demo

# Load environment variables from .env file
load_dotenv()


def run_api_server():
    """Run the FastAPI server in a separate thread"""
    uvicorn.run(app, host="0.0.0.0", port=8000)


def run_chatbot_interface():
    """Create and run the Gradio interface"""
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


# Run the application
if __name__ == "__main__":
    # Start API server in a separate thread
    api_thread = threading.Thread(target=run_api_server, daemon=True)
    api_thread.start()

    # Run the chatbot interface in the main thread
    run_chatbot_interface()
