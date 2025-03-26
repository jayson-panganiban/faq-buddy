import os
import threading

import uvicorn
from dotenv import load_dotenv

from src.api import app
from src.chatbot import create_chatbot_interface

# Load environment variables from .env file
load_dotenv()


def run_api_server():
    """Run the FastAPI server in a separate thread"""
    uvicorn.run(app, host="0.0.0.0", port=8000)


def run_chatbot_interface():
    """Create and run the Gradio interface"""
    interface = create_chatbot_interface()
    interface.launch(server_name="0.0.0.0", share=True)


# Run the application
if __name__ == "__main__":
    # Start API server in a separate thread
    api_thread = threading.Thread(target=run_api_server, daemon=True)
    api_thread.start()

    # Set API URL for chatbot (using the local API server)
    os.environ["API_URL"] = "http://localhost:8000"

    # Run the chatbot interface in the main thread
    run_chatbot_interface()
