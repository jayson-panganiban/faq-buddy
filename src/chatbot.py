import logging
import os

import gradio as gr
import requests  # type: ignore
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    logger.warning(
        "⚠️ OPENAI_API_KEY environment variable not set. Chatbot functionality requires it."
    )
if not API_URL:
    raise ValueError("API_URL environment variable is not set.")


def call_api(question: str) -> str:
    headers = {"X-OpenAI-Key": OPENAI_API_KEY, "Content-Type": "application/json"}
    payload = {"text": question}

    try:
        logger.info(f"Sending request to API: {API_URL}/ask")
        response = requests.post(
            f"{API_URL}/ask", json=payload, headers=headers, timeout=30.0
        )
        response.raise_for_status()
        api_response = response.json()
        logger.info(f"Received API response: {api_response}")
        return api_response.get("answer", "⚠️ Error: No answer received from API.")

    except requests.exceptions.Timeout:
        logger.error("API request timed out.")
        return (
            "⚠️ Error: The request to the backend API timed out. Please try again later."
        )
    except requests.exceptions.ConnectionError:
        logger.error(f"Could not connect to API at {API_URL}.")
        return f"⚠️ Error: Could not connect to the backend API at {API_URL}. Is it running?"
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        error_detail = e.response.text
        logger.error(f"API request failed (Status: {status_code}): {error_detail}")
        if status_code == 401:  # Unauthorized
            return "⚠️ Error: Invalid OpenAI API Key. Please check the secret in Space settings."
        elif (
            status_code == 503
        ):  # Service Unavailable (often OpenAI issues or backend overload)
            return "⚠️ Error: The AI service is currently unavailable or overloaded. Please try again later."
        else:
            return (
                f"⚠️ Error: The backend API returned an error (Status: {status_code})."
            )
    except Exception as e:
        logger.error(
            f"An unexpected error occurred calling the API: {e}", exc_info=True
        )
        return f"⚠️ An unexpected error occurred: {str(e)}"


with gr.Blocks(title="Car Insurance FAQ Chatbot") as demo:
    gr.Markdown(
        """
        # 🚗 Car Insurance FAQ Chatbot

        Ask questions about car insurance in Australia and get instant answers!

        *Note: Ensure the `OPENAI_API_KEY` secret is set in the Hugging Face Space settings.*
        """
    )

    chatbot_ui = gr.Chatbot(
        label="Chat History",
        height=500,
        show_copy_button=True,
        render_markdown=True,
    )

    message_input = gr.Textbox(
        label="Your question",
        placeholder="Type your car insurance question here...",
        lines=2,
    )

    with gr.Row():
        submit_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear")

    gr.Examples(
        examples=[
            "What is the coverage for windscreen damage?",
            "How do I make a claim?",
            "What is the excess?",
            "Is my car covered if I drive interstate?",
            "What's the difference between comprehensive and third-party insurance?",
            "Can I choose my own repairer?",
            "Does my policy cover rental car costs after an accident?",
            "How does my driving record affect my premium?",
            "What happens if my car is written off?",
            "Are modifications to my car covered?",
        ],
        inputs=message_input,
        label="Examples",
    )

    # Gradio chat function
    def user_input(user_message, history):
        if not user_message.strip():
            # Return empty string for input clear, and unchanged history
            return "", history or []  # Ensure history is a list

        logger.info(f"User question: {user_message}")
        bot_response = call_api(user_message)
        logger.info(f"Bot response: {bot_response}")

        # Append interaction to history
        history = history or []  # Ensure history is initialized if None
        history.append((user_message, bot_response))

        # Clear the input field and return the updated history
        return "", history

    # Connect UI components to the function
    submit_btn.click(
        user_input,
        inputs=[message_input, chatbot_ui],
        outputs=[message_input, chatbot_ui],
    )

    message_input.submit(
        user_input,
        inputs=[message_input, chatbot_ui],
        outputs=[message_input, chatbot_ui],
    )

    # Clear button action
    clear_btn.click(
        lambda: (None, None), outputs=[message_input, chatbot_ui], queue=False
    )
