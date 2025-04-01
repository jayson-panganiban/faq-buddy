import os

import gradio as gr
import requests  # type: ignore
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")


class FAQChatbot:
    def __init__(self):
        self.api_url = API_URL
        self.api_key = None

    def set_api_key(self, api_key: str) -> dict:
        """Set the OpenAI API key and return health status"""
        self.api_key = api_key
        return self.check_api_health()

    def check_api_health(self) -> dict:
        """Check if the API is running and return status"""
        try:
            headers = {"X-OpenAI-Key": self.api_key} if self.api_key else {}
            response = requests.get(
                f"{self.api_url}/health", headers=headers, timeout=10.0
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {
                "status": "error",
                "message": f"Unexpected error during health check: {str(e)}",
            }

    def ask_question(self, question: str) -> dict:
        """Send question to API and get answer"""
        if not self.api_key:
            return {
                "answer": (
                    "⚠️ Error: OpenAI API key not found."
                    " Please set the OPENAI_API_KEY environment variable."
                ),
                "confidence": 0.0,
                "matched_question": None,
                "source_url": None,
                "brand": None,
            }

        try:
            headers = {"X-OpenAI-Key": self.api_key}
            response = requests.post(
                f"{self.api_url}/ask", json={"text": question}, headers=headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {
                "answer": f"An unexpected error occurred: {str(e)}",
                "confidence": 0.0,
                "matched_question": None,
                "source_url": None,
                "brand": None,
            }

    def format_response(self, response: dict) -> str:
        """Format the API response for display in the chatbot"""
        answer = response.get("answer", "No answer provided")
        confidence = response.get("confidence", 0.0)
        matched_question = response.get("matched_question")
        source_url = response.get("source_url")
        brand = response.get("brand")

        # Format the response with metadata
        formatted = f"{answer}"

        # Add metadata section if we have any relevant info
        metadata = []
        # Only show confidence if it's meaningful (e.g., > 0 and not an error message)
        if confidence > 0 and "Error" not in answer and "⚠️" not in answer:
            confidence_pct = f"{confidence * 100:.1f}%"
            metadata.append(f"Confidence: {confidence_pct}")

        if (
            matched_question
            and matched_question != "Synthesized from similar questions"
        ):
            metadata.append(f'Matched question: "{matched_question}"')
        elif matched_question:  # Handles the "Synthesized..." case
            metadata.append(f"Response: {matched_question}")

        if brand:
            metadata.append(f"Source: {brand}")

        if source_url:
            metadata.append(f"[More information]({source_url})")

        # Add metadata as smaller text if we have any
        if metadata:
            formatted += "\n\n---\n" + "\n".join(metadata)

        return formatted

    def process_message(self, message: str, history: list) -> str:
        """Process user message and return a response"""
        if not self.api_key:
            return "⚠️ Please enter your OpenAI API key in the settings panel first."

        response = self.ask_question(message)
        formatted_response = self.format_response(response)

        return formatted_response


chatbot = FAQChatbot()

with gr.Blocks(title="Car Insurance FAQ Chatbot") as demo:
    gr.Markdown(
        """
        # 🚗 Car Insurance FAQ Chatbot
        
        Ask questions about car insurance in Australia and get instant answers!
        """
    )

    with gr.Accordion("⚙️ OpenAI API Key (Required)", open=True):
        api_key_input = gr.Textbox(
            label="API Key",
            placeholder="Enter your OpenAI API key (starts with sk-...)",
            type="password",
        )
        status_info = gr.Markdown("⚠️ Please enter your OpenAI API key to start")

        def update_api_key(api_key):
            if not api_key or not api_key.startswith("sk-"):
                chatbot.api_key = None
                return "⚠️ Please enter a valid OpenAI API key starting with 'sk-'"

            # Set the API key and check health
            health_status = chatbot.set_api_key(api_key)

            if health_status.get("status") == "healthy":
                return f"✅ API key valid! FAQs Loaded: {health_status.get('faq_count', 0)}"
            else:
                return f"⚠️ API connection error: {health_status.get('message', 'Unknown error')}"

        # Update status when API key changes
        api_key_input.change(
            update_api_key, inputs=[api_key_input], outputs=status_info
        )

    chatbot_ui = gr.Chatbot(
        label="Chat History",
        height=500,
        show_copy_button=True,
        render_markdown=True,
        type="messages",
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

    def user_input(user_message, history):
        if not user_message.strip():
            # Return empty string for input clear, and unchanged history
            return "", history

        bot_response = chatbot.process_message(user_message, history)

        if not isinstance(
            history, list
        ):  # Handle potential initial None or other types
            history = []
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": bot_response})

        # Clear the input field and return the updated history
        return "", history

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

    clear_btn.click(lambda: ("", []), outputs=[message_input, chatbot_ui])


if __name__ == "__main__":
    demo.launch(share=False)
