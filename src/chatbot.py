import os

import gradio as gr
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
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
            response = requests.get(f"{self.api_url}/health", headers=headers)
            return response.json()
        except Exception as e:
            return {"status": "error", "message": f"API connection error: {str(e)}"}

    def get_sources(self) -> dict:
        """Get available FAQ sources from the API"""
        try:
            headers = {"X-OpenAI-Key": self.api_key} if self.api_key else {}
            response = requests.get(f"{self.api_url}/sources", headers=headers)
            return response.json()
        except Exception as e:
            return {"sources": {}, "total_faqs": 0, "error": str(e)}

    def reload_faqs(self) -> str:
        """Trigger FAQ reload in the API"""
        try:
            headers = {"X-OpenAI-Key": self.api_key} if self.api_key else {}
            response = requests.post(f"{self.api_url}/reload-faqs", headers=headers)
            result = response.json()
            return f"✅ FAQs reloaded successfully. Total FAQs: {result.get('faq_count', 0)}"
        except Exception as e:
            return f"❌ Failed to reload FAQs: {str(e)}"

    def ask_question(self, question: str) -> dict:
        """Send question to API and get answer"""
        if not self.api_key:
            return {
                "answer": "Please enter your OpenAI API key first.",
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
            if response.status_code == 200:
                return response.json()
            else:
                error_detail = response.json().get("detail", "Unknown error")
                return {
                    "answer": f"Error: {error_detail}",
                    "confidence": 0.0,
                    "matched_question": None,
                    "source_url": None,
                    "brand": None,
                }
        except Exception as e:
            return {
                "answer": f"Error connecting to FAQ service: {str(e)}",
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

        # Add metadata section if we have any
        metadata = []
        if confidence > 0:
            # Format confidence as percentage
            confidence_pct = f"{confidence * 100:.1f}%"
            metadata.append(f"Confidence: {confidence_pct}")

        if (
            matched_question
            and matched_question != "Synthesized from similar questions"
        ):
            metadata.append(f'Matched question: "{matched_question}"')
        elif matched_question:
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

        # Handle special commands
        if message.strip().lower() == "/reload":
            return self.reload_faqs()

        if message.strip().lower() == "/sources":
            sources_info = self.get_sources()
            sources_text = "Available FAQ Sources:\n\n"
            for brand, count in sources_info.get("sources", {}).items():
                sources_text += f"- {brand}: {count} FAQs\n"
            sources_text += f"\nTotal FAQs: {sources_info.get('total_faqs', 0)}"
            return sources_text

        if message.strip().lower() == "/health":
            health = self.check_api_health()
            return f"API Status: {health.get('status', 'unknown')}\nFAQ Count: {health.get('faq_count', 0)}"

        if message.strip().lower() == "/about":
            try:
                headers = {"X-OpenAI-Key": self.api_key} if self.api_key else {}
                response = requests.get(f"{self.api_url}/about", headers=headers)
                if response.status_code == 200:
                    about_data = response.json()
                    app_info = about_data.get("application", {})
                    team_info = about_data.get("team", {})

                    about_text = "## About FAQ Bot\n\n"
                    about_text += f"{app_info.get('description', '')}\n\n"
                    about_text += "### Features\n"
                    for feature in app_info.get("features", []):
                        about_text += f"- {feature}\n"

                    about_text += "\n### Team\n"
                    about_text += f"**Team Leader:** {team_info.get('leader', '')}\n"
                    about_text += (
                        f"**Members:** {', '.join(team_info.get('members', []))}\n\n"
                    )
                    about_text += f"**Version:** {about_data.get('version', '1.0.0')}\n"
                    about_text += f"**Repository:** {app_info.get('repository', '')}"

                    return about_text
                else:
                    return (
                        f"❌ Error retrieving about information: {response.status_code}"
                    )
            except Exception as e:
                return f"❌ Error connecting to API: {str(e)}"

        # Process regular questions
        response = self.ask_question(message)
        formatted_response = self.format_response(response)

        return formatted_response


def create_chatbot_interface():
    chatbot = FAQChatbot()

    # Create the interface
    with gr.Blocks(title="Car Insurance FAQ Chatbot") as interface:
        gr.Markdown(
            """
            # 🚗 Car Insurance FAQ Chatbot
            
            Ask questions about car insurance in Australia and get instant answers!
            
            *Special commands:*
            - `/reload` - Reload FAQ database
            - `/sources` - List available FAQ sources
            - `/health` - Check API health
            - `/about` - Display team and application information
            """
        )

        # API Key input
        with gr.Accordion("⚙️ OpenAI API Key (Required)", open=True):
            api_key_input = gr.Textbox(
                label="API Key",
                placeholder="Enter your OpenAI API key (starts with sk-...)",
                type="password",
            )
            status_info = gr.Markdown("⚠️ Please enter your OpenAI API key to start")

            # Function to validate and set the API key
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

        # Chat interface - Updated to use the 'messages' type instead of deprecated 'tuples'
        chatbot_ui = gr.Chatbot(
            label="Chat History",
            height=500,
            show_copy_button=True,
            render_markdown=True,
            type="messages",  # Specify the message format to avoid deprecation warning
        )

        message_input = gr.Textbox(
            label="Your question",
            placeholder="Type your car insurance question here...",
            lines=2,
        )

        with gr.Row():
            submit_btn = gr.Button("Send", variant="primary")
            clear_btn = gr.Button("Clear")

        def user_input(user_message, history):
            if not user_message.strip():
                return "", history

            # Get bot response
            bot_response = chatbot.process_message(user_message, history)

            # Use the new message format with 'role' and 'content' keys
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": bot_response})

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

        # Updated clear function to return empty list for messages format
        clear_btn.click(lambda: ("", []), outputs=[message_input, chatbot_ui])

    return interface
