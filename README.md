---
title: FAQ Mate
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
secrets:
  - OPENAI_API_KEY
---

# 🚗 FAQ Mate - Car Insurance FAQ Chatbot

A simple chatbot that answers questions about Australian car insurance.

## How it Works

1.  The Gradio interface captures the user's question.
2.  It sends the question to the FastAPI backend (`/ask` endpoint).
3.  The FastAPI backend:
    - Uses OpenAI embeddings to find relevant FAQs from `data/faqs.json`.
    - Uses an OpenAI chat model to synthesize an answer based on the relevant FAQs.
    - Returns the synthesized answer, confidence score, and source information.
4.  The Gradio interface displays the answer to the user.

## Configuration

- **SDK:** Docker
- **Application Port:** 7860 (Gradio interface)
- **Secrets:** Requires an `OPENAI_API_KEY` to be set in the Space secrets. Add your **OpenAI** API key in the Space settings under "Secrets".

## Running Locally (Optional)

1.  **Build the Docker image:**
    ```bash
    docker build -t faq-mate .
    ```
2.  **Run the Docker container:**

    ```bash
    docker run -p 7860:7860 -p 8000:8000 -e OPENAI_API_KEY="your_openai_api_key" faq-mate
    ```

    Replace `"your_openai_api_key"` with your actual key.

    - Access the Gradio UI at `http://localhost:7860`
    - Access the FastAPI docs at `http://localhost:8000/docs`
