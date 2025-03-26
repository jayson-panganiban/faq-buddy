import json
import os

import openai
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="FAQ-bot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Replace with specific origins later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Models
class Question(BaseModel):
    text: str


class Answer(BaseModel):
    answer: str
    confidence: float = 0.0
    matched_question: str | None = None
    source_url: str | None = None
    brand: str | None = None


# FAQ data handling
class FAQRepository:
    def __init__(self):
        self.faqs = []
        self.faq_list = []
        self.load_faqs()

    def load_faqs(self):
        try:
            with open("data/faqs.json", "r") as f:
                self.faqs = json.load(f)
                self.faq_list = [item["question"] for item in self.faqs]
        except Exception as e:
            print(f"Error loading FAQs: {e}")
            self.faqs = []
            self.faq_list = []

    def get_faq_by_question(self, matched_question: str) -> dict | None:
        """Get full FAQ details including answer, URL and brand by question"""
        for faq in self.faqs:
            if faq["question"] == matched_question:
                return faq
        return None

    async def find_similar_faqs(
        self, question: str, top_n=5, api_key=None
    ) -> list[dict]:
        """Find the most semantically similar FAQs using embeddings"""
        try:
            # Ensure top_n is an integer
            if not isinstance(top_n, int):
                print(
                    f"Warning: top_n is not an integer: {type(top_n)}. Using default value 5."
                )
                top_n = 5

            if not self.faqs:
                print("No FAQs loaded")
                return []

            faq_texts = [faq["question"] for faq in self.faqs]

            # Set API key if provided
            current_api_key = openai.api_key
            if api_key:
                openai.api_key = api_key

            try:
                # Use OpenAI embedding model to get question embedding
                question_embedding_response = openai.embeddings.create(
                    input=question, model="text-embedding-ada-002"
                )
                question_embedding = question_embedding_response.data[0].embedding

                # Get embeddings for all FAQs
                faq_embeddings_response = openai.embeddings.create(
                    input=faq_texts, model="text-embedding-ada-002"
                )

                # Calculate cosine similarity between question and each FAQ
                import numpy as np
                from numpy import dot
                from numpy.linalg import norm

                similarities = []
                for i, faq_embed_data in enumerate(faq_embeddings_response.data):
                    faq_embedding = np.array(faq_embed_data.embedding)
                    question_embedding_np = np.array(question_embedding)

                    sim = dot(question_embedding_np, faq_embedding) / (
                        norm(question_embedding_np) * norm(faq_embedding)
                    )
                    similarities.append((self.faqs[i], sim))

                # Sort by similarity score and return top_n
                similarities.sort(key=lambda x: x[1], reverse=True)

                # Return the top N results
                return [item[0] for item in similarities[:top_n]]
            finally:
                # Restore original API key
                openai.api_key = current_api_key
        except Exception as e:
            print(f"Error finding similar FAQs: {e}")
            import traceback

            traceback.print_exc()
            return []


# OpenAI service
class OpenAIService:
    def __init__(self):
        # Default API key from environment
        self.default_api_key = os.getenv("OPENAI_API_KEY")

    async def match_question(
        self, question: str, faq_list: list[str], api_key=None
    ) -> dict:
        try:
            # Use provided API key or fall back to default
            current_api_key = api_key or self.default_api_key

            if not current_api_key:
                raise ValueError("No OpenAI API key provided")

            prompt = (
                "You are a car insurance FAQ assistant specializing in Australia. "
                "Given a user question, find the most semantically similar FAQ from the list.\n"
                "If the question has similar meaning to an FAQ (over 75% confidence) and relates to car insurance in Australia, return the exact FAQ text.\n"
                "Be lenient with typos and minor variations in wording.\n"
                "If the question is not about car insurance in Australia, respond with 'Out of scope'.\n"
                "If no good match is found but the question is in scope, return 'No match'.\n"
                "For each match, provide a confidence score between 0.0 and 1.0.\n\n"
                f"FAQs: {json.dumps(faq_list)}\n\n"
                f"User question: {question}\n\n"
                "Response format (JSON): {'matched_question': 'exact FAQ text or No match or Out of scope', 'confidence': float between 0.0 and 1.0}"
            )

            # Set API key for this request
            openai.api_key = current_api_key

            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.3,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content or "{}")
            return result
        except Exception as e:
            print(f"OpenAI API error: {e}")
            raise HTTPException(
                status_code=500, detail=f"OpenAI API error: {str(e)}"
            ) from e


# Dependency Injection
faq_repo = FAQRepository()
openai_service = OpenAIService()


def get_faq_repo():
    return faq_repo


def get_openai_service():
    return openai_service


# Get API key from header
async def get_api_key(x_openai_key: str = Header(None, alias="X-OpenAI-Key")):
    return x_openai_key


# Routes
@app.post("/ask", response_model=Answer)
async def ask_question(
    question: Question,
    faq_repo: FAQRepository = Depends(get_faq_repo),
    openai_service: OpenAIService = Depends(get_openai_service),
    api_key: str = Depends(get_api_key),
):
    if not question.text.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if not faq_repo.faq_list:
        raise HTTPException(status_code=500, detail="FAQ data not available")

    try:
        match_result = await openai_service.match_question(
            question.text, faq_repo.faq_list, api_key
        )
        matched_question = match_result.get("matched_question", "No match")
        confidence = match_result.get("confidence", 0.0)

        if matched_question == "Out of scope":
            return Answer(
                answer="Hi there! I'm a car insurance assistant specializing in Australia. "
                "It seems your question is outside my area of expertise. "
                "I'd be happy to help with any questions about car insurance policies, "
                "claims, or coverage options specific to Australia. How can I assist you today?",
                confidence=0.9,
                matched_question=None,
                source_url=None,
                brand=None,
            )

        # If we have a high confidence match, use it directly
        if matched_question != "No match" and confidence >= 0.7:
            matched_faq = faq_repo.get_faq_by_question(matched_question)
            if matched_faq:
                return Answer(
                    answer=matched_faq["answer"],
                    confidence=confidence,
                    matched_question=matched_question,
                    source_url=matched_faq.get("url"),
                    brand=matched_faq.get("brand"),
                )

        # If no good match or low confidence, use embedding similarity search
        similar_faqs = await faq_repo.find_similar_faqs(question.text, 5, api_key)

        # If we found similar FAQs, use the top match
        if similar_faqs and len(similar_faqs) > 0:
            top_match = similar_faqs[0]

            # Use the top match directly
            return Answer(
                answer=top_match["answer"],
                confidence=0.85,  # High confidence for embedding-based match
                matched_question=top_match["question"],
                source_url=top_match.get("url"),
                brand=top_match.get("brand"),
            )

        # If no similar FAQs found, provide a generic response
        return Answer(
            answer="Hello! Thanks for your question about car insurance in Australia. "
            "I don't have a specific answer for that question yet, but I'd love to help. "
            "Could you try rephrasing your question, or ask about a different aspect of "
            "Australian car insurance? I'm here to assist however I can!",
            confidence=0.0,
            matched_question=None,
            source_url=None,
            brand=None,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing question: {str(e)}"
        )


@app.get("/sources")
async def list_sources(
    faq_repo: FAQRepository = Depends(get_faq_repo),
    api_key: str = Depends(get_api_key),
):
    """Returns the available FAQ sources (brand) and their counts."""
    sources = {}
    for faq in faq_repo.faqs:
        brand = faq.get("brand", "unknown")
        if brand not in sources:
            sources[brand] = 0
        sources[brand] += 1

    return {"sources": sources, "total_faqs": len(faq_repo.faqs)}


@app.get("/health")
async def health_check(api_key: str = Depends(get_api_key)):
    """Check if the API is running and OpenAI API key is valid"""
    # If API key is provided, verify it works with OpenAI
    if api_key:
        try:
            # Set API key temporarily
            current_api_key = openai.api_key
            openai.api_key = api_key

            # Simple test call to OpenAI
            openai.models.list()

            # Restore original API key
            openai.api_key = current_api_key

            return {"status": "healthy", "faq_count": len(faq_repo.faqs)}
        except Exception as e:
            return {"status": "error", "message": f"OpenAI API key error: {str(e)}"}

    # If no API key provided, just check if the API is running
    return {"status": "healthy", "faq_count": len(faq_repo.faqs)}


@app.post("/reload-faqs")
async def reload_faqs(
    faq_repo: FAQRepository = Depends(get_faq_repo),
    api_key: str = Depends(get_api_key),
):
    faq_repo.load_faqs()
    return {"status": "success", "faq_count": len(faq_repo.faqs)}


@app.get("/about")
async def about(api_key: str = Depends(get_api_key)):
    """Returns information about the team and application."""
    return {
        "application": {
            "name": "FAQ Bot",
            "description": "A simple Python chatbot designed to answer car insurance related questions based on a predefined set of Frequently Asked Questions (FAQs) knowledge base and OpenAI's model.",
            "features": [
                "Intelligent Matching: Utilizes OpenAI completions to match user queries with existing FAQs",
                "Semantic Search: Uses embeddings to find similar questions when no exact match is found",
            ],
            "repository": "https://github.com/Andreymae/faq-bot",
        },
        "team": {"leader": "Andreymae", "members": ["Ratna", "Savinay"]},
        "version": "1.0.0",
    }
