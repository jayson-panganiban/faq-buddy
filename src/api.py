import hashlib
import json
import logging
import os

import numpy as np
import openai
from cachetools import TTLCache  # type: ignore
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from numpy import dot
from numpy.linalg import norm
from pydantic import BaseModel

# TODO: Implement a more structured evaluation framework for response quality.

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="FAQ-bot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Replace with specific origins later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Question(BaseModel):
    text: str


class Answer(BaseModel):
    answer: str
    confidence: float = 0.0
    matched_question: str | None = None
    source_url: str | None = None
    brand: str | None = None


class FAQRepository:
    """Handles loading and retrieving FAQ data."""

    def __init__(self, faq_path: str = "data/faqs.json"):
        self.faq_path = faq_path
        self.faqs: list[dict] = []
        self.faq_list: list[str] = []
        # TODO: Make cache size and TTL configurable.
        # TODO: Consider caching list of FAQ IDs instead of full objects.
        self.embedding_cache: TTLCache = TTLCache(maxsize=1000, ttl=86400)
        self.load_faqs()

    def load_faqs(self):
        try:
            logger.info(f"Attempting to load FAQs from: {self.faq_path}")
            with open(self.faq_path, "r", encoding="utf-8") as f:
                self.faqs = json.load(f)

            # Extract questions, handling potential missing keys
            self.faq_list = []
            for i, item in enumerate(self.faqs):
                if "question" not in item:
                    logger.warning(
                        f"FAQ item at index {i} is missing 'question' key. Skipping."
                    )
                    continue
                self.faq_list.append(item["question"])

            logger.info(
                f"Successfully loaded {len(self.faqs)} FAQs, {len(self.faq_list)} questions extracted."
            )

        except Exception as e:
            # Catch-all errors during loading
            logger.error(
                f"An unexpected error occurred loading FAQs: {e}", exc_info=True
            )
            self.faqs = []
            self.faq_list = []

    def get_faq_by_question(self, matched_question: str) -> dict | None:
        """Get full FAQ details including answer, URL and brand by question"""
        return next(
            (faq for faq in self.faqs if faq.get("question") == matched_question), None
        )

    async def find_similar_faqs(
        self, question: str, top_n: int = 5, api_key: str | None = None
    ) -> list[dict]:
        """Finds the most semantically similar FAQs using OpenAI embeddings."""
        # TODO: This method currently handles OpenAI API key logic. Ideally,
        # embedding generation might be decoupled into a separate service.

        if not isinstance(top_n, int) or top_n <= 0:
            logger.warning(f"Invalid top_n value: {top_n}. Using default value 5.")
            top_n = 5

        if not self.faqs:
            logger.warning("Attempted to find similar FAQs, but no FAQs are loaded.")
            return []

        # Extract FAQ texts safely, skipping items without a 'question'
        faq_texts = [faq["question"] for faq in self.faqs if "question" in faq]
        if not faq_texts:
            logger.warning("No valid FAQ questions available for embedding comparison.")
            return []

        # Map indices of faq_texts back to original self.faqs indices if needed,
        # but simpler here is to just work with the filtered list and map back later
        valid_faqs = [faq for faq in self.faqs if "question" in faq]

        cache_key = f"embedding_{hashlib.md5(question.encode()).hexdigest()}"
        # Use TTLCache's get method (returns None if key doesn't exist or expired)
        cached_result = self.embedding_cache.get(cache_key)
        if cached_result is not None:  # Explicit check for None
            logger.info("Using cached embedding for question")
            return cached_result

        client = openai.AsyncOpenAI(api_key=api_key)

        try:
            logger.info(
                f"Generating embeddings for question and {len(faq_texts)} FAQs."
            )
            # TODO: Make embedding model name configurable.
            # Use OpenAI embedding model to get question embedding
            question_embedding_response = await client.embeddings.create(
                input=question, model="text-embedding-ada-002"
            )
            question_embedding = np.array(question_embedding_response.data[0].embedding)

            # Get embeddings for all valid FAQs
            faq_embeddings_response = await client.embeddings.create(
                input=faq_texts, model="text-embedding-ada-002"
            )

            similarities = []
            for i, faq_embed_data in enumerate(faq_embeddings_response.data):
                faq_embedding = np.array(faq_embed_data.embedding)

                # Calculate cosine similarity
                # Check for zero vectors to avoid division by zero
                norm_q = norm(question_embedding)
                norm_faq = norm(faq_embedding)
                if norm_q == 0 or norm_faq == 0:
                    sim = 0.0  # Or handle as an error/warning
                else:
                    sim = dot(question_embedding, faq_embedding) / (norm_q * norm_faq)

                # Store the original FAQ dict along with its similarity score
                similarities.append((valid_faqs[i], sim))

            # Sort by similarity score (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)

            result = [item[0] for item in similarities[:top_n]]

            # Cache the result using dictionary-style assignment
            # TODO: Consider faq IDs instead of the full objects
            self.embedding_cache[cache_key] = result
            logger.info(f"Returning top {top_n} similar FAQs (cached for future use).")
            return result

        except openai.APIError as e:
            logger.error(
                f"OpenAI API error during embedding generation: {e}", exc_info=True
            )
            return []
        except Exception as e:
            logger.error(f"Error finding similar FAQs: {e}", exc_info=True)
            return []


class OpenAIService:
    def __init__(self):
        self.default_api_key = os.getenv("OPENAI_API_KEY")
        # TODO: Make cache sizes and TTLs configurable.
        self.question_match_cache = TTLCache(maxsize=5000, ttl=3600)
        # TODO: Consider using sorted source IDs instead of questions in the cache key.
        self.synthesis_cache = TTLCache(maxsize=2000, ttl=3600)

    async def match_question(
        self, question: str, faq_list: list[str], api_key: str
    ) -> dict:
        # Create a cache key based on the question and the current FAQ list state
        faq_hash = hashlib.md5(str(faq_list).encode()).hexdigest()
        cache_key = f"match_{hashlib.md5(question.encode()).hexdigest()}_{faq_hash}"

        cached_result = self.question_match_cache.get(cache_key)
        if cached_result is not None:  # Explicit check for None
            logger.info("Using cached question match result")
            return cached_result

        client = openai.AsyncOpenAI(api_key=api_key)

        prompt = f"""
You are an expert Australian car insurance FAQ matching system.
Your goal is to find the single best semantic match for a user's question from a given list of FAQs, or determine if the question is out of scope or has no match.

Instructions:
1.  **Analyze Scope:** Is the user's question about car insurance specifically in Australia?
    *   If NO: Respond ONLY with: {{"matched_question": "Out of scope", "confidence": 0.0}}
    *   If YES: Proceed to step 2.
2.  **Find Best Semantic Match:** Compare the user's question against EACH FAQ in the list below. Focus on the underlying meaning, intent, and context. Be lenient with typos and phrasing variations. Identify the FAQ that is the *closest* semantic match.
3.  **Evaluate Confidence:** Assign a confidence score (float between 0.0 and 1.0) representing how well the *best* match you found actually addresses the user's specific question. A high score (e.g., > 0.7) indicates a strong semantic match.
4.  **Format Response (JSON only):**
    *   If you found a strong match (high confidence): Return the **EXACT** text of the matched FAQ and your confidence score.
        Example: {{"matched_question": "What happens if I drive interstate?", "confidence": 0.92}}
    *   If the question is in scope, but no FAQ provides a good answer (low confidence for the best potential match): Return:
        {{"matched_question": "No match", "confidence": 0.0}}  # Use 0.0 confidence for 'No match'

**Crucial Rules:**
*   You MUST return the matched FAQ text exactly as it appears in the provided list. DO NOT rephrase or modify it.
*   Your entire response MUST be a single JSON object and nothing else. No introductory text or explanations.

**List of FAQs:**
{json.dumps(faq_list)}

**User question:**
{question}

**JSON Response:**
"""
        try:
            # TODO: Make LLM model name configurable.
            # TODO: Make max_tokens configurable.
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250,
                temperature=0.2,
                response_format={"type": "json_object"},
            )

            result_str = response.choices[0].message.content or "{}"

            # In case the model doesn't return perfect JSON
            try:
                result = json.loads(result_str)
                if not isinstance(
                    result.get("matched_question"), str
                ) or not isinstance(result.get("confidence"), (float, int)):
                    # Log the invalid structure before raising
                    logger.warning(
                        "LLM response JSON structure invalid: %s", result_str
                    )
                    raise ValueError("Invalid JSON structure received from LLM")

                # Ensure confidence is within bounds
                result["confidence"] = max(
                    0.0, min(1.0, float(result.get("confidence", 0.0)))
                )

            except (json.JSONDecodeError, ValueError) as json_err:
                logger.warning(
                    "Could not parse or validate LLM response JSON: %s. Raw response: %s",
                    json_err,
                    result_str,
                )
                # Fallback to default response
                result = {"matched_question": "No match", "confidence": 0.0}

            # Cache the result
            self.question_match_cache[cache_key] = result
            return result

        except openai.APIError as api_err:
            logger.error(
                "OpenAI API error during question matching: %s", api_err, exc_info=True
            )
            raise HTTPException(
                status_code=503,
                detail=f"Error communicating with AI service: {str(api_err)}",
            ) from api_err
        except Exception as e:
            # Catch any other unexpected errors during the process
            logger.error(
                "Unexpected error during OpenAI API call or processing in match_question: %s",
                e,
                exc_info=True,
            )
            # General server error for unexpected issues
            raise HTTPException(
                status_code=500, detail=f"Error matching question: {str(e)}"
            ) from e

    async def synthesize_answer(
        self, question_text: str, sources: list[dict], api_key: str
    ) -> str:
        """Synthesizes a final answer from provided sources using an LLM."""

        if not sources:
            # Should ideally not happen if called correctly, but good practice
            return "I couldn't find any relevant information to answer your question."

        # Create a cache key based on the question and the sources used
        # Using source questions as part of the key assumes their content defines the source context
        # TODO: Consider using ID for cache key
        source_questions = tuple(sorted([s.get("question", "") for s in sources]))
        cache_key_parts = (question_text, source_questions)
        cache_key = (
            f"synthesis_{hashlib.md5(str(cache_key_parts).encode()).hexdigest()}"
        )

        # Check cache first
        cached_result = self.synthesis_cache.get(cache_key)
        if cached_result is not None:
            logger.info(f"Using cached synthesized answer for: {question_text[:50]}...")
            return cached_result

        # Prepare source text for the prompt
        source_text = "\n\n".join(
            f"Source {i + 1} (Question: {s.get('question', 'N/A')}):\nAnswer: {s.get('answer', 'N/A')}"
            for i, s in enumerate(sources)
        )

        synthesis_prompt = f"""
Think of yourself as a knowledgeable and helpful Aussie car insurance assistant.
Your main task is to put together a single, clear, and comprehensive answer to the user's query, using *only* the source info provided below.

**User's Question:**
{question_text}

**Relevant Source Information:**
{source_text}

**Instructions for Synthesizing the Answer:**
1.  **Combine & Consolidate:** Blend the info from all sources into one easy-to-read response. If bits are repeated, just state it once, clearly.
2.  **Answer the Question:** Make sure the final answer properly tackles what the user asked.
3.  **Stick to the Script:** Use ONLY the information given in the 'Relevant Source Information'. Don't add anything extra or make stuff up.
4.  **Tone:** Keep it helpful, straightforward, and professional. Like you're explaining it to a mate, but clearly.
5.  **No Behind-the-Scenes Talk:** Don't mention the sources, how you made the answer, or these instructions. Just give the answer itself.

**Synthesized Answer:**
"""
        client = openai.AsyncOpenAI(api_key=api_key)

        try:
            logger.info(f"Synthesizing answer for: {question_text[:50]}...")
            # TODO: Make LLM model name configurable.
            # TODO: Make max_tokens configurable.
            synthesis_response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": synthesis_prompt}],
                max_tokens=250,
                temperature=0.5,  # TODO: Make temperature configurable.
            )
            synthesized_answer = (
                synthesis_response.choices[0].message.content
                or "Sorry, I encountered an issue generating a detailed answer. Please try again."
            )
            logger.info("Answer synthesis complete.")

            # Store the result in the cache
            self.synthesis_cache[cache_key] = synthesized_answer
            return synthesized_answer

        except openai.APIError as api_err:
            logger.error(
                "OpenAI API error during answer synthesis: %s", api_err, exc_info=True
            )
            # Re-raise as HTTPException so the endpoint can handle it
            raise HTTPException(
                status_code=503,
                detail=f"Error communicating with AI service during synthesis: {str(api_err)}",
            ) from api_err
        except Exception as e:
            logger.error(
                "Unexpected error during answer synthesis: %s", e, exc_info=True
            )
            # Re-raise as HTTPException
            raise HTTPException(
                status_code=500, detail=f"Error synthesizing answer: {str(e)}"
            ) from e


faq_repo = FAQRepository()
openai_service = OpenAIService()


def get_faq_repo():
    return faq_repo


def get_openai_service():
    return openai_service


# Get API key from header
# TODO: Implement a more secure API key management strategy
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
    # TODO: Explore using asyncio.gather to run match_question and find_similar_faqs concurrently for potential performance improvement.
    if not question.text.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if not faq_repo.faqs:
        faq_repo.load_faqs()  # Attempt to reload if empty
        if not faq_repo.faqs:
            raise HTTPException(
                status_code=503,
                detail="FAQ data is currently unavailable. Please try again later.",
            )

    try:
        # 1. Initial Matching
        match_result = await openai_service.match_question(
            question.text, faq_repo.faq_list, api_key
        )
        matched_question_text = match_result.get("matched_question", "No match")
        initial_confidence = match_result.get("confidence", 0.0)

        # Handle out-of-scope questions early
        if matched_question_text == "Out of scope":
            return Answer(
                answer="G'day! I'm your assistant for Aussie car insurance questions. "
                "Looks like that query is a bit outside my wheelhouse, sorry! "
                "I can definitely help with questions about policies, claims, or cover specific to Australia, though. "
                "What can I help you with on that front?",
                confidence=0.9,  # High confidence in the 'out of scope' assessment
                matched_question=None,
                source_url=None,
                brand=None,
            )

        # 2. Gather Relevant Sources
        sources = []
        primary_source = None  # Keep track of the best initial match

        # Add the primary matched FAQ if it's a good match
        # TODO: Make confidence threshold (0.7) configurable.
        if matched_question_text != "No match" and initial_confidence >= 0.7:
            matched_faq = faq_repo.get_faq_by_question(matched_question_text)
            if matched_faq:
                sources.append(matched_faq)
                primary_source = matched_faq  # This is our best direct match

        # Find additional semantically similar FAQs via embeddings
        # Limit to top 3-5 to avoid overwhelming the synthesis prompt
        # TODO: Make top_n for similar FAQs configurable.
        similar_faqs = await faq_repo.find_similar_faqs(
            question.text, top_n=3, api_key=api_key
        )

        # Add unique similar FAQs to the sources list
        for faq in similar_faqs:
            # Avoid adding duplicates if the embedding search found the primary match again
            if not any(s["question"] == faq["question"] for s in sources):
                sources.append(faq)

        # Handle cases where no relevant information is found
        if not sources:
            return Answer(
                answer="G'day! Thanks for your question about Aussie car insurance. "
                "I've had a look through my info, but couldn't find a specific answer for that one right now. "
                "Could you maybe try asking it a different way, or ask about another aspect of cover? "
                "Happy to help with other Aussie car insurance topics!",
                confidence=0.0,  # No confidence as no source was found
                matched_question=None,
                source_url=None,
                brand=None,
            )

        # 3. Synthesize Answer
        synthesized_answer = await openai_service.synthesize_answer(
            question.text, sources, api_key
        )

        # 4. Format Final Response
        # Prepare source citations (only if multiple sources were used)
        source_citations = ""
        if len(sources) > 1:
            source_refs = []
            for i, source in enumerate(sources, 1):
                brand = source.get("brand", "Unknown Source")
                url = source.get("url")
                ref = f"[{i}] {brand}" + (f": {url}" if url else "")
                source_refs.append(ref)
            source_citations = "\n\nSources:\n" + "\n".join(source_refs)

        final_answer_text = synthesized_answer + source_citations

        # Determine the best metadata to return (use primary match if available)
        response_matched_question = (
            primary_source["question"] if primary_source else sources[0]["question"]
        )
        response_source_url = (
            primary_source.get("url") if primary_source else sources[0].get("url")
        )
        response_brand = (
            primary_source.get("brand") if primary_source else sources[0].get("brand")
        )
        # Confidence reflects the synthesis process based on found sources
        # Using a high value assumes the synthesis is reliable if good sources were found.
        # Alternatively, could be derived from initial_confidence or source similarities.
        # TODO: Consider a more nuanced confidence calculation based on source quality/similarity.
        response_confidence = (
            0.85  # Slightly lower than pure match, acknowledging synthesis step
        )

        return Answer(
            answer=final_answer_text,
            confidence=response_confidence,
            matched_question=response_matched_question,
            source_url=response_source_url,
            brand=response_brand,
        )

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions directly (e.g., from match_question)
        raise http_exc
    except Exception as e:
        print(f"Unexpected error processing question: {e}")
        import traceback

        traceback.print_exc()
        # Return a generic server error
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing your question.",
        ) from e


@app.get("/sources")
async def list_sources(
    faq_repo: FAQRepository = Depends(get_faq_repo),
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
async def health_check(
    api_key: str = Depends(get_api_key),
):  # TODO: Consider removing API key dependency for basic health check.
    """Check if the API is running and OpenAI API key is valid"""
    # If API key is provided, verify it works with OpenAI
    if api_key:
        try:
            # TODO: Use the AsyncOpenAI client for consistency.
            current_api_key = openai.api_key
            openai.api_key = api_key

            openai.models.list()  # Test connectivity and auth

            openai.api_key = current_api_key

            return {"status": "healthy", "faq_count": len(faq_repo.faqs)}
        except Exception as e:
            # Attempt to restore key even if the test failed
            openai.api_key = current_api_key
            return {"status": "error", "message": f"OpenAI API key error: {str(e)}"}

    # If no API key provided, just check if the API is running
    return {"status": "healthy", "faq_count": len(faq_repo.faqs)}


@app.post("/reload-faqs")
async def reload_faqs(
    faq_repo: FAQRepository = Depends(get_faq_repo),
    openai_service: OpenAIService = Depends(get_openai_service),
    # TODO: Secure this endpoint, perhaps with a dedicated admin key or role-based access.
):
    """Reloads FAQs from the source file and clears related caches."""
    logger.info("Reloading FAQs and clearing caches...")
    faq_repo.load_faqs()
    # Clear all relevant caches
    faq_repo.embedding_cache.clear()
    openai_service.question_match_cache.clear()
    openai_service.synthesis_cache.clear()
    logger.info("FAQ reload complete. Caches cleared.")
    return {"status": "success", "faq_count": len(faq_repo.faqs)}


async def about():
    """Returns information about the team and application."""
    return {
        "application": {
            "name": "FAQ Buddy (Aussie Car Insurance)",
            "description": "A smart chatbot helping answer Australian car insurance questions using FAQs and AI.",
            "features": [
                "AI Matching: Uses language models to find the best matching FAQ for your question.",
                "Semantic Search: Finds related questions even if your wording is different.",
                "Answer Synthesis: Combines info from multiple relevant FAQs for a comprehensive answer.",
            ],
            "repository": "https://github.com/jayson-panganiban/faq-buddy",
        },
        "team": {
            "leader": "Andreymae",
            "members": ["Ratna", "Savinay"],
        },
        "version": "1.0.0",
    }


if __name__ == "__main__":
    import uvicorn

    # TODO: Make host and port configurable via environment variables or CLI args.
    uvicorn.run(app, host="0.0.0.0", port=8000)
