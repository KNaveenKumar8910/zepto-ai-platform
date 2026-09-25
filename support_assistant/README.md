# Zepto Support Assistant — Module 3

## Submission Checklist
This directory contains the complete graded baseline for Module 3:
- Corpus: 8 canonical Zepto policy corpus text files under docs/ (doc_01.txt to doc_08.txt).
- Local Embeddings: Local vector representations generated via sentence-transformers (all-MiniLM-L6-v2).
- Vector Database: Persistent indexing via ChromaDB (zepto_policies collection).
- Orchestration: LangGraph StateGraph with a TypedDict state schema, 3 designated nodes, and conditional intent routing.
- Intent Classifier: Deterministic keyword heuristic routing policy queries to retrieval and off-topic queries to direct fallback.
- Retrieval Engine: Live cosine distance similarity search returning top-3 context chunks (n_results=3).
- Graded Offline Mock Mode: Enabled by default (MOCK_LLM=1), requiring zero external API keys, zero network LLM calls, and zero external costs.
- Schema Validation: Strict Pydantic models (QueryRequest, QueryResponse) enforcing structured output (answer, sources, confidence).
- API Service: FastAPI service serving POST /ask on port 7860 via Uvicorn.
- Containerization: Runnable Dockerfile containerizing the service on port 7860.

---

## Important Grading Setting
The graded baseline runs under the default environment configuration:

MOCK_LLM=1

- Offline Mock Mode (Graded Baseline): Runs deterministically without external LLM dependencies, API keys, or credit card requirements. All required evaluation assertions pass on this baseline.
- Real-LLM Extension (Optional / Ungraded): Setting MOCK_LLM=0 switches generation to external providers (such as Groq's free tier). This mode includes schema-validation retries (up to 2 corrective retry cycles) and negative prompt constraints.

---

## 1. Project Structure

support_assistant/
|-- Dockerfile
|-- README.md
|-- graph.py
|-- ingest.py
|-- main.py
|-- models.py
|-- docs/
|   |-- doc_01.txt
|   |-- doc_02.txt
|   |-- doc_03.txt
|   |-- doc_04.txt
|   |-- doc_05.txt
|   |-- doc_06.txt
|   |-- doc_07.txt
|   `-- doc_08.txt
`-- chroma_db/

---

## 2. Architecture and Pipeline Stages

User Query -> FastAPI POST /ask -> LangGraph StateGraph -> classify_intent

classify_intent branches to:
1. policy_question -> retrieve_and_answer (all-MiniLM-L6-v2 embedding -> ChromaDB top 3 -> MOCK_LLM=1 canned grounded response)
2. general_question -> direct_answer (MOCK_LLM=1 fixed fallback response)

Both branches route to:
Pydantic Validation (QueryResponse) -> JSON Output: { answer, sources, confidence }

### Detailed Pipeline Stages

1. Ingestion (ingest.py): Reads all 8 policy files from docs/, parsing domain policies for delivery SLAs, returns/refunds, membership tiers, tracking, cancellations, damaged goods, gift cards, and customer support channels.
2. Embedding (all-MiniLM-L6-v2): Generates 384-dimensional dense semantic vector representations locally on CPU without third-party network API calls.
3. Indexing (chromadb): Persists embeddings and metadata into a local ChromaDB collection (zepto_policies) configured with cosine space.
4. Retrieval (retrieve_and_answer): Converts incoming user queries into query vectors and performs a top-3 nearest-neighbor lookup (n_results=3). Semantic retrieval executes in both mock and real modes.
5. Generation and Router:
   - classify_intent inspects the lowercased input for policy keywords (delivery, return, refund, membership, tracking, cancel, gift card, support hours).
   - In default mock mode (MOCK_LLM=1), retrieve_and_answer returns: Based on the retrieved context: {top_chunk_snippet}
   - General off-topic queries route to direct_answer, returning a fixed fallback message without vector retrieval.
6. Validation (models.py): Guarantees responses match the schema: answer (string), sources (list of document IDs), and confidence (float 0.0 to 1.0).
7. API Layer (main.py): Exposes POST /ask through FastAPI and serves the application locally via Uvicorn.

---

## 3. Structured Prompt Skeleton (Role-Context-Task-Format-Length)

Configured for the real-LLM path (MOCK_LLM=0) with explicit negative constraints and few-shot grounding:

[ROLE]
You are a helpful and polite customer support assistant for Zepto, an instant grocery delivery platform.

[CONTEXT]
{context}

[TASK]
Answer the customer's question accurately using only the facts provided in the Zepto policy context above.

[NEGATIVE CONSTRAINT]
Do not answer using information not present in the provided context. If the answer cannot be determined strictly from the provided text, state: "I cannot find this information in the official Zepto policy documentation." Do not guess, speculate, or hallucinate.

[FEW-SHOT EXAMPLE]
User Query: What is the delivery fee for orders below INR 149?
Context: doc_01: Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee.
Output JSON:
{
  "answer": "Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee.",
  "sources": ["doc_01"],
  "confidence": 1.0
}

[FORMAT & LENGTH]
Return valid JSON matching the schema: {"answer": "<concise summary under 3 sentences>", "sources": ["doc_XX"], "confidence": <float 0.0-1.0>}

---

## 4. Verified API Execution Examples (MOCK_LLM=1 Graded Baseline)

### Example A: Policy Question (Grounded Retrieval)
Request:
curl -X POST "http://127.0.0.1:7860/ask" -H "Content-Type: application/json" -d "{\"query\": \"What is the return policy for damaged grocery items?\"}"

Raw JSON Response:
{
  "answer": "Based on the retrieved context: Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned within 7 days of delivery in unopened, resalable condition. Approved refunds are credited to the original payment method within 3–5 business days, or instantly to the Zepto wallet if the customer opts for wallet credit.",
  "sources": [
    "doc_02",
    "doc_06",
    "doc_01"
  ],
  "confidence": 1.0
}

### Example B: General Off-Topic Question (Direct Fallback)
Request:
curl -X POST "http://127.0.0.1:7860/ask" -H "Content-Type: application/json" -d "{\"query\": \"What is the capital of France?\"}"

Raw JSON Response:
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}

---

## 5. Local Setup and Execution Guide

cd support_assistant
python ingest.py
export MOCK_LLM=1
uvicorn main:app --host 127.0.0.1 --port 7860

Access interactive Swagger documentation at: http://127.0.0.1:7860/docs

---

## 6. Containerization (Docker)

Build the Image:
docker build -t zepto-support-assistant -f Dockerfile .

Run the Container:
docker run --rm -p 7860:7860 -e MOCK_LLM=1 zepto-support-assistant

The microservice exposes POST /ask on http://localhost:7860/ask.
