from fastapi import FastAPI, HTTPException
from models import QueryRequest, QueryResponse
from graph import app_graph

app = FastAPI(
    title="Zepto Support Assistant",
    description="Grounded GenAI Support Routing Service",
    version="1.0.0"
)

@app.get("/")
def health_check():
    return {"status": "ok", "service": "zepto-support-assistant"}

@app.post("/ask", response_model=QueryResponse)
def ask(payload: QueryRequest):
    if not payload.query or not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    initial_state = {
        "query": payload.query,
        "intent": "",
        "retrieved_docs": [],
        "retrieved_ids": [],
        "final_output": None
    }
    
    result = app_graph.invoke(initial_state)
    return result["final_output"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)
