from fastapi import FastAPI
from pydantic import BaseModel
from agents.orchestrator import workflow

class ChatRequest(BaseModel):
    user_input: str

app = FastAPI()

@app.get("/health")
async def health_status():
    return {"API": "successfull"}


@app.post("/user")
async def get_user_input(user_request: ChatRequest):
    user_data = user_request.user_input
    # LangGraph expects {"key_name": value}
    initial_state = {"user_input": user_data}
    response = workflow.invoke(initial_state)
    return response