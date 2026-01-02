import os
import uvicorn
import json        
import asyncio     
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

from engine import get_rag_chain, AMBIGUITY_MAP

load_dotenv()
OS_OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

llm = ChatOpenAI(
        model="xiaomi/mimo-v2-flash:free", 
        openai_api_key=OS_OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "http://localhost:3000", 
            "X-Title": "Local SOP Bot",             
        },
        temperature=0.0,
        streaming=True 
    )

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_chain = get_rag_chain()

class ChatRequest(BaseModel):
    text: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_query = request.text.strip()

    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "thanks", "thank you"]
    if user_query in greetings:
        return {
            "reply": "Hello! I am the SCOPE Chat Assistant. You can ask me about Industrial Visits, CCM meetings, Guest Lectures, Conference support etc.",
            "status": "success"
        }

    if len(user_query.split()) < 3: 
        intent_prompt = f"Is the following text a social greeting/small talk or a technical question about school policies? Answer ONLY 'Social' or 'Technical'. \nText: {user_query}"
        intent_result = llm.invoke(intent_prompt).content.strip()
        
        if "Social" in intent_result:
            return {
                "reply": "I'm here to help with SCOPE SOPs! Please ask a specific question about school activities.",
                "status": "success"
            }
    
    # --- AMBIGUITY CHECK ---
    for topic, info in AMBIGUITY_MAP.items():
        if topic in user_query.lower():
            has_tag = any(tag in user_query.lower() for tag in info["tags"])
            
            if not has_tag:
                return {
                    "reply": info["prompt"], 
                    
                    "status": "ambiguous"
                }
            
    async def event_generator():
        try:
            # We use astream for token-by-token output
            async for chunk in rag_chain.astream(user_query):
                # We send data in a format the frontend can easily parse
                yield f"data: {json.dumps({'text': chunk, 'status': 'streaming'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'text': str(e), 'status': 'error'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

'''   try:
        response = rag_chain.invoke(user_query)
        return {"reply": response, "status": "success"}
    except Exception as e:
        return {"reply": f"An error occurred: {str(e)}", "status": "error"}'''

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)