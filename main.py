from fastapi import FastAPI, HTTPException
from httpx import AsyncClient
import os
from dotenv import load_dotenv
from typing import List, Dict  
import uuid

from agent import chatbot_agent, weather_agent, Deps
from schemas import ChatRequest, ChatResponse, WeatherRequest, WeatherResponse
from db import init_db, save_message, get_history

# Load .env
load_dotenv()

geo_api_key = os.getenv("GEO_API_KEY")
weather_api_key = os.getenv("WEATHER_API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")

if gemini_key:
    os.environ["GOOGLE_API_KEY"] = gemini_key 

app = FastAPI(
    title="Weather + Chatbot API",
    description="Chatbot with weather agent powered by Gemini LLM via Pydantic-AI",
)

# Initialize database
init_db()

def format_history_as_prompt(history: List[Dict[str, str]]) -> str:
    lines = []
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        role_label = "User" if role.lower() == "user" else "Assistant"
        lines.append(f"{role_label}: {content}")
    lines.append("Assistant:")
    return "\n".join(lines)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, session_id: str = None):
    """
    Chat endpoint with permanent conversation history stored in SQLite.
    """
    # Generate session_id if not provided
    if not session_id:
        session_id = str(uuid.uuid4())

    async with AsyncClient() as client:
        deps = Deps(client=client, geo_api_key=geo_api_key or "", weather_api_key=weather_api_key or "")

        try:
            # Save user message to DB
            save_message(session_id, "user", request.query)

            # Load history from DB
            history = get_history(session_id)

            # Format prompt for chatbot
            prompt = format_history_as_prompt(history)

            # Get chatbot reply
            chat_result = await chatbot_agent.run(prompt)
            bot_reply = (chat_result.output or "").strip()

            if isinstance(bot_reply, (dict, list)):
                bot_reply = str(bot_reply)

            # If weather request
            if bot_reply.upper().startswith("WEATHER_QUERY"):
                weather_result = await weather_agent.run(request.query, deps=deps)
                weather_text = weather_result.output or "No weather info available."
                save_message(session_id, "assistant", weather_text)
                return ChatResponse(response=weather_text)

            # Save assistant reply to DB
            save_message(session_id, "assistant", bot_reply)

            return ChatResponse(response=bot_reply)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Chat failed: {e}")


@app.post("/weather", response_model=WeatherResponse)
async def get_weather(request: WeatherRequest):
    async with AsyncClient() as client:
        deps = Deps(
            client=client,
            geo_api_key=geo_api_key or "",
            weather_api_key=weather_api_key or "",
        )
        try:
            result = await weather_agent.run(request.query, deps=deps)
            output = result.output or "No weather info returned."
            return WeatherResponse(response=output)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Weather query failed: {e}")
