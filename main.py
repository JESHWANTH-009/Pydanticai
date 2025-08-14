from fastapi import FastAPI, HTTPException
from httpx import AsyncClient
import os
from dotenv import load_dotenv
from typing import List, Dict  # <-- Missing import

from agent import chatbot_agent, weather_agent, Deps
from schemas import ChatRequest, ChatResponse, WeatherRequest, WeatherResponse

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

# In-memory conversation store
conversation_history: List[Dict[str, str]] = []


def format_history_as_prompt(history: List[Dict[str, str]]) -> str:
    """
    Convert history into a readable conversation transcript for the model.
    """
    lines = []
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        role_label = "User" if role.lower() == "user" else "Assistant"
        lines.append(f"{role_label}: {content}")
    lines.append("Assistant:")
    return "\n".join(lines)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    async with AsyncClient() as client:
        deps = Deps(
            client=client,
            geo_api_key=geo_api_key or "",
            weather_api_key=weather_api_key or "",
        )

        try:
            # Append user message to history
            conversation_history.append({"role": "user", "content": request.query})

            # Format full history into prompt
            prompt = format_history_as_prompt(conversation_history)

            # Get chatbot reply
            chat_result = await chatbot_agent.run(prompt)
            bot_reply = (chat_result.output or "").strip()

            if isinstance(bot_reply, (dict, list)):
                bot_reply = str(bot_reply)

            # Handle weather query case
            if bot_reply.upper().startswith("WEATHER_QUERY"):
                weather_result = await weather_agent.run(request.query, deps=deps)
                weather_text = weather_result.output or "No weather info available."
                conversation_history.append({"role": "assistant", "content": weather_text})
                return ChatResponse(response=weather_text)

            # Normal reply
            conversation_history.append({"role": "assistant", "content": bot_reply})
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
