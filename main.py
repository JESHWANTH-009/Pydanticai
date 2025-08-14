from fastapi import FastAPI, HTTPException
from httpx import AsyncClient
import os
from dotenv import load_dotenv

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

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint:
    - Ask chatbot agent for response.
    - If response is 'WEATHER_QUERY', call weather_agent.
    """
    async with AsyncClient() as client:
        deps = Deps(client=client, geo_api_key=geo_api_key or "", weather_api_key=weather_api_key or "")
        try:
            chat_result = await chatbot_agent.run(request.query)
            out = (chat_result.output or "").strip()

            if out.upper() == "WEATHER_QUERY":
                weather_result = await weather_agent.run(request.query, deps=deps)
                return ChatResponse(response=weather_result.output or "No weather info available.")
            return ChatResponse(response=out)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Chat failed: {e}")


@app.post("/weather", response_model=WeatherResponse)
async def get_weather(request: WeatherRequest):
    """
    Direct weather query endpoint.
    """
    async with AsyncClient() as client:
        deps = Deps(client=client, geo_api_key=geo_api_key or "", weather_api_key=weather_api_key or "")
        try:
            result = await weather_agent.run(request.query, deps=deps)
            return WeatherResponse(response=result.output or "No weather info returned.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Weather query failed: {e}")
