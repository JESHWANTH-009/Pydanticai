from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from pydantic_ai import Agent, RunContext
from httpx import AsyncClient
from pydantic import BaseModel
from fastapi import HTTPException
import os
from dotenv import load_dotenv

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY missing in .env")
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# -----------------------------
# Dependency container for tools
# -----------------------------
@dataclass
class Deps:
    client: AsyncClient
    geo_api_key: str
    weather_api_key: str

# -----------------------------
# Weather Agent
# -----------------------------
weather_agent = Agent(
    model="google-gla:gemini-1.5-flash",
    instructions=(
        "You are a focused weather assistant. Use get_lat_lng() and get_weather() "
        "tools to answer weather queries in short, accurate sentences."
    ),
    deps_type=Deps,
    retries=2,
)

class LatLng(BaseModel):
    lat: float
    lng: float

@weather_agent.tool
async def get_lat_lng(ctx: RunContext[Deps], location_description: str) -> LatLng:
    """Resolve location name to latitude and longitude."""
    try:
        r = await ctx.deps.client.get(
            "https://geocode.maps.co/search",
            params={"q": location_description},
            timeout=15.0,
        )
        r.raise_for_status()
        data = r.json()
        if not data:
            raise ValueError("No location found")
        first_result = data[0]
        return LatLng(lat=float(first_result["lat"]), lng=float(first_result["lon"]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Geocoding error: {e}")

@weather_agent.tool
async def get_weather(ctx: RunContext[Deps], lat: float, lng: float) -> dict[str, Any]:
    """Fetch realtime weather for given coordinates."""
    try:
        r = await ctx.deps.client.get(
            "https://api.tomorrow.io/v4/weather/realtime",
            params={"location": f"{lat},{lng}", "apikey": ctx.deps.weather_api_key},
            timeout=15.0,
        )
        r.raise_for_status()
        data = r.json()
        values = data.get("data", {}).get("values", {})
        temp = values.get("temperature")
        code = values.get("weatherCode")
        desc_map = {
            1000: "clear", 1100: "mostly clear", 1101: "partly cloudy", 1102: "mostly cloudy",
            2000: "fog", 2100: "light fog",
            3000: "light wind", 3001: "wind", 3002: "strong wind",
            4000: "drizzle", 4001: "rain", 4200: "light rain", 4201: "heavy rain",
            5000: "snow", 5001: "flurries", 5100: "light snow", 5101: "heavy snow",
            6000: "freezing drizzle", 6001: "freezing rain", 6200: "light freezing rain", 6201: "heavy freezing rain",
            7000: "ice pellets", 7101: "heavy ice pellets", 7102: "light ice pellets",
            8000: "thunderstorm"
        }
        description = desc_map.get(code, "clear sky")
        return {"temperature": f"{temp} °C" if temp is not None else "N/A", "description": description}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Weather API error: {e}")

# -----------------------------
# Chatbot Agent
# -----------------------------
chatbot_agent = Agent(
    model="google-gla:gemini-2.0-flash",  # choose a supported model
    instructions=(
        "You are a conversational assistant.\n"
        "1) If a user's query is ambiguous or general (e.g., 'Tell me a joke'), "
        "first ask clarifying questions to get context.\n"
        "   Example: 'What type of joke would you like? Short, multi-line, or situational?'\n"
        "2) Only provide a final response after getting enough information.\n"
        "3) If query is about weather, output 'WEATHER_QUERY' token.\n"
        "4) Avoid hallucinations; if unsure, ask for clarification.\n"
        "5) Give helpful, concise responses for other queries."
    ),
    retries=2,
)
