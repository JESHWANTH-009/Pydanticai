from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from httpx import AsyncClient
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys from environment
gemini_key = os.getenv("GEMINI_API_KEY")
weather_api_key = os.getenv("WEATHER_API_KEY")
geo_api_key = os.getenv("GEO_API_KEY")
if gemini_key:
    os.environ["GOOGLE_API_KEY"] = gemini_key
# Warn if keys missing
if not all([gemini_key, weather_api_key, geo_api_key]):
    print("Warning: One or more API keys are missing in .env. API calls will fail.")

# Dependencies for tools
@dataclass
class Deps:
    client: AsyncClient
    geo_api_key: str
    weather_api_key: str

# Weather Agent with clear instructions
weather_agent = Agent(
    model='google-gla:gemini-1.5-flash',
    instructions=(
        "You are a helpful weather assistant. "
        "When the user asks about the weather, ALWAYS: "
        "1. Use 'get_lat_lng' to get latitude/longitude for each location. "
        "2. Use 'get_weather' to get real-time weather for each location. "
        "3. Respond in one short sentence summarizing the results."
    ),
    deps_type=Deps,
    retries=2,
)

# Lat/Lng Model
class LatLng(BaseModel):
    lat: float
    lng: float

# Tool: Get coordinates from location name
@weather_agent.tool
async def get_lat_lng(ctx: RunContext[Deps], location_description: str) -> LatLng:
    """Get the latitude and longitude of a location."""
    try:
        r = await ctx.deps.client.get(
            'https://geocode.maps.co/search',
            params={'q': location_description, 'api_key': ctx.deps.geo_api_key},
        )
        r.raise_for_status()
        data = r.json()
        if not data:
            raise ValueError("No location found for the given description.")
        first_result = data[0]
        return LatLng(lat=float(first_result['lat']), lng=float(first_result['lon']))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Geocoding API error: {str(e)}")

# Tool: Get weather for coordinates
@weather_agent.tool
async def get_weather(ctx: RunContext[Deps], lat: float, lng: float) -> dict[str, Any]:
    """Get the weather at a location."""
    try:
        r = await ctx.deps.client.get(
            'https://api.tomorrow.io/v4/weather/realtime',
            params={'location': f'{lat},{lng}', 'apikey': ctx.deps.weather_api_key},
        )
        r.raise_for_status()
        data = r.json()
        temp = data['data']['values']['temperature']
        description_code = data['data']['values']['weatherCode']

        weather_descriptions = {
            1000: 'clear', 1100: 'mostly clear', 1101: 'partly cloudy',
            1102: 'mostly cloudy', 2000: 'fog', 2100: 'light fog',
            4000: 'drizzle', 4001: 'rain', 4200: 'light rain',
            5000: 'snow', 5001: 'flurries', 5100: 'light snow',
        }
        description = weather_descriptions.get(description_code, 'unspecified')

        return {"temperature": f'{temp} °C', "description": description}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Weather API error: {str(e)}")

# FastAPI app
app = FastAPI(
    title="Weather Agent API",
    description="An API that uses a Pydantic AI agent to get real-time weather information.",
)

# Request & Response models
class WeatherRequest(BaseModel):
    query: str = Field(..., example="What's the weather like in New York and Tokyo?")

class WeatherResponse(BaseModel):
    response: str = Field(..., example="The weather in New York is 22 °C and clear. The weather in Tokyo is 18 °C and cloudy.")

# Endpoint
@app.post("/weather", response_model=WeatherResponse)
async def get_weather_info(request: WeatherRequest):
    async with AsyncClient() as client:
        deps = Deps(client=client, geo_api_key=geo_api_key, weather_api_key=weather_api_key)
        try:
            result = await weather_agent.run(request.query, deps=deps)
            return WeatherResponse(response=result.output)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")
