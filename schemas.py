from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    query: str = Field(..., example="Tell me a joke.")

class ChatResponse(BaseModel):
    response: str = Field(..., example="Sure — what type of joke would you like? (1) sci-fi (2) general (3) doctor")

class WeatherRequest(BaseModel):
    query: str = Field(..., example="What's the weather in Hyderabad?")

class WeatherResponse(BaseModel):
    response: str = Field(..., example="Hyderabad: 32 °C, clear.")
