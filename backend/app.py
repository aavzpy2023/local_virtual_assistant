import json

import requests
from fastapi import FastAPI
from pydantic import BaseModel


# Define a data model using Pydantic for the request body
class GenerateRequest(BaseModel):
    model: str  # Name of the model to be used
    prompt: str  # Prompt to be sent to the model
    stream: bool = False  # Flag to enable streaming of responses


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "to the API"}


@app.post("/get_answer/")
async def generate_formatted(model: str, prompt: str, stream: bool = False):
    url = "http://ollama:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {"model": model, "prompt": prompt, "stream": stream}

    response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)

    # Capture and format streamed responses
    formatted_response = ""
    for line in response.iter_lines():
        if line:
            try:
                json_line = json.loads(line.decode("utf-8"))  # Parse each line as JSON
                if "response" in json_line:
                    formatted_response += json_line[
                        "response"
                    ]  # Extract the response content
            except json.JSONDecodeError:
                # Handle any decoding errors
                continue

    # Ensure the response ends with a newline character
    formatted_response = formatted_response.strip() + "\n"

    # Return the formatted response as a dictionary
    return {"response": formatted_response}
