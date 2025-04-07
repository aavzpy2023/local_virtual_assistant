import json
from typing import List

# import sys
# import milvus.milvus
# sys.path.append("/app")
import requests
from fastapi import FastAPI  # type: ignore
from pymilvus import MilvusClient  # type: ignore

app = FastAPI()


# # app.include_route(milvus.milvus_router)
client = MilvusClient("Versat.db")


@app.get("/mv_insert")
async def insert(collection: str, data: dict):
    if not client.has_collection(collection_name=collection):
        client.create_collection(
            collection_name=collection,
            dimension=768,  # The vectors we will use in this demo has 768 dimensions
        )
        res = client.insert(collection_name=collection, data=data)
    return res


@app.post("/get_answer/")
# async def generate_formatted(request: GenerateRequest):
async def generate_formatted(model: str, prompt: str, stream: bool = False):
    """
    Get answer from ollama
    """
    url = "http://ollama:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    # data = {"model": request.model, "prompt": request.prompt, "stream": request.stream}
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


@app.post("/get_embeddings")
async def get_embeddings(text: str, overlap: int, answer_split_chr: str = "\n"):
    chunks = text.split(answer_split_chr)
    url = "http://nomic:8000/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {"text_to_embed": chunks}
    embeddings = requests.get(url, headers=headers, json=payload)
    return embeddings.json()


@app.get("/hola")
async def hola():
    return {"elia": "hola"}
