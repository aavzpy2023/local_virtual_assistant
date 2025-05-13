import json
import os
from typing import Dict, List

# import sys
# import milvus.milvus
# sys.path.append("/app")
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException  # type: ignore
from models import Answer_Request, Data_embed

# from milvus.milvus import milvus_router
from pymilvus import MilvusClient  # type: ignore

load_dotenv()

app = FastAPI()


# # app.include_route(milvus.milvus_router)
client = MilvusClient("Versat.db")


@app.get("/mv_insert")
async def insert(
    collection: str,
    data: List[dict] = [
        {"vector": [], "text": "", "subject": ""},
        {"vector": [], "text": "", "subject": ""},
    ],
):
    if not client.has_collection(collection_name=collection):
        client.create_collection(
            collection_name=collection,
            dimension=768,  # The vectors we will use in this demo has 768 dimensions
        )

        res = client.insert(collection_name=collection, data=data)
    return res


@app.post("/get_answer/")
# async def generate_formatted(request: GenerateRequest):
async def generate_formatted(data: Answer_Request):
    """
    Get answer from ollama
    """
    url = f"http://ollama_llm:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    # data = {"model": request.model, "prompt": request.prompt, "stream": request.stream}

    model = data.model
    prompt = data.prompt
    stream = data.stream
    data = {"model": model, "prompt": prompt}

    response = requests.post(url, headers=headers, data=json.dumps(data), stream=stream)

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
    url = "http://ollama_llm:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {"text_to_embed": chunks}
    embeddings = requests.get(url, headers=headers, json=payload)
    return embeddings.json()


@app.post("/generate-embeddings/")
async def generate_embeddings(data: Data_embed):
    """
    Genera embeddings para una lista de textos.
    """
    url_embed = os.getenv("URL_FOR_EMBED", "ollama_llm")
    port_embed = os.getenv("PORT_FOR_EMBED", 11434)

    OLLAMA_URL = f"http://{url_embed}:{port_embed}/api/embed"
    print("OLLAMA_URL", OLLAMA_URL)
    try:
        # Extraer la lista de textos del cuerpo de la solicitud
        input_texts = data.texts
        if not isinstance(input_texts, list) or not all(
            isinstance(text, str) for text in input_texts
        ):
            raise HTTPException(
                status_code=400,
                detail="El campo 'texts' debe ser una lista de cadenas.",
            )

        # Generar embeddings para cada texto
        embeddings = []
        for text in input_texts:
            payload = {"model": "nomic-embed-text:latest", "input": text}

            # Enviar la solicitud a Ollama
            response = requests.post(OLLAMA_URL, json=payload)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code, detail=response.text
                )

            # Agregar el embedding generado a la lista
            embeddings.append(response.json()["embeddings"])

        # Devolver los embeddings generados
        return {"embeddings": embeddings}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/generate-predefined-embeddings")
async def generate_predefined_embeddings():
    """
    Genera embeddings para una lista predefinida de textos.
    """
    try:
        # Lista predefinida de textos
        predefined_texts = ["asdasd ", "asd asd asdasdf "]

        # URL del endpoint /generate-embeddings/
        url = "http://backend:5000/generate-embeddings/"  # Usar el nombre del servicio Docker

        # Payload con la lista de textos
        payload = {"texts": predefined_texts}

        # Encabezados para indicar que el contenido es JSON
        headers = {"Content-Type": "application/json"}

        # Enviar la solicitud POST al endpoint /generate-embeddings/ con un timeout
        response = requests.post(url, headers=headers, json=payload, timeout=10)

        # Verificar si la solicitud fue exitosa
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        # Procesar la respuesta
        embeddings = response.json().get("embeddings", [])

        # Devolver los embeddings generados
        return {"embeddings": embeddings}

    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504, detail="Tiempo de espera excedido al generar embeddings."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
