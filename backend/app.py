import json
import os
from typing import Dict, List, Any

# import sys
# import milvus.milvus
# sys.path.append("/app")
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException  # type: ignore
from models import Answer_Request, Data_embed

# from milvus.milvus import milvus_router
from pymilvus import MilvusClient  # type: ignore

load_dotenv()

app = FastAPI(
    title="Mi API",
    version="1.0.0",
    openapi_version="3.0.0",
    description="Esta es una API de prueba",
    root_path="/api"
)

# Crear sesión con pooling
ollama_session = requests.Session()

# Configurar adaptador
adapter: HTTPAdapter = HTTPAdapter(
    pool_connections=20,
    pool_maxsize=20,
    max_retries=Retry(total=3, backoff_factor=1)
)

# Montar adaptador
ollama_session.mount("http://", adapter)
ollama_session.mount("https://", adapter)


# # app.include_route(milvus.milvus_router)
client = MilvusClient("Versat.db")


@app.get("/")
async def read_api_root():
    return {"message": "Bienvenido a Mi API. Accede a /docs para la documentación."}

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
async def get_answer_from_ollama(request_data: Answer_Request):
    """
    Obtiene una respuesta de un modelo LLM a través de Ollama en modo no-streaming.
    """
    # OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
    OLLAMA_BASE_URL = "http://ollama_llm:11434"
    ollama_generate_url = f"{OLLAMA_BASE_URL}/api/generate"

    print(ollama_generate_url)

    headers = {"Content-Type": "application/json"}

    # Payload para la API de Ollama.
    # Forzamos stream: False aquí porque este endpoint está diseñado para no-streaming.
    api_payload = {
        "model": request_data.model,
        "prompt": request_data.prompt,
        "stream": False,  # Solicitando explícitamente a Ollama que NO haga stream
    }

    print(
        f"Enviando solicitud a Ollama: URL={ollama_generate_url}, Payload={api_payload}"
    )

    try:
        # Realizar la solicitud POST usando la sesión configurada.
        # stream=False aquí le indica a la librería 'requests' que lea
        # toda la respuesta del servidor de una vez.
        response: requests.Response = ollama_session.post(
            ollama_generate_url,
            headers=headers,
            data=json.dumps(api_payload),
            stream=False,  # Redundante si el default de la sesión no se cambia, pero explícito.
            timeout=180,  # Timeout de 3 minutos, ajustable según la lentitud esperada del LLM.
        )

        # Verificar si Ollama devolvió un error HTTP (e.g., 4xx, 5xx).
        # Esto lanzará una excepción si el código de estado indica un error.
        response.raise_for_status()

        # Si la solicitud fue exitosa (código 200-299) y stream=False,
        # la respuesta de Ollama es un único objeto JSON.
        json_response: Dict[str, Any] = response.json()

        print(f"Respuesta JSON de Ollama: {json_response}")

        # Extraer el texto de la respuesta del campo "response" que Ollama usa.
        generated_text = json_response.get("response", "")

        # Opcional: Limpiar el texto (quitar espacios en blanco al inicio/final).
        # El añadir "\n" al final depende de si el cliente lo espera o no.
        # Generalmente, es mejor que el cliente maneje el formato final.
        formatted_response = generated_text.strip()

        return {"response": formatted_response}

    except requests.exceptions.Timeout as e:
        request = getattr(e.response, 'request', None) if hasattr(e, 'response') else None
        timeout = getattr(request, 'timeout', None) if request else None

        error_message = f"Timeout (después de {timeout or 'N/A'} segundos) al comunicarse con Ollama."
        print(error_message)  # Loguear en el backend
        raise HTTPException(status_code=504, detail=error_message)

    except requests.exceptions.HTTPError as http_err:
        # Captura errores específicos de HTTPError que response.raise_for_status() puede lanzar.
        # El cuerpo de la respuesta de error de Ollama podría estar en response.text o response.json()

        error_detail = f"Error HTTP de Ollama: {http_err}. "
        try:
            ollama_error = response.json().get("error", response.text)
            error_detail += f"Detalle de Ollama: {ollama_error}"
        except ValueError:  # Si la respuesta de error no es JSON
            error_detail += f"Respuesta de Ollama (no JSON): {response.text[:200]}"  # Mostrar solo parte

        print(error_detail)  # Loguear en el backend
        raise HTTPException(
            status_code=response.status_code if response else 503, detail=error_detail
        )

    except requests.exceptions.RequestException as req_err:
        # Otros errores de la librería requests (e.g., problemas de conexión DNS, red).
        error_message = (
            f"Error de la librería Requests al comunicarse con Ollama: {req_err}"
        )
        print(error_message)  # Loguear en el backend
        raise HTTPException(status_code=503, detail=error_message)

    except json.JSONDecodeError as json_err:
        # Ocurre si Ollama devuelve un status 2xx pero el cuerpo no es JSON válido.
        error_message = (
            f"Error al decodificar la respuesta JSON de Ollama. "
            f"Respuesta recibida (primeros 200 chars): '{response.text[:200] if response else 'No response'}'. Error: {json_err}"
        )
        print(error_message)  # Loguear en el backend
        raise HTTPException(status_code=500, detail=error_message)

    except Exception as e:
        # Captura general para cualquier otro error inesperado.
        error_message = f"Error inesperado en el endpoint /get_answer/: {e}"
        print(error_message)  # Loguear en el backend
        # Es buena idea loguear el traceback completo aquí para depuración.
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_message)


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
