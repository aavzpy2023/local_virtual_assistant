import datetime
import json
import logging
import re
from typing import Optional

import requests
import streamlit as st
from dotenv import load_dotenv
from pymilvus import MilvusClient

load_dotenv()


def print_with_date(message: str):
    print(datetime.datetime.now(), "-->", message)


def get_answer_from_model(
    prompt: str,
    model: str = "qwen2.5:3B",
    endpoint: str = "http://backend:5000/get_answer/",
):
    """
    Get answer from model.

    Args:
        prompt (str): The prompt to be sent to the model.
        model (str): The model to be used.
        endpoint (str): The endpoint to be used.

    Returns:
        str: The answer from the model.
    """
    # print("The len of prompt is:", prompt)

    # Validate inputs
    if not prompt:
        return "Error: the prompt must be a non-empty string."
    if not model:
        return "Error: The model parameter must be a non-empty string."

    # Build payload
    payload = {"model": model, "prompt": prompt, "stream": False, "port": 11434}

    # Validate JSON
    try:
        json_payload = json.dumps(payload)
        json.loads(json_payload)
    except ValueError as e:
        return f"Error: The JSON generated is not valid: {e}"

    # Send the request
    try:
        response = requests.post(endpoint, json=payload, timeout=500)
        response.raise_for_status()

        print("The answer is:", response.text)

        # Process response
        data = response.json()
        if "response" in data:
            return data["response"]
        else:
            logging.error(f"Wrong response format from server: {data}")
            return "Error: The answer format from server is not valid."
    except requests.exceptions.RequestException as e:
        logging.error(f"Connection error: {e}")
        return f"Connection error: {e}"


def get_question_contents(questions_id: list[str]) -> str:
    """
    Retrieve the contents of questions from a file.

    Args:
        questions_id (list[str]): A list of question IDs.

    Returns:
        str: The contents of the questions.
    """
    file_path = "./documents/mf3.txt"
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        # Regular expression to capture IDs and their content
        pattern = r"(ID:\s*[A-Za-z0-9_-]+_[\w-]+_P\d+)\s*([\s\S]*?)(?=ID:\s*[A-Za-z0-9_-]+_[\w-]+_P\d+|\Z)"
        matches = re.findall(pattern, content)

        # Create a dictionary with IDs as keys and content as values
        result: dict[str, str] = {}
        for match in matches:
            question_id = match[0].strip()
            question_content = match[1].strip()
            result[question_id] = question_content

        # Adjust IDs in questions_id to match dictionary keys
        adjusted_questions_id: list[str] = [f"ID: {c_id}" for c_id in questions_id]

        # Obtain contents corresponding to adjusted IDs
        contents = [
            result.get(c_id, f"ID not found: {c_id}") for c_id in adjusted_questions_id
        ]
        return contents  # type: ignore

    except Exception as e:
        st.error(f"Error reading file or extracting data: {e}")
        return []  # type: ignore


@st.cache_resource
def get_milvus_client(db_name: str, collection_name: str) -> Optional[MilvusClient]:
    """
    Initialize and return a Milvus client.

    Args:
        db_name (str): The name of the database.
        collection_name (str): The name of the collection.

    Returns:
        Optional[MilvusClient]: The initialized Milvus client or None if an error occurs.
    """
    milvus_url = "http://milvus-standalone:19530"
    try:
        print_with_date(f"Intentando conectar a Milvus en {milvus_url}...")
        client = MilvusClient(uri=milvus_url)
        print_with_date(
            f"Conectado a Milvus. Intentando usar base de datos '{db_name}'..."
        )
        client.use_database(db_name)  # type: ignore
        print_with_date(f"Base de datos '{db_name}' seleccionada.")

        print_with_date(
            f"Verificando existencia de la colección '{collection_name}'..."
        )
        if not client.has_collection(collection_name):  # type: ignore
            st.error(
                f"Error crítico: La colección '{collection_name}' no existe en la base de datos '{db_name}'."
            )
            logging.error(
                f"La colección '{collection_name}' no existe en la DB '{db_name}'."
            )
            return None

        print_with_date(
            f"Colección encontrada. Intentando cargar '{collection_name}' en memoria..."
        )
        client.load_collection(collection_name)  # type: ignore
        print_with_date(f"Colección '{collection_name}' cargada exitosamente.")
        return client
    except Exception as e:
        st.error(f"No se pudo conectar o preparar la colección en Milvus: {e}")
        logging.error(f"Error conectando/cargando colección Milvus: {e}", exc_info=True)
        return None


def get_embedding_ollama(text: str):
    """
    Sends a POST request to the Ollama API to convert text into an embedding.

    text: list of texts: ["your text"]
    Returns:
        list: The embedding vector as a list of floats.
    """
    url = "http://backend:5000/generate-embeddings/"
    payload = {"texts": [text]}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        # Se asume que la respuesta tiene el formato:
        # {"embeddings": [[0.123,...,0.456]]}
        return data["embeddings"][0]
    except Exception as e:
        st.error(f"Error al obtener el embedding desde la API: {e}")
        return None
