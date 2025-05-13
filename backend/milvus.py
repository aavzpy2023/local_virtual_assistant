import json

import re


import requests
from pymilvus import DataType, MilvusClient


def connect_to_milvus_db(db_name: str):
    """
    Connect to Milvus database
    """
    client = MilvusClient(uri="http://localhost:19530")
    client.using_database(db_name=db_name)   # type: ignore
    return client


def search_vector(
    client: MilvusClient, collection_name: str, vector: list[dict[str, float]]
) -> str:
    """
    Perform a similarity search on an embedding vector within a specified collection in an AI platform.

    Parameters:
        client (MilvusClient): The Milvus Client object that handles communication with the AI platform.
        collection_name (str): The name of the collection where the vectors are stored.
        vector: The embedding vector to search for similarity against other embeddings in the specified collection. Must be a list.

    Returns:
        List[Dict]: A nested list of dictionaries containing the search results. Each dictionary contains an 'id' and 'distance'.
    """
    result = client.search(   # type: ignore
        collection_name=collection_name,
        anns_field="q_vector",
        data=[vector],
        limit=2,
        search_params={
            "metric_type": "COSINE",
            "params": {"nprobe": 32},
        },
    )
    return result # type: ignore


def split_text_into_chunks(text:str, max_length: int=512, overlap:int=100):
    """
    Split a text into chunks with the maximum specified length,
    ensuring that each chunk does not cut off ideas or incomplete sentences.

    Args:
        text (str): The text to be split.
        max_length (int, optional): Maximum length of each chunk. Default is 512.
        overlap (int, optional): Number of characters that overlap between consecutive chunks. Default is 100.

    Returns:
        list: A list of text fragments.
    """
    sections = re.split(r"\n{2,}", text)
    chunks: list[str] = []
    for section in sections:
        if len(section) > max_length:
            sentences = re.split(r"(?<=[.!?])\s+", section)
            current_chunk = ""
            for sentence in sentences:
                if (
                    len(current_chunk) + len(sentence) + 1 > max_length
                    and current_chunk
                ):
                    chunks.append(current_chunk.strip())

                    current_chunk = current_chunk[-overlap:] if overlap > 0 else ""
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            if current_chunk:
                chunks.append(current_chunk.strip())
        else:
            if chunks and len(chunks[-1]) + len(section) + 2 <= max_length:
                chunks[-1] += "\n\n" + section
            else:
                chunks.append(section.strip())

    final_chunks: list[str] = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            final_chunks.append(chunk)
        else:
            overlapped_chunk = chunks[i - 1][-overlap:] + chunk
            final_chunks.append(overlapped_chunk[:max_length])
    return final_chunks


def process_questions_file(
    file_path: str, max_length: int = 512, overlap: int = 100
) -> dict[str, list[str]]:
    """
    Process a file containing questions by splitting the text into chunks with the specified maximum length and handling overlaps.

    Args:
        file_path (str): The path to the file containing questions.
        max_length (int, optional): Maximum length of each chunk. Default is 512.
        overlap (int, optional): Number of characters that overlap between consecutive chunks. Default is 100.

    Returns:
        dict: A dictionary where keys are question IDs and values are lists of text chunks for each question.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    questions = re.split(r"ID:", content)[1:]
    processed_questions: dict[str, list[str]] = {}
    for question in questions:
        full_question = f"ID:{question}".strip()
        id_match = re.search(r"ID:\s*(\S+)", full_question)
        if not id_match:
            print(
                f"Error: No se pudo extraer el ID de la pregunta: {full_question[:100]}..."
            )
            continue
        question_id: str = id_match.group(1)
        cleaned_text = re.sub(r"Sct\.", "", full_question).strip()
        cleaned_text = re.sub(r"(?i)\bN/A\b", "", cleaned_text).strip()
        chunks: list[str] = split_text_into_chunks(cleaned_text, max_length, overlap)
        if chunks:
            processed_questions[question_id] = chunks
        else:
            print(
                f"Advertencia: La pregunta con ID {question_id} no tiene contenido válido."
            )
    return processed_questions


# URL de la API local
API_URL = "http://localhost:5000/generate-embeddings/"


def get_embeddings(texts: list[str]):
    """
    Get embeddings for a list of texts by sending a POST request to an API.

    :param texts: A list of text strings.

    :return: A list of lists, where each inner list represents the embeddings for a given text.
    """
    headers = {"Content-Type": "application/json"}
    data = {"texts": texts}
    response = requests.post(API_URL, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()["embeddings"]
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")


def create_database(client: MilvusClient, db_name: str):
    """
    Allows creating or verifying if it already exists. If it does not exist,
    it will be created with three replicas.

    :param client: The Milvus client object used to interact with the Milvus server.
    :param db_name: A unique name for the database that will be created or verified.

    :return: The updated Milvus client object after using the specified database.
    """
    print(f"Creating database {db_name}")
    if db_name not in client.list_databases():  # type: ignore
        client.create_database(  # type: ignore
            db_name=db_name, properties={"database.replica.number": 3}
        )
    else:
        print("Database already exists")
    client.using_database(db_name)  # type: ignore
    return client


def create_schema(client: MilvusClient, collection_name: str):
    """
    Create a new collection with a specific schema and its fields.

    :param client: The Milvus client object to interact with the Milvus server.
    :param collection_name: A unique name for the collection that will be created or verified.

    :return: The updated Milvus client object after using the specified collection and schema.
    """
    print("Creating schema")
    # Remove collection if exists
    if client.has_collection(collection_name):  # type: ignore
        client.drop_collection(collection_name)  # type: ignore

    # Create schema
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)  # type: ignore
    schema.add_field("q_id", DataType.VARCHAR, is_primary=True, max_length=64)  # type: ignore
    schema.add_field("q_vector", DataType.FLOAT_VECTOR, dim=768)  # type: ignore
    schema.add_field("q_chunk", DataType.VARCHAR, max_length=512)  # type: ignore

    # Create collections
    client.create_collection(collection_name=collection_name, schema=schema)  # type: ignore
    return client


def create_index(client: MilvusClient, index_name: str, collection_name: str):
    """
    Create a new database (index) for a specific set of data.

    :param client (obj): The Milvus client object used to interact with the Milvus server.
    :param index_name: The name of the index that will be created. This is the field on which elements are searched.
    :param collection_name: The name of the collection (database) for which you want to create the index.

    :return: Will return the updated Milvus client object after using the specified collection and index.
    """
    print("Creating index")
    index_params = [
        {
            "field_name": index_name,
            "index_type": "IVF_SQ8",
            "metric_type": "COSINE",
            "params": {"nlist": 256},
        }
    ]
    try:
        client.create_index(collection_name=collection_name, index_params=index_params) # type: ignore
        print("Índice creado exitosamente.")
    except Exception as ex:
        print("Error al crear el índice:", ex)
    return client


if __name__ == "__main__":
    file_path = "./documents/mf3.txt"
    json_data = process_questions_file(file_path, max_length=450)

    # Crear una lista plana con todos los chunks
    ps: list[str] = []
    ids: list[str] = []

    for question_id, chunks in json_data.items():
        c_id: list[str] = []
        for i, chunk in enumerate(chunks):
            ps.append(chunk)  # Añadir el chunk a la lista plana
            if question_id in c_id:
                ids.append(f"{question_id}_{i + 1}")
            else:
                ids.append(question_id)  # Añadir el ID correspondiente
                c_id.append(question_id)

    # Verificar que ningún chunk exceda los 512 caracteres
    invalid_chunks = [(i, len(st)) for i, st in enumerate(ps) if len(st) > 512]
    if invalid_chunks:
        print("Chunks inválidos encontrados:")
        for idx, length in invalid_chunks:
            print(f"Chunk {idx}: Longitud = {length}")
    else:
        print("Todos los chunks tienen una longitud válida.")

    # Generar embeddings
    emb = get_embeddings(ps)

    # Inicializar cliente de Milvus
    client = MilvusClient(uri="http://localhost:19530")

    # Crear y activar la base de datos "versat"
    client = create_database(client, db_name="versat")

    # Crear colección "sarasola" en la base de datos "versat"
    client = create_schema(client, collection_name="sarasola")

    # Crear índice en "sarasola"
    client = create_index(client, index_name="q_vector", collection_name="sarasola")

    # Preparar datos
    dt_ok: list[dict[str, float|str]] = []
    for q_id, chunk, vector in zip(ids, ps, emb):
        if chunk.strip():  # Asegurar que el chunk no esté vacío
            if len(vector) == 768:  # Validar la longitud del vector
                dt_ok.append({"q_id": q_id, "q_vector": vector, "q_chunk": chunk})
            else:
                print(f"Vector inválido para ID {q_id}: Longitud = {len(vector)}")

    # Insertar datos
    print("Inserting data")
    try:
        res = client.insert(collection_name="sarasola", data=dt_ok)  # type: ignore
        print("Inserción exitosa:")
    except Exception as e:
        print("Error al insertar datos:", e)
