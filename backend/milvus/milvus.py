from fastapi import APIRouter  # type: ignore

# from pydantic import BaseModel
from pymilvus import MilvusClient  # type: ignore

client = MilvusClient("Versat.db")

milvus_router = APIRouter(prefix="/milvus", tags="milvus")


@milvus_router.post("/insert")
def insert(data: str):
    if not client.has_collection(collection_name="Sarasola"):
        client.create_collection(collection_name="Sarasola")
    return {"data": data}
