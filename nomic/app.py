from fastapi import FastAPI

from nomic import embed

app = FastAPI()


@app.post("/embed")
async def generate_embeddings(text_to_embed: str):

    output = embed.text(
        texts=["Nomic Embedding API", "#keepAIOpen"],
        model="nomic-embed-text-v1",
        task_type="search_document",
        inference_mode="local",
    )
    return {"output": output}
