from typing import List

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from nomic import embed

app = FastAPI()


@app.post(
    "/api/generate",
    summary="Generar embeddings",
    description="Genera embeddings para una lista de textos.",
)
async def testing_embeddings(
    text_to_embed: List[str],
    model: str = "nomic-embed-text-v1",
):
    try:
        output = embed.text(
            texts=text_to_embed,
            model=model,
            task_type="search_document",
            inference_mode="local",
        )
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e), "message": "Error en los parámetros de entrada."},
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "message": "Error inesperado al generar embeddings.",
            },
        )
    return {"output": output}
