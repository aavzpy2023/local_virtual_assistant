from pydantic import BaseModel


# Define a data model using Pydantic for the request body
class Answer_Request(BaseModel):
    model: str = "qwen2.5:3b"  # Name of the model to be used
    prompt: str  # Prompt to be sent to the model
    stream: bool = False  # Flag to enable streaming of responses


class Data_embed(BaseModel):
    texts: list[str]
