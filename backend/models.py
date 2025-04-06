from pydantic import BaseModel


# Define a data model using Pydantic for the request body
class GenerateRequest(BaseModel):
    model: str  # Name of the model to be used
    prompt: str  # Prompt to be sent to the model
    stream: bool = False  # Flag to enable streaming of responses
