from pydantic import BaseModel, Field





class RunSchema(BaseModel):
    id: str = Field(..., description="ID if the requirement from Bogota te escucha")