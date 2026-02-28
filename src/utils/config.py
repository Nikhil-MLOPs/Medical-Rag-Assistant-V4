import yaml # To read YAML configuration files
from pydantic import BaseModel, Field # To define data models with validation


# Phase-1: Ingestion Configuration
class IngestionConfig(BaseModel): # Pydantic is going to check if all the fields below and their types are correct.
    raw_dir: str = Field(description = "Directory where raw data is stored")
    processed_dir: str = Field(description = "Directory where processed data will be stored")
    skip_start_pages: int = Field(description = "Number of pages to skip from start of the document")
    skip_after_pages: int = Field(description = "Number of pages to skip after processing a certain number of pages")


# Helper function
def load_yaml(path: str) -> dict: # This function reads a YAML file and returns its contents as a dictionary.
    with open(path, "r") as f: # Open the file in read mode.
        return yaml.safe_load(f) # yaml.safe_load parses the yaml file and returns dictionary.

"""
The input that it takes is - path of yaml file, for example - "configs/ingestion.yaml"

The output returned is - 
dictionary which will look like this: -
{
    "raw_dir": "data/raw",
    "processed_dir": "data/processed/pages",
    "skip_start_pages": 30,
    "skip_after_pages": 4065
}
"""
def load_ingestion_config(path: str) -> IngestionConfig:
    return IngestionConfig(**load_yaml(path))

"""
Step-1:
**load_yaml(path) becomes exactly this
IngestionConfig(
    raw_dir='data/raw',
    processed_dir='data/processed/pages',
    skip_start_pages=30,
    skip_after_pages=4065
)

step-2:
IngestionConfig(...), Pydantic (the BaseModel) does these things automatically -
- Receives the four keyword arguments
- Validates every field
- Converts types if needed
- Creates a real object of type IngestionConfig
- If anything is wrong (missing field, wrong type, negative pages, etc.), it raises a clear ValidationError immediately
"""

# Phase-2: Cleaning Configuration

class CleaningConfig(BaseModel):
    chunk_size: int = Field(description="Size of each chunk in characters")
    chunk_overlap: int = Field(description="Number of characters to overlap in consecutive chunks")


def load_cleaning_config(path: str) -> CleaningConfig:
    return CleaningConfig(**load_yaml(path))