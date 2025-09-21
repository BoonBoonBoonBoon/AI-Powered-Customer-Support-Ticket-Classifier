from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # Semantic version of the model artifacts produced by training
    MODEL_VERSION: str = "1.0.0"
    # Base directory for saved model versions
    MODELS_BASE_DIR: str = "models"
    # Validation split ratio for training
    VALIDATION_SPLIT: float = 0.2
    # Random seed for reproducibility
    RANDOM_SEED: int = 42
    # Whether to compute evaluation metrics during training
    ENABLE_EVAL: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()