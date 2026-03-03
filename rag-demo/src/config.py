# Configuration settings for the application

class Config:
    API_KEY = "your_api_key_here"
    EMBEDDING_MODEL = "your_embedding_model_here"
    RETRIEVAL_MODEL = "your_retrieval_model_here"
    GENERATOR_MODEL = "your_generator_model_here"
    DATABASE_URL = "sqlite:///./test.db"  # Example for SQLite, change as needed
    DEBUG = True  # Set to False in production

config = Config()