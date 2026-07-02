from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }

    # LLM
    llm_model: str = "qwen-plus"
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_api_key: str = ""

    # Database
    database_url: str = ""

    # Commerce API
    commerce_api_base_url: str = "http://127.0.0.1:18000"

    # App
    app_host: str = "127.0.0.1"
    app_port: int = 18000


settings = Settings()
