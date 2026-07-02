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


if __name__ == "__main__":
    # Quick smoke tests — run with: python Customer_Service_Assistant/config/settings.py

    # 1. Singleton is a Settings instance
    assert isinstance(settings, Settings), "settings should be a Settings instance"

    # 2. Default values
    s = Settings(_env_file="/nonexistent")  # rely only on field defaults
    assert s.llm_model == "qwen-plus"
    assert s.llm_base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert s.llm_api_key == ""
    assert s.database_url == ""
    assert s.commerce_api_base_url == "http://127.0.0.1:18000"
    assert s.app_host == "127.0.0.1"
    assert s.app_port == 18000
    assert isinstance(s.app_port, int)

    # 3. Env-var overrides (set before constructing)
    import os

    os.environ["LLM_MODEL"] = "qwen-max"
    os.environ["APP_PORT"] = "9000"
    s2 = Settings(_env_file="/nonexistent")
    assert s2.llm_model == "qwen-max"
    assert s2.app_port == 9000

    print("All settings tests passed.")
