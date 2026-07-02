from langchain_openai import ChatOpenAI

from Customer_Service_Assistant.config.settings import settings

_kwargs = {
    "model": settings.llm_model,
    "base_url": settings.llm_base_url,
    "temperature": 0,
}
if settings.llm_api_key:
    _kwargs["api_key"] = settings.llm_api_key

llm = ChatOpenAI(**_kwargs)

if __name__ == "__main__":
    response = llm.invoke("你好")
    print(response.content)
