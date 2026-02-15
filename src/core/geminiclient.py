# import os
# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.messages import HumanMessage

# # Load .env
# load_dotenv()

# # Create model (API key automatically read from GOOGLE_API_KEY)
# llm = ChatGoogleGenerativeAI(
#     model="gemini-3-flash-preview",   # gemini-3-flash-preview may change; use stable if needed
#     temperature=1
# )

# # Send prompt
# response = llm.invoke([
#     HumanMessage(content="Explain how AI works in detail")
# ])

# print(response.text)

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# load .env from this folder
load_dotenv()

class GeminiClient:
    def __init__(self, model_name: str, temperature: float, max_tokens: int):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    def generate(self, prompt: str) -> str:
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
