
from dotenv import load_dotenv
load_dotenv()

from langchain_core.language_models import BaseLanguageModel
from langchain_core.outputs import LLMResult
from typing import List, Optional, Any
from google import genai
import os 
from langchain_google_vertexai import VertexAIEmbeddings

class VertexAIWrapper(BaseLanguageModel):
    def __init__(self, model_name: str = "gemini-2.5-flash-lite", api_key: Optional[str] = None):
        self.client = genai.Client(
            vertexai=True,
            api_key=api_key
        )
        self.model = self.client.get_model(model_name)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.model.generate_content([prompt])
        return response.text

    def invoke(self, input: str, **kwargs: Any) -> str:
        return self._call(input)

    @property
    def _llm_type(self) -> str:
        return "vertexai-wrapper"
    
llm = VertexAIWrapper(api_key=os.getenv("GOOGLE_API_KEY"))

embedding_provider = VertexAIEmbeddings(
    model_name="text-embedding-004",
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),  # Replace with your GCP project ID
    location=os.getenv("GOOGLE_CLOUD_LOCATION")        # Replace with your Vertex AI region
     # Optional if using default credentials
)

