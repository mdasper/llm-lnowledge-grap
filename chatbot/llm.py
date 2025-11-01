
import os
from dotenv import load_dotenv
load_dotenv()

# tag::llm[]
from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(
    model_name="gemini-2.5-flash-lite",  # or "gemini-pro" if supported
    temperature=0,
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location=os.getenv("GOOGLE_CLOUD_LOCATION")
)
 
# Initialize Vertex AI Embeddings

 
# end::llm[]

# tag::embedding[]
from langchain_google_vertexai import VertexAIEmbeddings

embedding_provider = VertexAIEmbeddings(
    model_name="gemini-embedding-001",
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location=os.getenv("GOOGLE_CLOUD_LOCATION")
  
)
