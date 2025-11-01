import os
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_google_vertexai import ChatVertexAI
from dotenv import load_dotenv
load_dotenv()

llm = ChatVertexAI(
    model_name="gemini-2.5-flash-lite",  # or "gemini-pro" if supported
    temperature=0.2,
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location=os.getenv("GOOGLE_CLOUD_LOCATION")
)

doc_transformer = LLMGraphTransformer(
    llm=llm,
)

for chunk in chunks:
    graph_docs = doc_transformer.convert_to_graph_documents([chunk])