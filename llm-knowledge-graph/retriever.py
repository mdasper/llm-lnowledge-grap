import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")

# Initialize Vertex AI LLM
llm = ChatVertexAI(
    model_name="gemini-2.5-flash-lite",  
    temperature=0,
    project=GOOGLE_CLOUD_PROJECT,
    location=GOOGLE_CLOUD_LOCATION
)

# Initialize Vertex AI Embeddings
embedding_provider = VertexAIEmbeddings(
    model_name="gemini-embedding-001",
    project=GOOGLE_CLOUD_PROJECT,
    location=GOOGLE_CLOUD_LOCATION

)

# Connect to Neo4j
graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD')
)

# Use existing vector index
chunk_vector = Neo4jVector.from_existing_index(
    embedding_provider,
    graph=graph,
    index_name="chunkVector",
    embedding_node_property="textEmbedding",
    text_node_property="text",
    retrieval_query="""
    MATCH (node)-[:PART_OF]->(d:Document)
    WITH node, score, d
    MATCH (node)-[:HAS_ENTITY]->(e)
    MATCH p = (e)-[r]-(e2)
    WHERE (node)-[:HAS_ENTITY]->(e2)
    UNWIND relationships(p) as rels
    WITH 
        node, 
        score, 
        d, 
        collect(apoc.text.join(
            [labels(startNode(rels))[0], startNode(rels).id, type(rels), labels(endNode(rels))[0], endNode(rels).id]
        ," ")) as kg
    RETURN
        node.text as text, score,
        { 
            document: d.id,
            entities: kg
        } AS metadata
    """
)

# Prompt for answering questions
instructions = (
    "Use the given context to answer the question. "
    "Reply with an answer that includes the id of the document and other relevant information from the text. "
    "If you don't know the answer, say you don't know. "
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instructions),
        ("human", "{input}"),
    ]
)

# Create retriever and chain
chunk_retriever = chunk_vector.as_retriever()
chunk_chain = create_stuff_documents_chain(llm, prompt)
chunk_retriever = create_retrieval_chain(
    chunk_retriever, 
    chunk_chain
)

# Query loop
def find_chunk(q):
    return chunk_retriever.invoke({"input": q})

while (q := input("> ")) != "exit":
    print(find_chunk(q))