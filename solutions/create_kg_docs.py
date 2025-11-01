import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs.graph_document import Node, Relationship

# Load environment variables
DOCS_PATH = "llm-knowledge-graph/data/course/pdfs"
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

# Choose loader: CSV or PDF
# Uncomment the one you want to use

# CSV Loader
# loader = CSVLoader(file_path="path/to/your_file.csv")

# PDF Loader
loader = DirectoryLoader(DOCS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)

# Split text into chunks
text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=1500, chunk_overlap=200)

docs = loader.load()
chunks = text_splitter.split_documents(docs)

# Graph transformer using Vertex AI LLM
doc_transformer = LLMGraphTransformer(llm=llm)

for chunk in chunks:
    filename = os.path.basename(chunk.metadata["source"])
    chunk_id = f"{filename}.{chunk.metadata['page']}"
    print("Processing -", chunk_id)

    # Embed the chunk
    chunk_embedding = embedding_provider.embed_query(chunk.page_content)

    # Add the Document and Chunk nodes to the graph
    properties = {
        "filename": filename,
        "chunk_id": chunk_id,
        "text": chunk.page_content,
        "embedding": chunk_embedding
    }

    graph.query("""
        MERGE (d:Document {id: $filename})
        MERGE (c:Chunk {id: $chunk_id})
        SET c.text = $text
        MERGE (d)<-[:PART_OF]-(c)
        WITH c
        CALL db.create.setNodeVectorProperty(c, 'textEmbedding', $embedding)
    """, properties)

    # Generate entities and relationships
    graph_docs = doc_transformer.convert_to_graph_documents([chunk])

    # Link entities to the chunk
    for graph_doc in graph_docs:
        chunk_node = Node(id=chunk_id, type="Chunk")
        for node in graph_doc.nodes:
            graph_doc.relationships.append(
                Relationship(source=chunk_node, target=node, type="HAS_ENTITY")
            )

    # Add graph documents to Neo4j
    graph.add_graph_documents(graph_docs)

# Create vector index
graph.query("""
    CREATE VECTOR INDEX `chunkVector`
    IF NOT EXISTS
    FOR (c: Chunk) ON (c.textEmbedding)
    OPTIONS {indexConfig: {
        `vector.dimensions`: 768,
        `vector.similarity_function`: 'cosine'
    }};
""")