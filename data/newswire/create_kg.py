import os, csv

from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
# --- GCP Vertex AI Import ---
from langchain_google_vertexai import ChatVertexAI
# ---------------------------
from langchain_neo4j import Neo4jGraph
from langchain_community.graphs.graph_document import Node, Relationship

from dotenv import load_dotenv
load_dotenv()

ARTICLES_REQUIRED = [6,8,22]
DATA_PATH = 'llm-knowledge-graph/data/newswire'
ARTICLE_FILENAME = os.path.join(DATA_PATH, 'articles.csv')

GOOGLE_CLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
GOOGLE_CLOUD_LOCATION = os.getenv('GOOGLE_CLOUD_LOCATION')

def create_kg():

    articles_csvfile = open(ARTICLE_FILENAME, encoding="utf8", newline='')
    articles_csv = csv.DictReader(articles_csvfile)

    llm = ChatVertexAI(
        model_name="gemini-2.5-flash-lite",  # Using the specified Gemini model
        temperature=0,                   # Using the specified temperature
        project=GOOGLE_CLOUD_PROJECT,
        location=GOOGLE_CLOUD_LOCATION
    )

    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD")
    )

    article_transformer = LLMGraphTransformer(
        llm=llm,
        # allowed_nodes=["Person", "Organization", "Location", "Outcome", "Event", "Object"],
    )

    article_num = -1
    for article in articles_csv:
        
        article_num += 1
        if article_num not in ARTICLES_REQUIRED:
            continue

        print(article_num)

        article_doc = [Document(
            page_content=article["text"], 
            metadata={"id": article["id"]}
        )]

        graph_docs = article_transformer.convert_to_graph_documents(article_doc)

        graph.query(
            "MERGE (a:Article {id: $id}) SET a.date = $date, a.text = $text",
            {"id": article["id"], "date": article["date"], "text": article["text"]}
        )

        article_node = Node(
            id=article["id"],
            type="Article"
        )
        
        for graph_doc in graph_docs:
            for node in graph_doc.nodes:
                graph_doc.relationships.append(
                    Relationship(
                        source=article_node,
                        target=node, 
                        type="HAS_ENTITY"
                        )
                    )
        graph.add_graph_documents(graph_docs)

    articles_csvfile.close()

if __name__ == "__main__":
    create_kg()