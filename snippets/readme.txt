snippet file
1.chunk_data.py-gcp need no
text splitting and chunk creation

2.creat_graph.py-gcp no need
It reads through pre-processed text chunks, uses a Large Language Model (LLM) to extract structured entities and relationships from each chunk, and then links those extracted entities back to the original source chunk. Finally, it persists all the extracted data into a Neo4j database.

3.extract_nodes.py-gcp need -yes
read text and translate the facts within it into a structured, machine-readable Knowledge Graph format using the capabilities of the Gemini LLM.

4.load_data.py-gcp need-no
preparing the raw text data by scraping all the courses' PDF contents into memory, ready to be chunked, embedded, and fed into the Knowledge Graph and RAG pipeline.

5.vectorize_data.py-gcp embedding  need -yes
this program transforms raw text documents into a searchable, structured, and vectorized data source within a graph database.