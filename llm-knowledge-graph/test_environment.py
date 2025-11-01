import os
import unittest
from dotenv import load_dotenv, find_dotenv

load_dotenv()

class TestEnvironment(unittest.TestCase):

    skip_env_variable_tests = True
    skip_vertexai_test = True
    skip_neo4j_test = True

    def test_env_file_exists(self):
        env_file_exists = True if find_dotenv() else False
        if env_file_exists:
            TestEnvironment.skip_env_variable_tests = False
        self.assertTrue(env_file_exists, ".env file not found.")

    def env_variable_exists(self, variable_name):
        self.assertIsNotNone(
            os.getenv(variable_name),
            f"{variable_name} not found in .env file"
        )

    def test_vertexai_variables(self):
        if TestEnvironment.skip_env_variable_tests:
            self.skipTest("Skipping Vertex AI env variable test")

        self.env_variable_exists('GOOGLE_API_KEY')
        self.env_variable_exists('GOOGLE_CLOUD_PROJECT')
        self.env_variable_exists('GOOGLE_CLOUD_LOCATION')
        TestEnvironment.skip_vertexai_test = False

    def test_neo4j_variables(self):
        if TestEnvironment.skip_env_variable_tests:
            self.skipTest("Skipping Neo4j env variables test")

        self.env_variable_exists('NEO4J_URI')
        self.env_variable_exists('NEO4J_USERNAME')
        self.env_variable_exists('NEO4J_PASSWORD')
        TestEnvironment.skip_neo4j_test = False

    def test_vertexai_connection(self):
        if TestEnvironment.skip_vertexai_test:
            self.skipTest("Skipping Vertex AI test")

        try:
            from langchain_google_vertexai import ChatVertexAI
            llm = ChatVertexAI(
                model_name="gemini-2.5-flash-lite",
                temperature=0,
                project=os.getenv("GOOGLE_CLOUD_PROJECT"),
                location=os.getenv("GOOGLE_CLOUD_LOCATION")
            )
            response = llm.invoke("Hello, are you working?")
            self.assertIsNotNone(response)
        except Exception as e:
            self.fail(f"Vertex AI connection failed: {e}")

    def test_neo4j_connection(self):
        if TestEnvironment.skip_neo4j_test:
            self.skipTest("Skipping Neo4j connection test")

        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            os.getenv('NEO4J_URI'),
            auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
        )
        try:
            driver.verify_connectivity()
            connected = True
        except Exception as e:
            connected = False

        driver.close()
        self.assertTrue(
            connected,
            "Neo4j connection failed. Check the NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD values in .env file."
        )

def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestEnvironment('test_env_file_exists'))
    suite.addTest(TestEnvironment('test_vertexai_variables'))
    suite.addTest(TestEnvironment('test_neo4j_variables'))
    suite.addTest(TestEnvironment('test_vertexai_connection'))
    suite.addTest(TestEnvironment('test_neo4j_connection'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
