import os
from dotenv import load_dotenv
from autogen import ConversableAgent
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

load_dotenv()


def initialize_index():
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("my-docs-collection")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if chroma_collection.count() > 0:
        print("Loading existing index...")
        return VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )
    else:
        print("Creating new index...")
        documents = SimpleDirectoryReader("./documents").load_data()
        return VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )


index = initialize_index()
query_engine = index.as_query_engine()


def create_prompt(user_input):
    result = query_engine.query(user_input)

    prompt = f"""
    Your Task: Provide a concise and informative response to the user's query, drawing on the provided context.

    Context: {result}

    User Query: {user_input}

    Guidelines:
    1. Relevance: Focus directly on the user's question.
    2. Conciseness: Avoid unnecessary details.
    3. Accuracy: Ensure factual correctness.
    4. Clarity: Use clear language.
    5. Contextual Awareness: Use general knowledge if context is insufficient.
    6. Honesty: State if you lack information.

    Response Format:
    - Direct answer
    - Brief explanation (if necessary)
    - Citation (if relevant)
    - Conclusion
    """

    return prompt


llm_config = {
    "config_list": [
        {
            "model": "llama-3.1-8b-instant",
            "api_key": os.getenv("GROQ_API_KEY"),
            "api_type": "groq",
        }
    ]
}

rag_agent = ConversableAgent(
    name="RAGbot",
    system_message="You are a RAG chatbot",
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="NEVER",
)


def main():
    print("Welcome to RAGbot! Type 'exit', 'quit', or 'bye' to end the conversation.")
    while True:
        user_input = input(f"\nUser: ")

        if user_input.lower() in ["exit", "quit", "bye"]:
            print(f"Goodbye! Have a great day!!")
            break

        prompt = create_prompt(user_input)

        reply = rag_agent.generate_reply(messages=[{"content": prompt, "role": "user"}])

        print(f"\nRAGbot: {reply['content']}")


if __name__ == "__main__":
    main()
