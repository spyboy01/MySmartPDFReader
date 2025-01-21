# Import necessary modules
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_graph_from_storage
from llama_index.core.response.pprint_utils import pprint_response
from sentence_transformers import SentenceTransformer
import os
 
# Initialize a custom embedding model using Hugging Face's SentenceTransformer
class HuggingFaceEmbeddedModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # Load the specified SentenceTransformer model
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, text):
        # Generate and return embeddings for the given text
        return self.model.encode(text)

# Function to create an index using Hugging Face embeddings
def create_huggingface_index(documents, embedding_model):
    # Generate embeddings for all documents
    #embeddings = [embedding_model.get_embedding(doc.get_text()) for doc in documents]
    embeddings = [embedding_model.get_embedding(doc.text) for doc in documents]
    print ("##################################################################")
    print ("Converting the text into embedding", embeddings)
    print ("##################################################################")
    print("Printing the type of embeddings", type(embeddings))
    print("this is to confirm llama-index methods", dir(VectorStoreIndex))



    # Create a VectorStoreIndex from the embeddings
    #return VectorStoreIndex.from_embeddings(documents, embeddings)
    # Create a vector store and initialize the index
    vector_store = SimpleVectorStore.from_documents(documents)
    index = VectorStoreIndex.from_vector_store(vector_store)
    return index

# Specify the directory for storing or loading the index
index_cache_non_api = "./index_store_non_api"

# Check if the index already exists in storage
if not os.path.exists(index_cache_non_api):
    # If the index doesn't exist, read data from a local directory
    documents = SimpleDirectoryReader("./data").load_data()
    print("Documents loaded:", documents)
    
    # Initialize the custom Hugging Face embedding model
    hf_model = HuggingFaceEmbeddedModel()
    
    # Create the index using the documents and Hugging Face model
    index = create_huggingface_index(documents, hf_model)
    
    # Persist the created index for future use
    index.storage_context.persist(persist_dir=index_cache_non_api)
else:
    # If the index already exists, load it from the specified directory
    storage_context = StorageContext.from_defaults(persist_dir=index_cache_non_api)
    index = load_graph_from_storage(storage_context)

# Query the index using a query engine
query_engine = index.as_query_engine()
response = query_engine.query("What is meditation?")

# Print the response with source information
pprint_response(response, show_source=True)
