from langchain_community.vectorstores import FAISS

# Path to your saved index
INDEX_PATH = "vectorstore/faiss_index"

# Load FAISS index WITHOUT the embeddings model (just to inspect)
vectorstore = FAISS.load_local(INDEX_PATH, embeddings=None, allow_dangerous_deserialization=True)

# Access the underlying FAISS index
faiss_index = vectorstore.index

# Print index details
print("FAISS index type:", type(faiss_index))
print("Number of vectors in index:", faiss_index.ntotal)
print("Dimension of embeddings:", faiss_index.d)
