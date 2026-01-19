from pinecone import Pinecone, ServerlessSpec

PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"] 

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "lenskart-eyewear"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=512,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=PINECONE_ENV
        )
    )

index = pc.Index(index_name)
print("Pinecone index ready!")
print(index.describe_index_stats())
