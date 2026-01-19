from pinecone import Pinecone, ServerlessSpec

PINECONE_API_KEY = "pcsk_6E6LwA_EZP8x4Zo4yt4QJGoeE6jt6GE6Zq7sRVN9Zi1Q7zMiiLMzAVAq5HdDBdt5BD5dzW"
PINECONE_ENV = "us-east-1"  

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
