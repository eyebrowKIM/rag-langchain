from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings


def get_embedding(model_name="BAAI/bge-m3"):
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        cache_folder="./cache/embedding/",
        model_kwargs={"device": "cuda:2"},
        encode_kwargs={"normalize_embeddings": True},
    )

    return embeddings
