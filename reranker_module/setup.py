from setuptools import setup, find_packages

setup(
    name="reranker-module",
    version="1.0.0",
    description="Module re-ranking 2 tầng (Bi-Encoder + Cross-Encoder) cho RAG",
    packages=find_packages(include=["reranker", "reranker.*"]),
    python_requires=">=3.9",
    install_requires=[
        "sentence-transformers>=4.0.0",
        "transformers>=4.40.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "faiss-cpu>=1.7.4",
    ],
    extras_require={
        "train": ["datasets>=2.14.0", "accelerate>=0.26.0"],
        "flag": ["FlagEmbedding>=1.2.0"],
        "dev": ["pytest>=7.4.0"],
    },
)
