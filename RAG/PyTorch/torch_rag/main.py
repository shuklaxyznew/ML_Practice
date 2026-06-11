from torchrag import EmbeddingModel, RAGPipeline

DOCUMENTS = [
    {
        "title": "PyTorch Basics",
        "content": """PyTorch is an open-source machine learning framework developed by Meta AI.
        It uses dynamic computation graphs making debugging easier than TensorFlow.
        Tensors are the core data structure similar to NumPy arrays but with GPU support.
        The torch.no_grad context manager disables gradient tracking during inference
        saving memory and speeding up predictions. PyTorch supports CUDA for GPU acceleration."""
    },
    {
        "title": "RAG Architecture",
        "content": """Retrieval-Augmented Generation combines a retrieval system with a language model
        to answer questions from a knowledge base. During indexing documents are split into chunks
        embedded into vectors and stored in a vector database. During querying the user question is
        embedded similar chunks retrieved using cosine similarity and an LLM generates an answer
        grounded in the retrieved context. RAG reduces hallucinations because answers are grounded
        in real documents."""
    },
    {
        "title": "ONNX Deployment",
        "content": """ONNX is an open format for AI models allowing models trained in PyTorch to be
        deployed without the original framework. ONNX Runtime runs ONNX models 2 to 5 times faster
        than native PyTorch. It supports CPU GPU and edge hardware. Dynamic axes allow variable
        batch sizes and sequence lengths. ONNX is useful for production deployments where you want
        to minimize dependencies and maximize performance."""
    },
    {
        "title": "GPU vs CPU",
        "content": """GPUs are optimized for parallel matrix operations making them ideal for deep
        learning. A GPU can perform thousands of operations simultaneously while a CPU handles
        tasks sequentially. Use CPU for low-traffic applications and GPU for high-throughput
        production systems. In PyTorch use model.to(device) and tensor.to(device) to move data.
        Both model and input data must be on the same device or you get a runtime error."""
    },
    {
        "title": "Batch Processing",
        "content": """Batch processing groups multiple inputs together for a single model forward
        pass. Instead of embedding one text at a time you process 16 32 or 64 texts simultaneously.
        This exploits GPU parallelism as the GPU performs the same operation on many data points
        at once. Larger batch sizes mean higher GPU utilization but more memory usage.
        If your batch is too large you get an Out of Memory error."""
    },
]

QUESTIONS = [
    "What is PyTorch and why is it used?",
    "How does ONNX make deployment faster?",
    "Why should I use GPU instead of CPU?",
    "What is batch processing and why does it matter?",
    "How does RAG reduce hallucinations?",
]


def main():
    print("=" * 55)
    print("  TorchRAG — Demo")
    print("=" * 55)

    model = EmbeddingModel()
    model.load()
    model.save("models/embedding.pt")
    model.export_onnx("models/embedding.onnx")

    rag = RAGPipeline(embedder=model)
    rag.ingest(DOCUMENTS, batch_size=4)
    rag.store.save("vectorstore/store.npz")

    print("\n" + "=" * 55)
    for q in QUESTIONS:
        result = rag.query(q, top_k=2)
        print(f"\nQ: {result['question']}")
        print(f"   Sources : {result['sources']}")
        print(f"   Scores  : {result['scores']}")
        print(f"   Answer  : {result['answer'][:120]}...")
        print("-" * 55)


if __name__ == "__main__":
    main()
