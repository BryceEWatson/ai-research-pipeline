"""Sample documents for testing the RAG pipeline."""

SAMPLE_DOCUMENTS = [
    {
        "title": "Introduction to Retrieval-Augmented Generation",
        "content": """
Retrieval-Augmented Generation (RAG) is a natural language processing technique that combines the power of pre-trained language models with external knowledge retrieval. RAG systems work by first retrieving relevant documents from a knowledge base and then using these documents as context for generating responses.

The key advantage of RAG is that it allows language models to access up-to-date information beyond their training data. This is particularly important for applications requiring current information or domain-specific knowledge that may not be well-represented in the model's training corpus.

RAG systems typically consist of two main components: a retriever and a generator. The retriever searches for relevant documents based on the input query, while the generator produces the final response using both the query and the retrieved documents as input.

Common applications of RAG include question answering systems, chatbots, and research assistants. The technique has shown significant improvements in factual accuracy and reduces hallucinations compared to traditional language model approaches.
""",
        "source_url": "https://example.com/rag-intro",
        "eval_score": 0.9
    },
    {
        "title": "Vector Databases and Embeddings",
        "content": """
Vector databases are specialized databases designed to store and query high-dimensional vector representations of data. These databases are essential components of modern AI applications, particularly for similarity search and retrieval tasks.

Embeddings are dense vector representations of data that capture semantic meaning. Text embeddings, for example, convert words, sentences, or documents into numerical vectors where similar content has similar vector representations. This enables efficient semantic search capabilities.

Popular vector databases include Chroma, Pinecone, Weaviate, and Qdrant. These systems offer features like approximate nearest neighbor search, metadata filtering, and horizontal scaling to handle large-scale vector data.

In RAG systems, vector databases store embeddings of documents in the knowledge base. When a query is received, it is converted to an embedding and used to find the most similar document embeddings, enabling semantic retrieval of relevant information.

Key considerations when choosing a vector database include query performance, scalability, ease of integration, and support for metadata filtering and hybrid search capabilities.
""",
        "source_url": "https://example.com/vector-databases",
        "eval_score": 0.85
    },
    {
        "title": "Large Language Models and AI Safety",
        "content": """
Large Language Models (LLMs) like GPT-4, Claude, and Llama have demonstrated remarkable capabilities in natural language understanding and generation. However, their deployment raises important safety and reliability concerns.

One major challenge is hallucination, where models generate plausible-sounding but factually incorrect information. This is particularly problematic in high-stakes applications like healthcare, legal advice, or scientific research. RAG systems help mitigate this issue by grounding responses in retrieved factual content.

Another concern is bias amplification, where models may reflect or amplify biases present in their training data. Careful evaluation and bias detection are essential for responsible AI deployment.

AI safety measures include factual consistency scoring, output monitoring, and human oversight. The Vectara Factual Consistency Score (FCS) is one example of automated fact-checking for AI-generated content.

Best practices for safe LLM deployment include comprehensive testing, gradual rollout, continuous monitoring, and clear limitations disclosure to users. Organizations should also implement proper governance frameworks and ethical guidelines for AI use.
""",
        "source_url": "https://example.com/ai-safety",
        "eval_score": 0.8
    },
    {
        "title": "Evaluation Metrics for RAG Systems",
        "content": """
Evaluating Retrieval-Augmented Generation (RAG) systems requires multiple metrics that assess both retrieval quality and generation performance. Traditional NLP metrics alone are insufficient for comprehensive RAG evaluation.

Retrieval metrics focus on the quality of document retrieval. Key metrics include:
- Precision@K: Proportion of retrieved documents that are relevant
- Recall@K: Proportion of relevant documents that are retrieved
- Mean Reciprocal Rank (MRR): Average of reciprocal ranks of first relevant documents
- NDCG (Normalized Discounted Cumulative Gain): Considers both relevance and ranking order

Generation metrics evaluate the quality of the final response:
- BLEU and ROUGE scores for text similarity
- BERTScore for semantic similarity
- Factual accuracy measures
- Human evaluation for coherence and helpfulness

End-to-end metrics assess the overall system performance:
- Answer accuracy on question-answering benchmarks
- Factual consistency scores (like Vectara FCS)
- Latency and throughput measurements
- User satisfaction scores

Modern evaluation frameworks like LangSmith provide tools for automated evaluation and continuous monitoring of RAG system performance in production environments.
""",
        "source_url": "https://example.com/rag-evaluation",
        "eval_score": 0.88
    }
]


def get_sample_documents():
    """Return sample documents for testing."""
    return SAMPLE_DOCUMENTS