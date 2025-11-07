import torch
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Разбивает список документов на более мелкие части для обработки.

    Args:
        documents (List[Document]): Список объектов Document, которые нужно разбить на части.

    Returns:
        List[Document]: Список более мелких объектов Document, каждый из которых представляет часть исходных документов.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks: list[Document] = text_splitter.split_documents(documents)
    return chunks


def get_embedding_function() -> HuggingFaceEmbeddings:
    """
    Инициализирует и возвращает модель встраивания для векторизации текста.

    Returns:
        HuggingFaceEmbeddings: Экземпляр класса HuggingFaceEmbeddings, настроенный на использование
                               конкретной модели для генерации встраиваний.
    """
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs: dict = {"device": device}
    encode_kwargs: dict = {"normalize_embeddings": False}

    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
    return embeddings
