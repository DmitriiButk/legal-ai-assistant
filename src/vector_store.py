from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

CHROMA_PATH: str = "chroma_db"


def create_vector_store(chunks: list[Document], embedding_function: Any) -> Chroma:
    """
    Создает векторное хранилище Chroma из предоставленных частей документа и функции встраивания.

    Args:
        chunks (List[Document]): Список объектов Document, которые нужно встроить и сохранить.
        embedding_function (Any): Функция или модель встраивания, используемая для векторизации частей документа.

    Returns:
        Chroma: Экземпляр векторного хранилища Chroma, содержащего встроенные документы.
    """
    vector_store = Chroma.from_documents(documents=chunks, embedding=embedding_function, persist_directory=CHROMA_PATH)
    print(f"Успешно создано и сохранено векторное хранилище в {CHROMA_PATH}")
    return vector_store
