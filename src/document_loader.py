from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_documents(file_path: str) -> list[Document]:
    """
    Загружает PDF-документ по указанному пути и возвращает его содержимое в виде списка объектов Document.

    Каждый объект Document представляет страницу PDF.

    Args:
        file_path (str): Путь к PDF-файлу, который нужно загрузить.

    Returns:
        List[Document]: Список объектов Document, каждый из которых содержит содержимое страницы PDF.
    """
    loader = PyPDFLoader(file_path)
    pages: list[Document] = loader.load()
    return pages
