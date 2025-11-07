from typing import Any

import torch
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import pipeline

from src.text_processing import get_embedding_function

CHROMA_PATH: str = "chroma_db"

PROMPT_TEMPLATE = """
Ты — юрист-аналитик.
Ответь на вопрос пользователя на основе предоставленного контекста.
Отвечай только на русском языке, кратко и по существу.
Не повторяй контекст или вопрос в ответе.

Контекст:
{context}

Вопрос:
{question}

Ответ:
"""


def format_docs(docs: list) -> str:
    """
    Форматирует список объектов документов в единую строку, соединяя их содержимое.

    Args:
        docs (list): Список объектов документов, каждый из которых имеет атрибут `page_content`.

    Returns:
        str: Единая строка, содержащая объединенное содержимое всех документов, разделенное двойными переносами строк.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def get_rag_chain(context_override: str | None = None) -> Any:
    """
    Создает и возвращает цепочку RAG (Retrieval-Augmented Generation) с использованием локальной языковой модели.

    Цепочка объединяет ретривер (для получения релевантного контекста из векторного хранилища) или предоставленный
    контекст с языковой моделью для генерации ответов на запросы пользователя.

    Args:
        context_override (Optional[str]): Если предоставлен, эта строка используется как контекст вместо получения
                                          из векторного хранилища. По умолчанию None.

    Returns:
        Any: Исполняемая цепочка LangChain, которая обрабатывает запрос и генерирует ответ.
    """
    embedding_function = get_embedding_function()
    vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.2,
        device=0 if device == "cuda" else -1,
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    if context_override:
        chain = {"context": lambda x: context_override, "question": RunnablePassthrough()} | prompt | llm
    else:
        chain = {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm

    return chain
