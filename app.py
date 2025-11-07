import os

import streamlit as st

from src.document_loader import load_documents
from src.rag_chain import get_rag_chain
from src.text_processing import get_embedding_function
from src.text_processing import split_documents
from src.vector_store import CHROMA_PATH
from src.vector_store import create_vector_store


def load_css(file_name):
    """
    Загружает CSS-файл и применяет его к Streamlit-приложению.

    Args:
        file_name (str): Имя CSS-файла, который нужно загрузить.
    """
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css("assets/style.css")
st.title("Юридический ассистент для анализа договоров")

if "document_path" not in st.session_state:
    st.session_state.document_path = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Загрузка документа")
    uploaded_file = st.file_uploader("Выберите PDF файл", type="pdf")

    if uploaded_file is not None:
        if not os.path.exists("data"):
            os.makedirs("data")
        temp_file_path = os.path.join("data", uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.document_path = temp_file_path

        if st.button("Загрузить и обработать документ"):
            with st.spinner("Идет обработка документа..."):
                documents = load_documents(st.session_state.document_path)
                chunks = split_documents(documents)
                embedding_function = get_embedding_function()
                create_vector_store(chunks, embedding_function)
                st.success("Документ успешно обработан и готов к анализу!")

st.header("Задайте вопрос по документу:")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ваш вопрос..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not st.session_state.document_path or not os.path.exists(CHROMA_PATH):
        with st.chat_message("assistant"):
            st.warning("Пожалуйста, сначала загрузите и обработайте документ.")
    else:
        with st.chat_message("assistant"), st.spinner("Думаю..."):
            analytical_keywords = ["анализируй", "анализ", "риски", "сильные и слабые стороны", "проверь"]
            is_analytical_query = any(keyword in prompt.lower() for keyword in analytical_keywords)

            context_override = None
            if is_analytical_query:
                st.info("Аналитический запрос. Загружаю полный контекст документа...")
                full_documents = load_documents(st.session_state.document_path)
                context_override = "\n\n".join([doc.page_content for doc in full_documents])

            rag_chain = get_rag_chain(context_override=context_override)

            stream_gen = rag_chain.stream(prompt)
            response_container = st.empty()
            full_response = ""

            for chunk in stream_gen:
                full_response += chunk
                response_container.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
