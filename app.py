import json
import os
import time

import streamlit as st
from dotenv import load_dotenv

from langchain.retrievers.ensemble import EnsembleRetriever

from rag import embedding
from rag import ConversationChain
from rag import FileDataLoader, URLDataLoader
from rag import *
from rag import CustomBM25
from rag import Retriever_flitering_and_reranking
from rag import LLMFactory
from rag import Question_Prompt
from rag import TextSplitter

# load_dotenv()
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "RAG test"
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Check if the file does not exist
if not os.path.exists("responses.jsonl"):
    # Create an empty responses.jsonl file
    with open("responses.jsonl", "w") as file:
        pass  # Creating an empty file


class StreamlitChatApp:
    def __init__(self):

        st.set_page_config(page_title="DirChat", page_icon=":books:")
        st.title("_Funzin :red[QA Chat]_ :books:")
        if "init" not in st.session_state:
            self.initialize_session_state()
        self.load_config()
        self.setup_sidebar()
        # if self.clear_DB:
        #     self.clear_DB()
        # TODO : DB 클리어 로직구현

        if self.process:
            self.processing()

        self.display_messages()
        self.handle_chat()

    def load_config(self):
        file_path = "config.json"
        with open(file_path, "r") as file:
            self.config = json.load(file)

    def initialize_session_state(self):
        """세션 상태를 초기화합니다."""

        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {
                    "role": "assistant",
                    "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!",
                }
            ]
        if "source" not in st.session_state:
            st.session_state.source = []

        if "times" not in st.session_state:
            st.session_state.times = []

        st.session_state["init"] = True

    def setup_sidebar(self):
        """사이드바 구성을 처리합니다."""
        with st.sidebar:
            self.load_or_save = st.selectbox(
                "Save or Load", ["Upload new files", "Proceed with existing files"]
            )
            if self.load_or_save == "Proceed with existing files":
                st.session_state.uploaded = None
            else:
                self.data_choice = st.selectbox("Select File Type", self.config["Doc_menu"]["menu"])

                if self.data_choice in ["url", "slack"]:
                    st.session_state.uploaded = st.text_input("Enter URL")
                else:
                    st.session_state.uploaded = st.file_uploader(
                        "파일을 업로드하세요",
                        type=(
                            self.data_choice
                            if self.data_choice in self.config["Doc_menu"]["menu"][:-2]
                            else None
                        ),
                        accept_multiple_files=True,
                    )

            self.splitter_choice = st.selectbox("Select Text Splitter", self.config["TextSplitter"]["menu"])
            self.model_choice = st.selectbox("Select Model Provider", ["Docker", "OpenAI", "Local"])
            self.model = self.select_model()
            self.embedding_model = self.select_embedding_model()
            self.chain_type = st.selectbox("Select Chain Type", ["map_reduce", "refine", "map_rerank", "stuff"])
            self.process = st.button("Process")

    def select_model(self):
        """모델 선택을 관리합니다."""
        if self.model_choice == "Local":
            self.openai_api_key = None
            model_menu = self.config["LLM"]["HuggingFace"]["menu"]
            model = st.selectbox("Select HuggingFace Model", model_menu)
            if model == "직접 입력":
                model = st.text_input("Enter HuggingFace LLM")
        elif self.model_choice == "Docker":
            self.openai_api_key = None
            model_menu = self.config["LLM"]["Docker"]["menu"]
            model = st.selectbox("Select Docker Model", model_menu)
        elif self.model_choice == "OpenAI":
            load_dotenv()
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            model_menu = self.config["LLM"]["OpenAI"]["menu"]
            model = st.selectbox("Select OpenAI Model", model_menu)
            if model == "직접 입력":
                model = st.text_input("Enter OpenAI Model")

        return model

    def select_embedding_model(self):
        """임베딩 모델 선택을 관리합니다."""
        embedding_menu = self.config["EmbeddingModel"]["menu"]
        embedding_model = st.selectbox("Select Embedding Model", embedding_menu)
        if embedding_model == "직접 입력":
            embedding_model = st.text_input("Enter Embedding Model")
        return embedding_model

    def preprocessing_files(self, files):
        """파일을 처리합니다."""
        assert len(files) > 0

        text_chunks = []
        for file in files:
            if self.data_choice == "url" or self.data_choice == "slack":
                files_text = URLDataLoader.load_data(uploaded=file)
            else:
                files_text = FileDataLoader.load_data(uploaded=file)

            text_chunk = self.text_splitter.split_documents(files_text)
            text_chunks.extend(text_chunk)
        return text_chunks

    def processing(self):
        """RAG 프로세스를 관리합니다."""
        loading_message = st.empty()
        loading_message.markdown("<h1 style='text-align: center;'>Loading...</h1>", unsafe_allow_html=True)
        self.embedding = embedding.get_embedding(self.embedding_model)
        if st.session_state.uploaded:
            self.text_splitter = TextSplitter(mode=self.splitter_choice, embeddings=self.embedding)
            files = st.session_state.uploaded
            docs = self.preprocessing_files(files)
            faiss_add_and_save_documents(docs, self.embedding)
            bm25_add_and_save_documents(docs)
        else:
            # 파일 경로
            bm25_index_path = "data/bm25_index/index.pkl"
            faiss_index_path = "data/faiss_index/index.faiss"
            # 파일 존재 여부 확인
            assert os.path.exists(bm25_index_path), f"{bm25_index_path} 파일이 존재하지 않습니다."
            assert os.path.exists(faiss_index_path), f"{faiss_index_path} 파일이 존재하지 않습니다."
        BM25DB = CustomBM25.load_local(folder_path="data/bm25_index/index.pkl")
        FAISSDB = FAISS.load_local(
            folder_path="data/faiss_index", embeddings=self.embedding, allow_dangerous_deserialization=True
        ).as_retriever()
        retriever = EnsembleRetriever(retrievers=[BM25DB, FAISSDB], weights=[0.5, 0.5], c=60)
        self.retriever = Retriever_flitering_and_reranking(retriever=retriever, embeddings=self.embedding)
        llm_factory = LLMFactory(
            self.model_choice, self.model, self.openai_api_key, temperature=0, streaming=True
        )
        self.llm = llm_factory.llm_instance.get_llm()
        self.prompt = Question_Prompt()
        conversation_manager = ConversationChain(QA_PROMPT=self.prompt, retriever=self.retriever, llm=self.llm)
        st.session_state.conversation = conversation_manager.get_chain(self.chain_type)
        st.session_state.processComplete = True
        loading_message.empty()

    def display_messages(self):
        """메시지 표시를 관리합니다."""
        try:
            for i in range(len(st.session_state.messages)):
                with st.chat_message(st.session_state.messages[i]["role"]):
                    st.markdown(st.session_state.messages[i]["content"])
                    if st.session_state.messages[i]["role"] == "assistant" and i >= 2:
                        with st.expander("참고 문서 확인", expanded=False):
                            for source_document in st.session_state.source[i // 2 - 1]:
                                headers = {
                                    key: value
                                    for key, value in source_document.metadata.items()
                                    if key.startswith("header")
                                }
                                source = source_document.metadata["source"]
                                sorted_values = "/".join(value for key, value in sorted(headers.items()))
                                st.markdown(
                                    f"**Source:** {sorted_values} from {source}\n"
                                    f"**Relevance score:** {source_document.state['query_similarity_score']:.3f}\n",
                                    help=source_document.page_content,
                                )

                        st.write(f"Time taken: {st.session_state.times[i // 2 - 1]:.2f} sec")

        except:
            pass

    def handle_chat(self):
        """채팅 로직을 관리합니다."""

        if query := st.chat_input("질문을 입력해주세요."):
            st.session_state.query = query
            st.session_state.messages.append({"role": "user", "content": st.session_state.query})

            with st.chat_message("user"):
                st.markdown(query)

            start_time = time.time()
            with st.chat_message("assistant"):
                chain = st.session_state.conversation

                with st.spinner("Thinking..."):
                    result = chain.invoke({"question": st.session_state.query})

                    # with get_openai_callback() as cb:
                    #     st.session_state.chat_history = result["chat_history"]
                    st.session_state.response = result["answer"]
                    source_documents = result["source_documents"]
                st.markdown(st.session_state.response)
                with st.expander("참고 문서 확인"):
                    for source_document in source_documents:
                        headers = {
                            key: value
                            for key, value in source_document.metadata.items()
                            if key.startswith("header")
                        }
                        source = source_document.metadata["source"]
                        sorted_values = "/".join(value for _, value in sorted(headers.items()))
                        st.markdown(
                            f"**Source:** {sorted_values} from {source}\n\n"
                            f"**Relevance score:** {source_document.state['query_similarity_score']:.3f}",
                            help=source_document.page_content,
                        )

                    st.session_state.source.extend([source_documents])

                col1, col2, _, _, _, _, _, _, _, _ = st.columns(10)
                # 사용자 반응 버튼
                with col1:
                    st.session_state.thumbup = st.button("👍", key="Good", on_click=self.save_positive_response)
                with col2:
                    st.session_state.thumbdown = st.button(
                        "👎", key="Bad", on_click=self.save_negative_response
                    )

                st.session_state.messages.append({"role": "assistant", "content": st.session_state.response})
                end_time = time.time()

                elapsed_time = end_time - start_time
                st.write(f"Time taken: {elapsed_time:.2f} sec")
                st.session_state.times.extend([elapsed_time])

    def save_response(self, is_positive):
        # JSON 파일에 반응 저장
        response_file = "responses.jsonl"

        response_entry = {
            "query": st.session_state.query,
            "result": st.session_state.response,
            "response": 1 if is_positive else -1,
        }

        with open(response_file, "a") as f:
            f.write(json.dumps(response_entry, ensure_ascii=False) + "\n")

    def save_positive_response(self):
        self.save_response(is_positive=True)

    def save_negative_response(self):
        self.save_response(is_positive=False)


if __name__ == "__main__":
    app = StreamlitChatApp()
