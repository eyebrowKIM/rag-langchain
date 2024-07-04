import os

from typing import List, Optional, Any, Callable
from langchain_core.pydantic_v1 import Field

import pickle
import sqlite3

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.retrievers import RetrieverLike
from langchain_core.runnables.config import ensure_config
from langchain_core.load.dump import dumpd

from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.chroma import Chroma
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from kiwipiepy import Kiwi


def kiwi_tokenize(text: str) -> List[str]:
    kiwi = Kiwi()
    return [token.form for token in kiwi.tokenize(text)]


class CustomBM25(BM25Retriever):
    docs: List[Document] = Field(default_factory=list)

    def __init__(
        self,
        docs: List[Document],
        vectorizer: Any = None,
        k: int = 4,
        preprocess_func: Callable[[str], List[str]] = kiwi_tokenize,
    ):
        super().__init__(docs=docs, vectorizer=vectorizer, k=k, preprocess_func=preprocess_func)
        self.docs = docs

    @classmethod
    def load_local(cls, folder_path="data/bm25_index/index.pkl") -> BM25Retriever:
        """
        주어진 경로에 있는 Pickle 파일에서 Document 리스트를 불러옵니다.

        :param file_path: Pickle 파일 경로
        :return: Document 객체의 리스트
        """
        try:
            with open(folder_path, "rb") as file:
                docs = pickle.load(file)
        except FileNotFoundError:
            raise FileNotFoundError("The file does not exist. Please upload a file")

        bm25 = BM25Retriever.from_documents(
            documents=docs,
            preprocess_func=kiwi_tokenize,
        )

        return bm25

    def clear(self):
        """
        Clear the BM25 index.
        """
        conn = sqlite3.connect("data/bm25_index.db")
        cursor = conn.cursor()

        cursor.execute("DROP TABLE documents")
        conn.commit()
        conn.close()
