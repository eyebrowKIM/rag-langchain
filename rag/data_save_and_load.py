import os

from typing import List

import pickle
import sqlite3

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores.faiss import FAISS


def default_preprocessing_func(text: str) -> List[str]:
    return text.split()


def faiss_add_and_save_documents(chunk: List[Document], embeddings: Embeddings):
    """
    Add new documents to the FAISS index and save it.
    :param chunk: Document chunk to be added
    :param embeddings: Embeddings function or object
    :param folder_path: Path to save the FAISS index
    """
    try:
        db = FAISS.load_local(folder_path="data/faiss_index", embeddings=embeddings)
        db.add_documents(chunk)
        db.save_local(folder_path="data/faiss_index")
        return db
    except Exception as e:
        print(f"An error occurred: {e}")
        db = FAISS.from_documents(chunk, embeddings)
        db.save_local(folder_path="data/faiss_index")
        return db

    # def clear(self):
    #     """
    #     Clear the FAISS index.
    #     """
    #     self.delete([self.index_to_docstore_id[:]])


def bm25_save_local(docs: List[Document], file_path: str = "data/bm25_index/index.pkl"):
    """
    주어진 경로에 Pickle 파일을 생성하고 Document 리스트를 저장합니다.
    :param file_path: Pickle 파일 경로
    """
    # 파일 경로에서 디렉터리 경로를 추출합니다.
    dir_name = os.path.dirname(file_path)
    # 디렉터리가 존재하지 않으면 생성합니다.
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with open(file_path, "wb") as file:
        pickle.dump(docs, file)


def bm25_load_local(folder_path="data/bm25_index/index.pkl") -> List[Document]:
    """
    Load a list of documents from a Pickle file.
    :param folder_path: Path to the Pickle file
    :return: List of documents
    """

    try:
        with open(folder_path, "rb") as file:
            docs = pickle.load(file)

    except FileNotFoundError:
        raise FileNotFoundError("The file does not exist. Please upload a file")

    return docs


def bm25_add_and_save_documents(new_data: List[Document]):
    """
    Add new documents to the BM25 index and save it.
    """
    try:
        existing_data = bm25_load_local()
        existing_data.extend(new_data)
        bm25_save_local(existing_data)
    except:
        bm25_save_local(new_data)


def bm25_clear(self):
    """
    Clear the BM25 index.
    """
    conn = sqlite3.connect("data/bm25_index.db")
    cursor = conn.cursor()
    cursor.execute("DROP TABLE documents")
    conn.commit()
    conn.close()
