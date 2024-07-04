from typing import List

from langchain.docstore.document import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter


# class TextSplitter:
#     """입력된 모드에 따라 텍스트를 분할하는 클래스입니다."""

#     def __init__(self, mode, embeddings=None):
#         self.mode = mode
#         if self.mode == "SementicChunker":
#             if embeddings is None:
#                 raise ValueError("Embeddings must be provided for semantic splitting.")
#             self.splitter = SemanticChunker(embeddings=embeddings)
#         elif self.mode == "RecursiveCharacterSplitter":
#             self.splitter = RecursiveCharacterTextSplitter(
#                 chunk_size=700, chunk_overlap=100, length_function=len, is_separator_regex=False
#             )
#         else:
#             raise ValueError(
#                 "Invalid mode specified. Choose 'SemanticChunker' or 'RecursiveCharacterSplitter'."
#             )

#     def split_documents(self, text):
#         """주어진 텍스트를 설정된 분할 방식에 따라 분할합니다."""
#         if self.mode ==  "SementicChunker":
#             return self.splitter.split_documents([text])
#         elif self.mode == "RecursiveCharacterSplitter":
#             return self.splitter.split_documents(text)


class TextSplitter:
    def __init__(self, mode=None, embeddings=None):
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "header1"),
                ("##", "header2"),
                ("###", "header3"),
                ("####", "header4"),
                ("#####", "header5"),
                ("######", "header6"),
                ("########", "header8"),
            ]
        )
        self.rc_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400, chunk_overlap=100, length_function=len, is_separator_regex=False
        )

    def split_documents(self, text: List[Document]):
        splitter = self.md_splitter
        md_chunks = splitter.split_text(text.page_content)
        splitted_md_chunks = self.rc_splitter.split_documents(md_chunks)
        for doc in splitted_md_chunks:
            doc.metadata.update({"source": text.metadata["pdf_name"]})
        return splitted_md_chunks
