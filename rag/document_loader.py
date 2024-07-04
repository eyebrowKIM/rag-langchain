import logging
import os

from langchain.docstore.document import Document
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    SlackDirectoryLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
    UnstructuredURLLoader,
    UnstructuredMarkdownLoader,
)
import pymupdf4llm

# 로깅 설정
logger = logging.getLogger(__name__)


class DataLoader:
    def _load_and_split(self, file_dir):
        raise NotImplementedError("Subclass must implement abstract method")


class FileDataLoader(DataLoader):
    def load_data(uploaded):
        # Supported file loaders mapping
        loaders = {
            # ".pdf": PyPDFLoader,
            ".pdf": pymupdf4llm.to_markdown,
            ".docx": Docx2txtLoader,
            ".pptx": UnstructuredPowerPointLoader,
            ".txt": TextLoader,
            ".csv": CSVLoader,
            ".md": UnstructuredMarkdownLoader,
        }

        save_dir = "./files"
        os.makedirs(save_dir, exist_ok=True)

        file_path = os.path.join(save_dir, uploaded.name)
        with open(file_path, "wb") as f:
            f.write(uploaded.getvalue())

        _, ext = os.path.splitext(file_path)

        loader_class = loaders.get(ext)
        if loader_class is not None:
            if ext == ".pdf":
                loaded = loader_class(file_path)
                documents = Document(page_content=loaded, metadata={"pdf_name": uploaded.name})
                return documents
            else:
                loader = loader_class(file_path)
                documents = loader.load()
                return documents
        else:
            logger.error(f"Unsupported file type: {file_path}")


class URLDataLoader(DataLoader):
    def load_data(uploaded):
        # Determine loader based on the content of uploaded
        if "slack" in uploaded:
            loader = SlackDirectoryLoader(uploaded)
        elif "http" in uploaded:
            loader = UnstructuredURLLoader(uploaded)
        else:
            raise ValueError("Unsupported URL type")

        # Load and split documents using the selected loader
        documents = loader.load()
        pdf_name = uploaded.name
        return documents
