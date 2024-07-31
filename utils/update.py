# utils.py
import csv
from langchain.schema import HumanMessage, SystemMessage, Document

# 다양한 PDF loader들
from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader

# CSVloader
from langchain_community.document_loaders.csv_loader import CSVLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from tqdm import tqdm
import re

def split_document(doc: Document, chunk_size) -> list[Document]:
    """LangchainDocument 객체를 받아 적절히 청크로 나눕니다."""
    filename = doc.metadata.get("filename", "")
    if filename.endswith(".csv"):
        return [doc]  # CSV 파일은 이미 로드 시에 청크로 나뉨
    else:
        content = doc.page_content
        chunks = [content[i : i + chunk_size] for i in range(0, len(content), chunk_size)]
        return [Document(page_content=chunk) for chunk in chunks]


def convert_file_to_documents(vector_store, file, force=True, chunk_size=500):
    """파일을 읽어 Langchain의 Document 객체로 변환"""
    SIMILARITY_THRESHOLD=0.3
    

    pattern = re.compile(r'\[(.*?)\]')
    matches = pattern.findall(file.name.split('\\')[-1])
    
    
    documents = []
    
    if file.name.endswith(".txt"):
        content = file.read().decode("utf-8")
        results = vector_store.similarity_search_with_score(content, k=1)
        # print(f'유사도 검사 중...results : {results}')
        if results and results[0][1] <= SIMILARITY_THRESHOLD:
            print(f"기존 DB에 유사한 청크가 있음으로 판단되어 추가되지 않음 - {results[0][1]}")
        else:
            documents = [Document(metadata={"source": file.path, "page": 0}, page_content=content)]
    else:
        # 파일에 맞는 loader
        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(file.name)         
        elif file.name.endswith(".csv"):
            loader = CSVLoader(file_path=file.name)

        temp_documents = loader.load()  # loader를 통해
        for i, row in enumerate(tqdm(temp_documents, total=len(temp_documents))):
            content = row.page_content
            if not force:
                results = vector_store.similarity_search_with_score(content, k=1)
                # print(f'유사도 검사 중...results : {results}')
                if results and (results[0][1] <= SIMILARITY_THRESHOLD):
                    print(f"기존 DB에 유사한 청크가 있음으로 판단되어 추가되지 않음  - {results[0][1]}")
                    continue

            metadata = {"source": row.metadata["source"].split('\\')[-1], "page": row.metadata["page"], 'year':int(matches[-1].split('.')[0]), 'category':matches[0]}
            
            chunks = [content[i : i + chunk_size] for i in range(0, len(content), chunk_size)]
            doc = [Document(metadata=metadata, page_content=chunk) for chunk in chunks]
            documents.extend(doc)

    return documents
