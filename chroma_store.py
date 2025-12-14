import traceback
from dotenv import load_dotenv
from pathlib import Path
from typing import List

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import MarkdownTextSplitter

load_dotenv()


# -----------------------------
# 기본 설정 유틸
# -----------------------------
def chroma_exists(persist_dir: str) -> bool:
    return (Path(persist_dir) / "chroma.sqlite3").exists()


# -----------------------------
# Embeddings / VectorStore
# -----------------------------
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model="text-embedding-3-large")


def get_chroma(persist_dir: str, collection_name: str) -> Chroma:
    # 동일 persist_dir + collection_name 조합이면 기존 DB를 로드하고,  없으면 새로 생성됨.
    return Chroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=get_embeddings(),
    )


def is_collection_nonempty(persist_dir: str, collection_name: str) -> bool:
    """
    sqlite 파일은 있어도 컬렉션이 비었을 수 있어서,
    가능하면 collection count로 한 번 더 확인.
    (내부 속성 접근이므로 try/except로 안전 처리)
    """
    if not chroma_exists(persist_dir):
        return False

    try:
        vs = get_chroma(persist_dir, collection_name)
        # langchain_chroma 내부의 chromadb collection
        return getattr(vs, "_collection").count() > 0
    except Exception:
        # 내부 구조가 바뀌거나 접근이 막히면,
        # sqlite 존재만으로 '있다'고 판단(보수적)
        return True


# -----------------------------
# 2) Indexing Helpers
# PDF -> Markdown -> Split
# -----------------------------


def load_pdf_as_markdown_docs(pdf_path: str) -> List[Document]:
    # PDF를 Markdown 기반 Document로 로드. mode="page"로 페이지 단위 Documen 생성.
    loader = PyMuPDF4LLMLoader(pdf_path, mode="page")
    docs = loader.load()

    # 출처 메타데이터 보강
    for d in docs:
        d.metadata.setdefault("source", str(pdf_path))

    return docs


def split_markdown_docs(
    docs: List[Document],
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> List[Document]:
    splitter = MarkdownTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


def index_documents(
    persist_dir: str,
    docs: List[Document],
    collection_name: str,
) -> int:
    """
    Document 리스트를 Chroma에 저장(upsert/add).
    반환: 저장한 문서(청크) 개수
    """
    vs = get_chroma(persist_dir, collection_name=collection_name)
    vs.add_documents(docs)
    return len(docs)


# -----------------------------
# Indexing
# -----------------------------
def index_pdfs(
    pdf_dir: str,
    persist_dir: str,
    collection_name: str,
    glob_pattern: str = "**/*.pdf",
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> int:
    """
    폴더 내 PDF들을:
      PDF -> Markdown docs -> (선택) Markdown 청킹 -> Chroma 저장
    반환: 저장된 총 청크 수
    """
    pdf_paths = sorted(Path(pdf_dir).glob(glob_pattern))
    if not pdf_paths:
        print(f"[index] No PDFs found: {pdf_dir}")
        return 0

    all_chunks: List[Document] = []
    for p in pdf_paths:
        raw_docs = load_pdf_as_markdown_docs(str(p))
        chunks = split_markdown_docs(
            raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        all_chunks.extend(chunks)

    return index_documents(
        persist_dir=persist_dir, collection_name=collection_name, docs=all_chunks
    )


def ensure_indexed_once(
    pdf_dir: str,
    persist_dir: str,
    collection_name: str,
) -> bool:
    """
    이미 인덱싱 되어있으면 스킵.
    안되어있으면 1회 인덱싱 수행.

    반환:
      - True  : 이번 실행에서 인덱싱 수행함
      - False : 이미 되어있어서 스킵함
    """
    if is_collection_nonempty(persist_dir, collection_name):
        return False

    index_pdfs(
        pdf_dir=pdf_dir,
        persist_dir=persist_dir,
        collection_name=collection_name,
    )
    return True


# -----------------------------
# (Retrieval)
# -----------------------------
# def search_mmr(
#     query: str,
#     persist_dir: str,
#     collection_name: str,
#     k: int = 4,
#     fetch_k: int = 20,
#     lambda_mult: float = 0.5,
# ) -> List[Document]:
#     vs = get_chroma(persist_dir, collection_name)
#     return vs.max_marginal_relevance_search(
#         query,
#         k=k,
#         fetch_k=fetch_k,
#         lambda_mult=lambda_mult,
#     )
