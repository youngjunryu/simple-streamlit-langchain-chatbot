from typing import Dict, Iterable
from operator import itemgetter

from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory

from config import answer_examples

load_dotenv()

# -----------------------------
# Storage config
# -----------------------------
PERSIST_DIR = str("./chroma_openai")
COLLECTION_NAME = str("housing_subscription_system_docs")

# 세션별 대화 히스토리 메모리
_store: Dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]


# -----------------------------
# LLM / Embeddings
# -----------------------------
def get_llm(model: str = "gpt-4o") -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=0)


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model="text-embedding-3-large")


# -----------------------------
# VectorStore / Retriever (Chroma)
# -----------------------------
def get_retriever():
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.9},
    )

    return retriever


def get_dictionary_chain():
    """
    질문 전처리 체인.
    - 내부 사전을 참고해서 질문을 치환하거나
    - 변경할 필요가 없으면 원문을 그대로 반환.
    """
    dictionary = ["사람을 나타내는 표현 -> 거주자"]
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(
        """
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요.
        사전: {dictionary}
        
        질문: {input}
        """.strip()
    ).partial(dictionary=str(dictionary))

    return prompt | llm | StrOutputParser()


def get_history_aware_retriever():
    """
    (input, chat_history) -> LLM (질문 재작성) -> retriever
    """
    llm = get_llm()
    retriever = get_retriever()

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # {"input": str, "chat_history": list[BaseMessage]} -> standalone question -> retriever
    return contextualize_q_prompt | llm | StrOutputParser() | retriever


def get_rag_chain():
    llm = get_llm()

    # Few-shot 예시 프롬프트
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )

    system_prompt = (
        "당신은 주택 청약 전문가입니다. 사용자의 청약 공고에 대한 질문에 답변해주세요. "
        "아래에 제공된 문서를 활용해서 답변해주시고, "
        "답변을 알 수 없다면 모른다고 답변해주세요. "
        "2-3 문장 정도의 짧은 내용의 답변을 원합니다.\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = get_history_aware_retriever()

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    base_chain = (
        RunnableParallel(
            {
                # retriever 결과는 Document[] -> 문자열로 변환해서 context에 주입
                "context": history_aware_retriever | format_docs,
                "input": itemgetter("input"),
                "chat_history": itemgetter("chat_history"),
            }
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    # RunnableWithMessageHistory 로 감싸서,
    # - input_messages_key: "input"
    # - history_messages_key: "chat_history" 에 자동으로 메세지 히스토리를 관리하게 함
    return RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )


# -----------------------------
# Public API: stream response
# -----------------------------
def get_ai_response(user_message: str, session_id: str) -> Iterable[str]:
    """
    Streamlit(chat.py)에서 호출하는 용도.

    - dictionary_chain: 질문 전처리
    - rag_chain: 답변 생성
    - session_id: 사용자 세션별 대화 히스토리 분리
    """
    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain()

    chain = {"input": dictionary_chain} | rag_chain

    return chain.stream(
        {"input": user_message},
        config={
            "configurable": {"session_id": session_id},
        },
    )
