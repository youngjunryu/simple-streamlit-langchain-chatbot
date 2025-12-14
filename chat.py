import streamlit as st
from dotenv import load_dotenv

from chroma_store import ensure_indexed_once
from llm import get_ai_response, PERSIST_DIR, COLLECTION_NAME

load_dotenv()

st.set_page_config(page_title="주택 청약 챗봇")
st.title("주택 청약 챗봇")
st.caption("주택 청약 공고 문서와 관련된 모든 것을 답해드립니다!")


PDF_DIR = "./"  # PDF들이 있는 폴더 (추천: ./pdfs)


# =========================
# Indexing (run once per app process)
# =========================
@st.cache_resource
def ensure_index_once_ui():
    """
    Streamlit은 rerun이 잦으므로 cache_resource로 '프로세스당 1회'만 보장.
    내부에서는 실제 DB/컬렉션 상태를 보고 필요할 때만 인덱싱 수행.
    """
    return ensure_indexed_once(
        pdf_dir=PDF_DIR,
        persist_dir=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
    )


indexed_now = ensure_index_once_ui()


# =========================
# Session state
# =========================
if "session_id" not in st.session_state:
    # 탭/세션 단위로 히스토리 분리
    st.session_state.session_id = "streamlit_session"

if "message_list" not in st.session_state:
    st.session_state.message_list = []


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.subheader("설정")
    st.write(f"PDF_DIR: `{PDF_DIR}`")
    st.write(f"PERSIST_DIR: `{PERSIST_DIR}`")
    st.write(f"COLLECTION: `{COLLECTION_NAME}`")

    if indexed_now:
        st.success("이번 실행에서 인덱싱을 1회 수행했습니다.")
    else:
        st.info("이미 인덱싱 되어 있어 인덱싱을 건너뛰었습니다.")

    if st.button("🧹 대화 초기화"):
        st.session_state.message_list = []
        st.rerun()


# =========================
# Render previous messages
# =========================
for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# =========================
# Input & streaming response
# =========================
if user_question := st.chat_input(
    placeholder="청약 공고에 관련된 궁금한 내용을 말씀해주세요!"
):
    # User message
    st.session_state.message_list.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    # Assistant message (stream)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        acc = ""

        stream = get_ai_response(
            user_message=user_question,
            session_id=st.session_state.session_id,
        )

        for chunk in stream:
            acc += chunk
            placeholder.write(acc)

    st.session_state.message_list.append({"role": "assistant", "content": acc})
