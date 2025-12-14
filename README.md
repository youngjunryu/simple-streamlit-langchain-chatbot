# simple-streamlit-langchain-chatbot

간단한 Streamlit 기반 RAG(실시간 검색 기반 생성) 챗봇 예제입니다. 이 저장소는 LangChain, Chroma(로컬 벡터 DB), OpenAI(또는 호환되는 LLM)를 사용해 문서 기반 대화형 에이전트를 빠르게 시도해볼 수 있도록 구성되어 있습니다.

**주요 기능**
- **RAG 기반 QA**: 저장된 문서에서 관련 문맥을 찾아 LLM에 전달하여 답변 생성
- **로컬 벡터 스토어**: `chroma`를 사용한 임베딩 저장 및 검색
- **간단한 Streamlit UI**: `chat.py`로 쉽게 웹에서 대화 실행

**빠른 시작**

1. Python 3.10 이상을 설치합니다.
2. 가상환경을 만들고 의존성을 설치합니다:

```zsh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


3. 필수 환경 변수 설정:
- **`OPENAI_API_KEY`**: OpenAI(또는 호환되는 제공자)의 API 키를 환경 변수로 설정하세요.
- 추가 설정은 `config.py` 파일을 확인하세요.

환경 변수는 로컬에서 `.env` 파일로 관리하는 방식을 권장합니다. 이 저장소는 `.env` 파일을 사용하도록 설계되어 있으며, 루트에 `.env` 파일을 만들고 다음과 같은 형식으로 값을 넣어주세요 (실제 키는 절대 공개하지 마세요):

```dotenv
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

안전 권장사항:
- **`.env` 파일을 저장소에 커밋하지 마세요.** 루트의 `.gitignore`에 이미 `.env`가 포함되어 있는지 확인하세요.
- 만약 실수로 키를 공개 리포지토리에 커밋했다면 즉시 해당 키를 폐기(revoke)하고 새 키로 교체하세요.
- 공개된 키를 Git 히스토리에서 삭제하려면 `git rm --cached .env` 후 커밋하고, 더 강력한 이력 제거가 필요하면 `git filter-repo` 또는 BFG를 사용하세요.

예시: `.env`를 Git에서 제거하고 새 커밋을 만든 후 푸시하는 기본 절차:

```zsh
git rm --cached .env
git commit -m "Remove .env from repository"
git push
```

더 강력한 히스토리 제거가 필요하면 별도 안내를 요청하세요.


4. Streamlit 앱 실행:

```zsh
streamlit run chat.py
```

**프로젝트 구조**
- `chat.py`: Streamlit 애플리케이션 진입점 — 사용자 인터페이스 및 세션 관리
- `chroma_store.py`: Chroma 벡터 스토어 관련 유틸리티 및 영구화 로직
- `llm.py`: LLM(예: OpenAI) 래퍼 및 호출 로직
- `config.py`: 환경 변수와 설정값 정의
- `requirements.txt`: 필요 패키지 목록

**설정 참고**
- 로컬에 문서를 임포트하거나 임베딩을 다시 생성하려면 `chroma_store.py`를 참고하세요.
- 다른 LLM 공급자를 사용하려면 `llm.py`의 래퍼를 수정하거나 확장하면 됩니다.

**기여 방법**
- 문제 제기(issues) 및 풀 리퀘스트 환영합니다. 변경 전 간단한 이슈로 의도를 공유해 주세요.

필요하면 README에 더 자세한 사용 예제(데이터 인덱싱 방법, 커스텀 파이프라인 예시)를 추가해 드리겠습니다.

