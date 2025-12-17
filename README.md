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
- [chat.py](chat.py): Streamlit 애플리케이션 진입점 — 사용자 인터페이스 및 세션 관리
- [chroma_store.py](chroma_store.py): Chroma 벡터 스토어 관련 유틸리티 및 영구화 로직
- [llm.py](llm.py): LLM(예: OpenAI) 래퍼 및 호출 로직
- [config.py](config.py): 환경 변수와 설정값 정의
- [requirements.txt](requirements.txt): 필요 패키지 목록
- [docs/](docs): 프로젝트 문서용 폴더 — 사용자 매뉴얼, 아키텍처 도면, 연구자료 등 문서

**docs 폴더 안내**
- 보관 위치: 문서는 [docs/](docs) 폴더에 PDF(.pdf) 형식으로 보관하세요. 예: `user-guide.pdf`, `architecture-overview.pdf`.
- 색인 타이밍(언제 인덱싱되는가): 색인은 저장된 벡터 DB(영구화 디렉터리)에 데이터가 없을 때 1회 실행됩니다. 구현은 [chroma_store.py](chroma_store.py) 내의 `ensure_indexed_once()`와 `index_pdfs()`를 따릅니다.
	- 동작 요약:
		- 저장소에 Chroma의 sqlite 파일(예: [chroma_openai/chroma.sqlite3](chroma_openai/chroma.sqlite3))이나 컬렉션에 이미 문서가 있으면(`is_collection_nonempty()`가 True) 색인을 건너뜁니다.
		- 컬렉션이 비어있거나 sqlite 파일이 없으면 `index_pdfs()`가 실행되어 [docs/](docs) (기본) 내부의 `**/*.pdf` 패턴에 매칭되는 모든 PDF를 찾아 색인합니다.
		- PDF는 `PyMuPDF4LLMLoader`로 페이지 단위로 로드되고(`mode="page"`), `MarkdownTextSplitter`로 청킹(chunking)되어 Chroma에 저장됩니다.
- 기본 동작을 변경하거나 수동 재색인이 필요할 때:
	- 강제 재색인: 영구화 디렉터리의 `chroma.sqlite3` 파일을 삭제하거나 컬렉션을 비우면 다음 실행에서 다시 색인됩니다.
	- 수동 색인 호출: `index_pdfs()`를 직접 호출하여 원하는 `pdf_dir`, `persist_dir`, `collection_name`, `chunk_size`, `chunk_overlap` 등을 전달할 수 있습니다. 구현 참고: [chroma_store.py](chroma_store.py).
- 권장 사항:
	- PDF 파일명은 목적이 드러나도록 지정하세요(예: `user-guide.pdf`).
	- 색인 범위(하위 폴더 포함)는 기본 glob(`**/*.pdf`)에 따릅니다. 특정 하위폴더만 색인하려면 glob 패턴을 조정하세요.
	- 색인 결과를 RAG에서 바로 쓰려면 `persist_dir`(예: [chroma_openai/](chroma_openai))를 애플리케이션에서 사용하도록 설정하세요.

**설정 참고**
- 로컬에 문서를 임포트하거나 임베딩을 다시 생성하려면 `chroma_store.py`를 참고하세요.
- 다른 LLM 공급자를 사용하려면 `llm.py`의 래퍼를 수정하거나 확장하면 됩니다.

**기여 방법**
- 문제 제기(issues) 및 풀 리퀘스트 환영합니다. 변경 전 간단한 이슈로 의도를 공유해 주세요.

필요하면 README에 더 자세한 사용 예제(데이터 인덱싱 방법, 커스텀 파이프라인 예시)를 추가해 드리겠습니다.

