# RAG langhchain

### .env 파일 내의 LANGCHAIN_API_KEY 본인 키로 바꾸기. (Langsmith 홈페이지에서 발급)

`streamlit run app.py`

### Evaluation 방법
- eval/data.json 파일에 Q&A pair 넣기.(A : ground truth)

- `python eval/data_create.py`
langsmith에 데이터셋 업로드

- `python eval/rag_eval.py`


DONE
1. vectorDB 데이터 저장 후, 로드해서 사용 가능
2. 사용자 피드백 가능 (responses.jsonl에 저장)

TODO List
1. LLM 모델, 임베딩 모델 실험
2. LLaVA 기능추가
3. RLHF, DPO 기능
4. 디스플레이 오류 해결

ISSUE List
1. 가끔씩 docker timeout
