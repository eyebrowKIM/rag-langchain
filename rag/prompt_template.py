from langchain_core.prompts import PromptTemplate
from langchain.output_parsers.regex import RegexParser


def Question_Prompt():
    """
    Stuff Chain에서 쓰이는 프롬프트
    """

    template = PromptTemplate(
        template="""
        당신은 질의응답 임무를 수행하는 비서입니다.
        다음의 문맥을 이용하여 제 질문에 답해주세요.
        질문과 관계 없는 문맥이 존재한다면 무시해주세요.
        답을 모르면 모른다고 말하세요. 답을 지어내려고 하지 마세요. 
        문서에 존재하는 내용에 기반하여 답해주세요.
        질문과 관련된 문맥이 존재하지 않는다면, 본인의 생각으로 답해주세요. 그러나 '제공된 정보에서 찾을 수 없음'이라고 명시해주세요.
        답은 한국어로 작성해주세요.
        
        질문:
        {question}
        
        문맥:
        {context}
        
        """,
        input_variables=["question", "context"],
    )

    return template


def Map_Prompt():
    """
    Map Reduce Chain에서 Map 과정에 쓰이는 프롬프트.
    추출된 문서에서 쿼리에 관련된 부분만 반환하는 과정.
    """

    template = PromptTemplate(
        template="""
        질문: {question}
        =========
        참고할 내용: {context}
        =========
        답변 지침:
        - 문서에서 질문과 관련된 정보만을 기반으로 답변을 작성합니다.
        - 각 정보를 요약하고 중복을 피하세요.
        - 관련된 핵심 정보만 포함되도록 하세요.
        =========
        최종 답안:
        """,
        input_variables=["question", "context"],
    )

    return template


def Reduce_Prompt():
    """
    Map Reduce Chain에서 Reduce 과정에 사용되는 프롬프트.
    Map에서 추출된 문서들을 종합하여 최종 답변을 만드는 과정.
    """
    template = PromptTemplate(
        template="""
        답변 지침:
        문서에서 질문과 관련된 정보만을 바탕으로 답변합니다.
        ==========
        참고 내용: {summaries}
        ==========
        질문: {question}
        ==========
        최종 답변:
        """,
        input_variables=["question", "summaries"],
    )

    return template


def Refine_QA_Prompt():
    """
    Refine chain에서 Question에 사용되는 prompt
    """
    template = PromptTemplate(
        template="""
        문맥 정보는 아래와 같습니다. \n
        질문과 관련된 문맥의 내용을 요약해주세요.\n
        ------------\n
        문맥 : {context_str}\n
        ------------\n
        질문 : {question}\n
        """,
        input_variables=["context_str", "question"],
    )

    return template


def Refine_Prompt():
    """
    Refine chain의 Refine 과정에서 쓰이는 프롬프트
    """
    template = PromptTemplate(
        template="""
        질문: {question}
        당신의 역할은 최종적인 답안을 생성하는 것입니다.  
        지금까지 요약된 내용을 제공해드리도록 하겠습니다
        요약된 내용: {existing_answer}
        (only if needed) 아래에 추가적인 문맥을 이용해주세요.
        ------------
        {context_str}
        ------------
        새로운 문맥을 고려하여 원래 답변을 더 잘 대답하도록 요약 내용을 수정해주세요.
        추가적인 문맥이 유용하지 않다면, 원래의 요약된 내용을 그대로 출력해주세요.
        """,
        input_variables=["question", "existing_answer", "context_str"],
    )

    return template


def Map_Rerank_Prompt():
    """
    Map rerank chain에서 score를 매기기 위한 프롬프트
    """
    template = PromptTemplate(
        template="""
        다음의 문맥들을 사용해서 제 질문에 대답해주세요. 답을 모르면 모른다고 말해주세요. 답을 지어내려고 하지 마세요.
        
        답을 제공할 뿐만 아니라, 사용자의 질문에 대한 답변의 완전성을 나타내는 점수를 반환해주세요. 이는 다음 형식으로 되어 있어야 합니다:
        
        질문 : [질문 내용]
        도움이 되는 답변 : [답변 내용]
        점수 : [0과 100 사이의 점수]
        
        점수를 결정하는 방법:
        - 높은 점수가 더 좋은 답변을 의미합니다.
        - 사용자의 질문에 완전히 응답하며, 충분한 세부 정보를 제공합니다.
        - 문맥에 기반하여 답변을 제공하지 못하는 경우, 점수는 0이어야 하며 "이 문서는 질문에 대한 답변을 제공하지 않습니다"라는 메시지를 반환해야 합니다.
        - 너무 자신감을 가지지 마세요!
        
        예시 #1
        
        문맥:
        ---------
        사과는 빨간색입니다
        ---------
        질문: 사과는 어떤 색인가요?
        도움이 되는 답변: 빨간색
        점수: 100
        
        
        예시 #2
        
        문맥:
        ---------
        밤이었고, 목격자는 안경을 잊어버렸습니다. 그는 스포츠카인지 SUV인지 확신하지 못했습니다.
        ---------
        질문: 차의 종류는 무엇인가요?
        도움이 되는 답변: 스포츠카 또는 SUV
        점수: 60
        
        
        예시 #3
        
        문맥:
        ---------
        배는 빨간색 또는 주황색입니다
        ---------
        질문: 사과는 어떤 색인가요?
        도움이 되는 답변: 이 문서는 질문에 대한 답변을 제공하지 않습니다
        점수: 0
        
        시작!
        
        문맥:
        ---------
        {context}
        ---------
        질문: {question}
        도움이 되는 답변:""",
        input_variables=["context", "question"],
        output_parser=RegexParser(
            regex=r"(.*?)\n\s*점수:\s*(.*)",
            output_keys=["answer", "점수"],
        ),
    )

    return template


def Condense_Question_Prompt():
    """
    Chat history를 이용하여 질문을 요약하는 프롬프트
    """

    template = PromptTemplate(
        template="""
    아래의 채팅 기록과 사용자의 질문을 참고하여, 사용자가 묻고자 하는 핵심 질문을 하나의 간결한 질문으로 요약하세요.
    요약된 질문은 물음표로 끝나야 합니다.
    질문을 모를 경우 사용자의 질문을 그대로 출력하세요.
    ---

    채팅 기록:

    {chat_history}

    사용자의 질문: {question}

    질문: 
    ---

        """,
        input_variables=["chat_history", "question"],
    )

    return template


def Filter_prompt() -> PromptTemplate:
    """
    Filtering chain에서 사용되는 프롬프트
    """
    template = PromptTemplate(
        template="""
    
    다음의 질문과 메타데이터를 참고해주세요.
    메타데이터와 질문이 관계가 있다면, '있음'이라고 작성해주세요.
    메타데이터와 질문이 관계가 없다면, '없음'이라고 작성해주세요.
    '있음' 혹은 '없음'만 출력하세요
    
    ex.
    1)
    질문:
    사과가 무슨 색인가요?
    메타데이터:
    사과는 빨간색입니다.
    답변:
    있음
    
    2)질문:
    사과는 무슨 색인가요?
    메타데이터:
    바나나는 노란색입니다.
    답변:
    없음
    
    ---
    질문:
    {question}
    ---
    메타데이터:
    {document}
    ---
    답변 :
    """,
        input_variables=["question", "document"],
    )

    return template


QUESTION_PROMPT = Question_Prompt()
MAP_PROMPT = Map_Prompt()
REDUCE_PROMPT = Reduce_Prompt()
REFINE_QA_PROMPT = Refine_QA_Prompt()
REFINE_PROMPT = Refine_Prompt()
MAP_RERANK_PROMPT = Map_Rerank_Prompt()
CONDENSE_QUESTION_PROMPT = Condense_Question_Prompt()
FILTER_PROMPT = Filter_prompt()
