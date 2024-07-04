import streamlit as st
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from .memory import ConversationHistorySummaryBufferMemory


class ConversationChain:
    def __init__(self, llm, retriever, QA_PROMPT):
        self.llm = llm
        self.retriever = retriever
        self.QA_PROMPT = QA_PROMPT

    def get_chain(self, chain_type):
        """
        Returns a conversation chain based on the specified chain type.

        Parameters:
        - chain_type (str): The type of chain to retrieve, e.g., 'map_reduce', 'refine', 'stuff'.

        Returns:
        - ConversationalRetrievalChain: Configured chain instance.
        """
        chain_map = {
            "map_reduce": self.map_reduce_chain,
            "refine": self.refine_chain,
            "stuff": self.stuff_chain,
            "map_rerank": self.map_rerank_chain,
        }

        return chain_map[chain_type]()

    def stuff_chain(self):
        """
        Creates a conversation chain using the provided QA prompt, retriever, and language model (llm).

        Parameters:
        - QA_PROMPT (str): The prompt used for question-answering.
        - retriever: The retriever component used for information retrieval.
        - llm: The language model used for generating responses.

        Returns:
        - ConversationalRetrievalChain: An instance of ConversationalRetrievalChain configured with the given parameters.
        """
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            chain_type="stuff",
            retriever=st.session_state.retriever,
            memory=ConversationBufferMemory(
                memory_key="chat_history", return_messages=True, output_key="answer"
            ),
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose=False,
            combine_docs_chain_kwargs={"prompt": self.QA_PROMPT},
        )

        return conversation_chain

    def map_reduce_chain(self):
        """
        Creates a map_reduce type conversation chain.
        """
        from .prompt_template import (
            MAP_PROMPT,
            REDUCE_PROMPT,
            CONDENSE_QUESTION_PROMPT,
        )

        map_prompt = MAP_PROMPT
        reduce_prompt = REDUCE_PROMPT
        condense_question_prompt = CONDENSE_QUESTION_PROMPT
        # TODO: 요약모델 구분(2B정도 모델로 요약)
        #       reduce_llm_instance = LLM_HuggingFace(model_name="daekeun-ml/phi-2-ko-v0.1")
        #       reduce_llm = reduce_llm_instance.get_llm()

        combine_docs_chain_kwargs = {
            "question_prompt": map_prompt,
            "combine_prompt": reduce_prompt,
            "combine_document_variable_name": "summaries",
            "map_reduce_document_variable_name": "context",
            "collapse_prompt": reduce_prompt,
            # "reduce_llm" : reduce_llm,
            # "collapse_llm" : reduce_llm,
            "token_max": 3000,
        }

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            condense_question_prompt=condense_question_prompt,
            retriever=self.retriever,
            chain_type="map_reduce",
            memory=ConversationHistorySummaryBufferMemory(
                llm=self.llm, max_token_limit=1024, memory_key="chat_history", output_key="answer"
            ),
            verbose=False,
            return_source_documents=True,
            get_chat_history=lambda h: h,
            combine_docs_chain_kwargs=combine_docs_chain_kwargs,
        )

        return conversation_chain

    def refine_chain(self):
        """
        Creates a refine type conversation chain.
        """
        from .prompt_template import REFINE_PROMPT, CONDENSE_QUESTION_PROMPT

        initial_prompt = REFINE_PROMPT
        refine_prompt = REFINE_PROMPT
        condense_question_prompt = CONDENSE_QUESTION_PROMPT

        combine_docs_chain_kwargs = {
            "question_prompt": initial_prompt,
            "refine_prompt": refine_prompt,
            "document_variable_name": "context_str",
            "initial_response_name": "existing_answer",
        }

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            condense_question_prompt=condense_question_prompt,
            retriever=self.retriever,
            chain_type="refine",
            memory=ConversationHistorySummaryBufferMemory(
                llm=self.llm, max_token_limit=1024, memory_key="chat_history", output_key="answer"
            ),
            verbose=False,
            return_source_documents=True,
            get_chat_history=lambda h: h,
            combine_docs_chain_kwargs=combine_docs_chain_kwargs,
        )

        return conversation_chain

    def map_rerank_chain(self):
        """
        Creates a map_rerank type conversation chain.
        This function is not yet implemented and will raise an error if called.
        """

        from .prompt_template import CONDENSE_QUESTION_PROMPT, MAP_RERANK_PROMPT

        condense_question_prompt = CONDENSE_QUESTION_PROMPT
        map_rerank_prompt = MAP_RERANK_PROMPT

        combine_docs_chain_kwargs = {
            "prompt": map_rerank_prompt,
            "document_variable_name": "context",
            "rank_key": "점수",
            "answer_key": "answer",
        }

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            condense_question_prompt=condense_question_prompt,
            retriever=self.retriever,
            chain_type="map_rerank",
            memory=ConversationHistorySummaryBufferMemory(
                llm=self.llm, max_token_limit=1024, memory_key="chat_history", output_key="answer"
            ),
            verbose=False,
            return_source_documents=True,
            get_chat_history=lambda h: h,
            combine_docs_chain_kwargs=combine_docs_chain_kwargs,
        )

        return conversation_chain
