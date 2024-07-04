from typing import Union

from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.messages import AIMessage, HumanMessage


class ConversationHistorySummaryBufferMemory(ConversationSummaryBufferMemory):
    """Buffer with summarizer for storing conversation memory."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def add_user_message(self, message: Union[HumanMessage, str]):
        """Add a user message to the conversation history."""
        self.message_history.add_user_message(message)
        self.chat_memory.add_user_message(message)

    def add_ai_message(self, message: Union[AIMessage, str]):
        """Add an AI message to the conversation history."""
        self.message_history.add_ai_message(message)
        self.chat_memory.add_ai_message(message)

    def add_final_conversation_and_update_summary(
        self,
        user_message: Union[HumanMessage, str],
        ai_message: Union[AIMessage, str],
    ):
        """Add the final user and AI messages to the history and update the summary."""
        self.add_user_message(user_message)
        self.add_ai_message(ai_message)
        self.prune()
