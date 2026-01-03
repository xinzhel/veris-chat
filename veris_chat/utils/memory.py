
# copy from https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/memory/llama-index-memory-mem0/llama_index/memory/mem0/utils.py
from typing import Any, Dict, List, Optional
from llama_index.core.base.llms.types import ChatMessage, MessageRole

DEFAULT_INTRO_PREFERENCES = "Below are a set of relevant preferences retrieved from potentially several memory sources:"
DEFAULT_OUTRO_PREFERENCES = "This is the end of the retrieved preferences."


def convert_memory_to_system_message(
    response: List[Dict[str, Any]], existing_system_message: ChatMessage = None
) -> ChatMessage:
    """
    Convert Mem0 search results into a system message for LLM context injection.
    
    Takes memories retrieved from Mem0's vector store and formats them into a
    system message that can be prepended to the chat history, giving the LLM
    access to relevant long-term context.
    
    Args:
        response: List of memory dicts from Mem0 search. Each dict contains:
            - "memory": The extracted fact/preference string
            - "categories": Optional list of category tags
            Example:
            [
                {"memory": "User's name is Alice", "categories": ["personal"]},
                {"memory": "User prefers sci-fi movies", "categories": ["preferences"]},
                {"memory": "User is working on the VERIS project"}
            ]
        existing_system_message: Optional existing system message to append to.
            If provided, memories are appended after the existing content.
            
    Returns:
        ChatMessage with role=SYSTEM containing formatted memories.
        
    Example:
        >>> memories = [
        ...     {"memory": "User's name is Alice", "categories": ["personal"]},
        ...     {"memory": "User prefers detailed explanations"}
        ... ]
        >>> msg = convert_memory_to_system_message(memories)
        >>> print(msg.content)
        
        Below are a set of relevant preferences retrieved from potentially several memory sources:
        
         [personal] : User's name is Alice 
        
         User prefers detailed explanations 
        
        This is the end of the retrieved preferences.
        
        # With existing system message:
        >>> existing = ChatMessage(content="You are a helpful assistant.", role=MessageRole.SYSTEM)
        >>> msg = convert_memory_to_system_message(memories, existing)
        >>> print(msg.content)
        You are a helpful assistant.
        
        Below are a set of relevant preferences retrieved from potentially several memory sources:
        ...
    """
    memories = [format_memory_json(memory_json) for memory_json in response]
    formatted_messages = "\n\n" + DEFAULT_INTRO_PREFERENCES + "\n"
    for memory in memories:
        formatted_messages += f"\n {memory} \n\n"
    formatted_messages += DEFAULT_OUTRO_PREFERENCES
    system_message = formatted_messages
    # If existing system message is available
    if existing_system_message is not None:
        system_message = existing_system_message.content.split(
            DEFAULT_INTRO_PREFERENCES
        )[0]
        system_message = system_message + formatted_messages
    return ChatMessage(content=system_message, role=MessageRole.SYSTEM)


def format_memory_json(memory_json: Dict[str, Any]) -> List[str]:
    categories = memory_json.get("categories")
    memory = memory_json.get("memory", "")
    if categories is not None:
        categories_str = ", ".join(categories)
        return f"[{categories_str}] : {memory}"
    return f"{memory}"


def convert_chat_history_to_dict(messages: List[ChatMessage]) -> List[Dict[str, str]]:
    chat_history_dict = []
    for message in messages:
        if (
            message.role in [MessageRole.USER, MessageRole.ASSISTANT]
            and message.content
        ):
            chat_history_dict.append(
                {"role": message.role.value, "content": message.content}
            )
    return chat_history_dict


def convert_messages_to_string(
    messages: List[ChatMessage], input: Optional[str] = None, limit: int = 5
) -> str:
    recent_messages = messages[-limit:]
    formatted_messages = [f"{msg.role.value}: {msg.content}" for msg in recent_messages]
    result = "\n".join(formatted_messages)

    if input:
        result += f"\nuser: {input}"

    return result


# copy from https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/memory/llama-index-memory-mem0/llama_index/memory/mem0/base.py
from typing import Dict, List, Optional, Union, Any
from llama_index.core.memory import BaseMemory, Memory as LlamaIndexMemory
from mem0 import MemoryClient, Memory
from llama_index.core.bridge.pydantic import (
    BaseModel,
    Field,
    ValidationError,
    model_validator,
    SerializeAsAny,
    PrivateAttr,
    ConfigDict,
    model_serializer,
)
from llama_index.core.base.llms.types import ChatMessage, MessageRole


class BaseMem0(BaseMemory):
    """Base class for Mem0."""

    _client: Optional[Union[MemoryClient, Memory]] = PrivateAttr()

    def __init__(
        self, client: Optional[Union[MemoryClient, Memory]] = None, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        if client is not None:
            self._client = client

    def add(
        self, messages: Union[str, List[Dict[str, str]]], **kwargs
    ) -> Optional[Dict[str, Any]]:
        if self._client is None:
            raise ValueError("Client is not initialized")
        if not messages:
            return None
        return self._client.add(messages=messages, **kwargs)

    def search(self, query: str, **kwargs) -> Optional[Dict[str, Any]]:
        if self._client is None:
            raise ValueError("Client is not initialized")
        return self._client.search(query=query, **kwargs)


class Mem0Context(BaseModel):
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None

    @model_validator(mode="after")
    def check_at_least_one_assigned(cls, values):
        if not any(
            getattr(values, field) for field in ["user_id", "agent_id", "run_id"]
        ):
            raise ValueError(
                "At least one of 'user_id', 'agent_id', or 'run_id' must be assigned."
            )
        return values

    def get_context(self) -> Dict[str, Optional[str]]:
        return {key: value for key, value in self.__dict__.items() if value is not None}


class Mem0Memory(BaseMem0):
    """
    Hybrid memory combining LlamaIndex's in-session chat history with Mem0's long-term memory.
    
    This class implements a dual-memory architecture:
    
    1. **primary_memory** (LlamaIndexMemory): Stores the current session's chat history
       in-memory. This is the immediate, ephemeral conversation buffer that holds
       user/assistant message exchanges for the active session. It provides fast
       access to recent messages without external storage lookups.
       
    2. **_client** (Mem0 Memory/MemoryClient): Long-term semantic memory that persists
       across sessions. Mem0 extracts and stores key facts, preferences, and context
       from conversations, enabling recall of relevant information in future sessions.
    
    The `get()` method combines both memory sources:
    - Retrieves chat history from primary_memory (current session messages)
    - Searches Mem0's long-term memory for relevant past context using recent messages as query
    - Injects retrieved long-term memories as a system message prefix
    - Returns the combined message list ready for LLM consumption
    
    This architecture enables:
    - Fast access to current conversation (primary_memory)
    - Semantic search over historical context (Mem0)
    - Session isolation via Mem0Context (user_id/agent_id/run_id)
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    primary_memory: SerializeAsAny[LlamaIndexMemory] = Field(
        description=(
            "In-session chat history buffer (LlamaIndex Memory). "
            "Stores current conversation's user/assistant messages in-memory for fast access. "
            "Used by get() to retrieve recent messages and by put()/set() to store new messages."
        )
    )
    context: Optional[Mem0Context] = None
    search_msg_limit: int = Field(
        default=5,
        description="Limit of chat history messages to use for context in search API",
    )

    def __init__(self, context: Optional[Mem0Context] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        if context is not None:
            self.context = context

    @model_serializer
    def serialize_memory(self) -> Dict[str, Any]:
        # leaving out the two keys since they are causing serialization/deserialization problems
        return {
            "primary_memory": self.primary_memory.model_dump(
                exclude={
                    "memory_blocks_template",
                    "insert_method",
                }
            ),
            "search_msg_limit": self.search_msg_limit,
            "context": self.context.model_dump(),
        }

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "Mem0Memory"

    @classmethod
    def from_defaults(cls, **kwargs: Any) -> "Mem0Memory":
        raise NotImplementedError("Use either from_client or from_config")

    @classmethod
    def from_client(
        cls,
        context: Dict[str, Any],
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        org_id: Optional[str] = None,
        project_id: Optional[str] = None,
        search_msg_limit: int = 5,
        **kwargs: Any,
    ):
        primary_memory = LlamaIndexMemory.from_defaults()

        try:
            context = Mem0Context(**context)
        except ValidationError as e:
            raise ValidationError(f"Context validation error: {e}")

        client = MemoryClient(
            api_key=api_key, host=host, org_id=org_id, project_id=project_id
        )
        return cls(
            primary_memory=primary_memory,
            context=context,
            client=client,
            search_msg_limit=search_msg_limit,
        )

    @classmethod
    def from_config(
        cls,
        context: Dict[str, Any],
        config: Dict[str, Any],
        search_msg_limit: int = 5,
        **kwargs: Any,
    ):
        primary_memory = LlamaIndexMemory.from_defaults()

        try:
            context = Mem0Context(**context)
        except Exception as e:
            raise ValidationError(f"Context validation error: {e}")

        client = Memory.from_config(config_dict=config)
        return cls(
            primary_memory=primary_memory,
            context=context,
            client=client,
            search_msg_limit=search_msg_limit,
        )

    def get(self, input: Optional[str] = None, **kwargs: Any) -> List[ChatMessage]:
        """
        Get chat history augmented with long-term memory context.
        
        This method combines two memory sources to build a context-rich message list:
        
        1. **primary_memory.get()**: Retrieves the current session's chat history
           (user/assistant messages stored in-memory). This gives us the immediate
           conversation context.
           
        2. **Mem0 search**: Uses recent messages (limited by search_msg_limit) plus
           the current input as a query to search Mem0's long-term memory for
           relevant historical context (facts, preferences, prior discussions).
           
        3. **System message injection**: Retrieved long-term memories are formatted
           and prepended as a system message, giving the LLM access to relevant
           historical context alongside the current conversation.
        
        Args:
            input: Optional current user input to include in memory search query.
            **kwargs: Additional arguments passed to primary_memory.get().
            
        Returns:
            List[ChatMessage]: Messages in order [system_message_with_memories, ...chat_history]
            ready for LLM consumption.
            
        Example:
            # User asks a follow-up question
            messages = memory.get(input="What was the site address again?")
            # Returns: [SystemMessage with relevant memories, ...previous chat messages]
        """
        messages = self.primary_memory.get(input=input, **kwargs)
        input = convert_messages_to_string(messages, input, limit=self.search_msg_limit)

        search_results = self.search(query=input, **self.context.get_context())

        if isinstance(self._client, Memory) and self._client.api_version == "v1.1":
            search_results = search_results["results"]

        system_message = convert_memory_to_system_message(search_results)

        # If system message is present
        if len(messages) > 0 and messages[0].role == MessageRole.SYSTEM:
            assert messages[0].content is not None
            system_message = convert_memory_to_system_message(
                response=search_results, existing_system_message=messages[0]
            )
        messages.insert(0, system_message)
        return messages

    def get_all(self) -> List[ChatMessage]:
        """Returns all chat history."""
        return self.primary_memory.get_all()

    def _add_msgs_to_client_memory(self, messages: List[ChatMessage]) -> None:
        """Add new user and assistant messages to client memory."""
        self.add(
            messages=convert_chat_history_to_dict(messages),
            **self.context.get_context(),
        )

    def put(self, message: ChatMessage) -> None:
        """Add message to chat history and client memory."""
        self._add_msgs_to_client_memory([message])
        self.primary_memory.put(message)

    def set(self, messages: List[ChatMessage]) -> None:
        """Set chat history and add new messages to client memory."""
        initial_chat_len = len(self.primary_memory.get_all())
        # Insert only new chat messages
        self._add_msgs_to_client_memory(messages[initial_chat_len:])
        self.primary_memory.set(messages)

    def reset(self) -> None:
        """Only reset chat history."""
        self.primary_memory.reset()