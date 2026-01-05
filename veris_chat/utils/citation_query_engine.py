import time
from typing import Any, Dict, List, Optional, Sequence

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.indices.base import BaseGPTIndex
from llama_index.core.llms.llm import LLM
from llama_index.core.node_parser import SentenceSplitter, TextSplitter
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.base import BasePromptTemplate
from llama_index.core.prompts.mixin import PromptMixinType
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.response_synthesizers import (
    BaseSynthesizer,
    ResponseMode,
    get_response_synthesizer,
)
from llama_index.core.response_synthesizers.compact_and_refine import CompactAndRefine
from llama_index.core.response_synthesizers.generation import Generation
from llama_index.core.schema import (
    MetadataMode,
    NodeWithScore,
    QueryBundle,
    TextNode,
)
from llama_index.core.settings import Settings
from llama_index.core.prompts.prompt_utils import get_biggest_prompt

# =============================================================================
# IN-TEXT CITATION WITH MARKDOWN LINKS
# =============================================================================
# Instead of numbered citations like [1], [2] that require a separate reference
# list, we use inline markdown links that can be directly clicked in the frontend.
#
# Format: [Document Name (p.X)](url)
# Example: "The site is classified as priority [OL000071228.pdf (p.2)](https://...)"
#
# This approach:
# 1. Provides immediate context (document name + page number)
# 2. Allows direct navigation via clickable links
# 3. Works with standard markdown renderers (React-Markdown, etc.)
# 4. No need for a separate reference list at the end
# =============================================================================

CITATION_QA_TEMPLATE = PromptTemplate(
    "Please provide an answer based on the provided sources and conversation context. "
    "When referencing information from a source document, cite it using the EXACT markdown link format provided with each source. "
    "For document-based answers, include at least one source citation. "
    "Only cite a source when you are explicitly referencing it. "
    "If the query can be answered from conversation context (e.g., user preferences, previous discussions), "
    "you may answer without document citations. "
    "If neither sources nor conversation context are helpful, indicate that. "
    "\n\n"
    "IMPORTANT: Use the citation link EXACTLY as provided (do not modify the URL or format).\n"
    "\n"
    "For example:\n"
    "Source: [weather_report.pdf (p.1)](https://example.com/weather.pdf)\n"
    "The sky is red in the evening and blue in the morning.\n\n"
    "Source: [science_facts.pdf (p.3)](https://example.com/science.pdf)\n"
    "Water is wet when the sky is red.\n\n"
    "Query: When is water wet?\n"
    "Answer: Water will be wet when the sky is red [science_facts.pdf (p.3)](https://example.com/science.pdf), "
    "which occurs in the evening [weather_report.pdf (p.1)](https://example.com/weather.pdf).\n"
    "\n"
    "Now it's your turn. Below are the sources of information:\n"
    "------\n"
    "{context_str}"
    "\n------\n"
    "Query: {query_str}\n"
    "Answer: "
)

CITATION_REFINE_TEMPLATE = PromptTemplate(
    "Please provide an answer based on the provided sources and conversation context. "
    "When referencing information from a source document, cite it using the EXACT markdown link format provided with each source. "
    "For document-based answers, include at least one source citation. "
    "Only cite a source when you are explicitly referencing it. "
    "If the query can be answered from conversation context, you may answer without document citations. "
    "If neither sources nor conversation context are helpful, indicate that. "
    "\n\n"
    "IMPORTANT: Use the citation link EXACTLY as provided (do not modify the URL or format).\n"
    "\n"
    "We have provided an existing answer: {existing_answer}\n"
    "Below are additional sources of information. "
    "Use them to refine the existing answer. "
    "If the provided sources are not helpful, you will repeat the existing answer."
    "\nBegin refining!\n"
    "------\n"
    "{context_msg}"
    "\n------\n"
    "Query: {query_str}\n"
    "Answer: "
)

DEFAULT_CITATION_CHUNK_SIZE = 512
DEFAULT_CITATION_CHUNK_OVERLAP = 20


class CitationQueryEngine(BaseQueryEngine):
    """
    Citation query engine.

    Args:
        retriever (BaseRetriever): A retriever object.
        response_synthesizer (Optional[BaseSynthesizer]):
            A BaseSynthesizer object.
        citation_chunk_size (int):
            Size of citation chunks, default=512. Useful for controlling
            granularity of sources.
        citation_chunk_overlap (int): Overlap of citation nodes, default=20.
        text_splitter (Optional[TextSplitter]):
            A text splitter for creating citation source nodes. Default is
            a SentenceSplitter.
        callback_manager (Optional[CallbackManager]): A callback manager.
        metadata_mode (MetadataMode): A MetadataMode object that controls how
            metadata is included in the citation prompt.

    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: Optional[LLM] = None,
        response_synthesizer: Optional[BaseSynthesizer] = None,
        citation_chunk_size: int = DEFAULT_CITATION_CHUNK_SIZE,
        citation_chunk_overlap: int = DEFAULT_CITATION_CHUNK_OVERLAP,
        text_splitter: Optional[TextSplitter] = None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        callback_manager: Optional[CallbackManager] = None,
        metadata_mode: MetadataMode = MetadataMode.NONE,
    ) -> None:
        self.text_splitter = text_splitter or SentenceSplitter(
            chunk_size=citation_chunk_size, chunk_overlap=citation_chunk_overlap
        )
        self._retriever = retriever

        callback_manager = callback_manager or Settings.callback_manager
        llm = llm or Settings.llm

        # =====================================================================
        # WHY USE ResponseSynthesizer INSTEAD OF DIRECT LLM CALLS?
        # =====================================================================
        # "Why not just do: prompt = template.format(...); llm.complete(prompt)?"
        #
        # ResponseSynthesizer provides several advantages over raw LLM calls:
        #
        # 1. CONTEXT WINDOW MANAGEMENT
        #    - What if retrieved nodes exceed the LLM's context window (e.g., 4K tokens)?
        #    - ResponseSynthesizer handles this via different ResponseModes:
        #      * COMPACT: Stuffs as many nodes as fit, truncates the rest
        #      * REFINE: Iteratively refines answer by processing nodes in batches
        #      * TREE_SUMMARIZE: Hierarchically summarizes nodes before answering
        #      * SIMPLE_SUMMARIZE: Truncates context to fit
        #
        # 2. MULTI-STEP REFINEMENT (refine_template)
        #    - For large context, REFINE mode calls LLM multiple times:
        #      Step 1: Answer with first batch of nodes
        #      Step 2: Refine answer with next batch (using refine_template)
        #      Step N: Continue until all nodes processed
        #    - This is why we have both text_qa_template AND refine_template
        #
        # 3. STRUCTURED RESPONSE HANDLING
        #    - Returns Response object with:
        #      * response.response: The text answer
        #      * response.source_nodes: Which nodes were used (for citations!)
        #      * response.metadata: Additional info (token counts, etc.)
        #    - Raw llm.complete() just returns a string
        #
        # 4. STREAMING SUPPORT
        #    - streaming=True enables token-by-token streaming
        #    - Handles the complexity of streaming + source node tracking
        #
        # 5. ASYNC SUPPORT
        #    - use_async=True enables async LLM calls
        #    - Properly handles async context and callbacks
        #
        # 6. CALLBACK INTEGRATION
        #    - Automatically emits LLM events for observability
        #    - Tracks token usage, latency, prompts, completions
        #
        # COMPARISON:
        # -----------
        # Direct LLM approach (simple but limited):
        #   prompt = f"Context: {nodes}\nQuery: {query}\nAnswer:"
        #   response = llm.complete(prompt)  # Just a string, no source tracking
        #
        # ResponseSynthesizer approach (production-ready):
        #   response = synthesizer.synthesize(query, nodes)
        #   # response.response = answer text
        #   # response.source_nodes = nodes used (for citations)
        #   # Handles context overflow, streaming, async, callbacks
        #
        # ResponseMode options (each maps to a specific class):
        # - COMPACT -> CompactAndRefine: Fit as much context as possible, single LLM call
        # - REFINE -> Refine: Multiple LLM calls, iteratively refine answer
        # - TREE_SUMMARIZE -> TreeSummarize: Build summary tree, then answer
        # - SIMPLE_SUMMARIZE -> SimpleSummarize: Truncate and summarize
        # - NO_TEXT -> NoText: Return nodes only, no LLM call
        # - ACCUMULATE -> Accumulate: Separate answer per node, then combine
        # - COMPACT_ACCUMULATE -> CompactAndAccumulate: Compact + accumulate
        # - GENERATION -> Generation: Simple generation without context
        # - CONTEXT_ONLY -> ContextOnly: Return context only, no LLM call
        #
        # EXPLICIT DEFINITION (instead of get_response_synthesizer):
        # -----------------------------------------------------------
        # Below we explicitly instantiate CompactAndRefine (ResponseMode.COMPACT)
        # to make the code more transparent and easier to understand.
        #
        # get_response_synthesizer() is just a factory function that does:
        #   if response_mode == ResponseMode.COMPACT:
        #       return CompactAndRefine(llm=llm, callback_manager=..., ...)
        #   elif response_mode == ResponseMode.REFINE:
        #       return Refine(llm=llm, ...)
        #   ...
        # =====================================================================
        if response_synthesizer is not None:
            self._response_synthesizer = response_synthesizer
        else:
            # Explicitly create CompactAndRefine (equivalent to ResponseMode.COMPACT)
            # This is what get_response_synthesizer() does internally
            prompt_helper = PromptHelper.from_llm_metadata(llm.metadata)
            self._response_synthesizer = CompactAndRefine(
                llm=llm,
                callback_manager=callback_manager,
                prompt_helper=prompt_helper,
                text_qa_template=CITATION_QA_TEMPLATE,
                refine_template=CITATION_REFINE_TEMPLATE,
                streaming=False,
            )
            
            # self._response_synthesizer = Generation(
            #     llm=llm,
            #     callback_manager=callback_manager,
            #     prompt_helper=prompt_helper,
            #     simple_template=CITATION_QA_TEMPLATE,
            #     streaming=False,
            # )

        self._node_postprocessors = node_postprocessors or []
        self._metadata_mode = metadata_mode

        for node_postprocessor in self._node_postprocessors:
            node_postprocessor.callback_manager = callback_manager

        super().__init__(callback_manager=callback_manager)

    @classmethod
    def from_args(
        cls,
        index: BaseGPTIndex,
        llm: Optional[LLM] = None,
        response_synthesizer: Optional[BaseSynthesizer] = None,
        citation_chunk_size: int = DEFAULT_CITATION_CHUNK_SIZE,
        citation_chunk_overlap: int = DEFAULT_CITATION_CHUNK_OVERLAP,
        text_splitter: Optional[TextSplitter] = None,
        retriever: Optional[BaseRetriever] = None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        # response synthesizer args
        response_mode: ResponseMode = ResponseMode.COMPACT,
        use_async: bool = False,
        streaming: bool = False,
        # class-specific args
        metadata_mode: MetadataMode = MetadataMode.NONE,
        **kwargs: Any,
    ) -> "CitationQueryEngine":
        """
        Initialize a CitationQueryEngine object.".

        Args:
            index: (BastGPTIndex): index to use for querying
            llm: (Optional[LLM]): LLM object to use for response generation.
            citation_chunk_size (int):
                Size of citation chunks, default=512. Useful for controlling
                granularity of sources.
            citation_chunk_overlap (int): Overlap of citation nodes, default=20.
            text_splitter (Optional[TextSplitter]):
                A text splitter for creating citation source nodes. Default is
                a SentenceSplitter.
            retriever (BaseRetriever): A retriever object.
            node_postprocessors (Optional[List[BaseNodePostprocessor]]): A list of
                node postprocessors.
            verbose (bool): Whether to print out debug info.
            response_mode (ResponseMode): A ResponseMode object.
            use_async (bool): Whether to use async.
            streaming (bool): Whether to use streaming.
            optimizer (Optional[BaseTokenUsageOptimizer]): A BaseTokenUsageOptimizer
                object.

        """
        retriever = retriever or index.as_retriever(**kwargs)
        return cls(
            retriever=retriever,
            llm=llm,
            response_synthesizer=response_synthesizer,
            callback_manager=Settings.callback_manager,
            citation_chunk_size=citation_chunk_size,
            citation_chunk_overlap=citation_chunk_overlap,
            text_splitter=text_splitter,
            node_postprocessors=node_postprocessors,
            metadata_mode=metadata_mode,
        )

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules."""
        return {"response_synthesizer": self._response_synthesizer}

    def _create_citation_nodes(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """
        Modify retrieved nodes to be granular sources with markdown citation links.
        
        Each source is formatted with a clickable markdown link that includes:
        - Document filename
        - Page number
        - Full URL for direct navigation
        
        Format: Source: [filename (p.X)](url)
        
        This allows the LLM to copy the exact citation format into its response,
        creating clickable links in the frontend without post-processing.
        """
        new_nodes: List[NodeWithScore] = []
        for node in nodes:
            metadata = node.node.metadata or {}
            
            # Extract citation metadata
            filename = metadata.get("filename", "document")
            page_number = metadata.get("page_number", "?")
            url = metadata.get("url", "")
            
            # Create markdown citation link: [filename (p.X)](url)
            if url:
                citation_link = f"[{filename} (p.{page_number})]({url})"
            else:
                # Fallback if no URL available
                citation_link = f"[{filename} (p.{page_number})]"
            
            text_chunks = self.text_splitter.split_text(
                node.node.get_content(metadata_mode=self._metadata_mode)
            )

            for text_chunk in text_chunks:
                # Format: Source: [citation_link]\n{content}
                text = f"Source: {citation_link}\n{text_chunk}\n"

                new_node = NodeWithScore(
                    node=TextNode.model_validate(node.node.model_dump()),
                    score=node.score,
                )
                new_node.node.set_content(text)
                new_nodes.append(new_node)
        return new_nodes

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = self._retriever.retrieve(query_bundle)

        for postprocessor in self._node_postprocessors:
            nodes = postprocessor.postprocess_nodes(nodes, query_bundle=query_bundle)

        return nodes

    async def aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = await self._retriever.aretrieve(query_bundle)

        for postprocessor in self._node_postprocessors:
            nodes = postprocessor.postprocess_nodes(nodes, query_bundle=query_bundle)

        return nodes

    def prepare_streaming_context(
        self,
        query: str,
        nodes: Optional[List[NodeWithScore]] = None,
    ) -> Dict[str, Any]:
        """
        Prepare context for async streaming generation.
        
        This method replicates the CitationQueryEngine workflow without executing
        the LLM call, allowing external async streaming with BedrockConverse.
        
        Workflow replicated:
        1. Retrieve nodes (if not provided)
        2. Create citation nodes with markdown links
        3. Pack text chunks using CompactAndRefine logic
        4. Format CITATION_QA_TEMPLATE with packed context
        
        Args:
            query: The user's query string.
            nodes: Optional pre-retrieved nodes. If None, will retrieve.
            
        Returns:
            Dict containing:
            - prompt: Formatted prompt ready for LLM
            - citation_nodes: The processed citation nodes (for metadata extraction)
            - context_str: The packed context string
            - context_overflow: Whether context exceeded single-call limit
            - total_chunks: Number of packed chunks (1 if no overflow)
        """
        query_bundle: QueryBundle = QueryBundle(query_str=query)
        
        # Step 1: Retrieve if not provided
        if nodes is None:
            nodes = self.retrieve(query_bundle)
        
        # Step 2: Create citation nodes with Source: [filename (p.X)](url) format
        citation_nodes: List[NodeWithScore] = self._create_citation_nodes(nodes)
        
        # Step 3: Extract text content from citation nodes
        text_chunks: List[str] = [
            node.node.get_content(metadata_mode=MetadataMode.LLM)
            for node in citation_nodes
        ]
        
        # Step 4: Pack text chunks using CompactAndRefine logic
        # This uses PromptHelper to fit context within LLM's context window
        # partial_format pre-fills {query_str}, leaving {context_str} unfilled
        text_qa_template: PromptTemplate = CITATION_QA_TEMPLATE.partial_format(query_str=query)
        refine_template: PromptTemplate = CITATION_REFINE_TEMPLATE.partial_format(query_str=query)
        
        # get_biggest_prompt compares token count of templates, returns larger one
        # Refine template is bigger (has {existing_answer}), used for conservative sizing
        max_prompt: PromptTemplate = get_biggest_prompt([text_qa_template, refine_template])
        
        # Get prompt_helper from synthesizer or create one
        # PromptHelper knows context_window size and handles token counting/packing
        prompt_helper: PromptHelper
        if hasattr(self._response_synthesizer, '_prompt_helper'):
            prompt_helper = self._response_synthesizer._prompt_helper
        else:
            llm: LLM = self._response_synthesizer._llm
            prompt_helper = PromptHelper.from_llm_metadata(llm.metadata)
        
        # Repack text chunks to fit context window
        # Returns List[str] - if all fits, single element; if overflow, multiple elements
        packed_chunks: List[str] = prompt_helper.repack(
            max_prompt, 
            text_chunks, 
            llm=self._response_synthesizer._llm
        )
        
        # Step 5: Handle context overflow
        # If packed_chunks has multiple elements, it means context is too large for
        # a single LLM call. CompactAndRefine would make multiple calls with refinement,
        # but for streaming we only use the first chunk.
        # See documents/TODO.md for details on this limitation.
        context_overflow: bool = len(packed_chunks) > 1
        if context_overflow:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Context overflow in streaming: {len(packed_chunks)} chunks needed, "
                f"but only using first chunk. Consider reducing top_k or chunk_size. "
                f"CompactAndRefine would use refinement loop for remaining {len(packed_chunks) - 1} chunks."
            )
        
        # Use only the first packed chunk for streaming (fits in context window)
        context_str: str = packed_chunks[0] if packed_chunks else ""
        
        # Step 6: Format final prompt - complete prompt string ready for LLM
        prompt: str = CITATION_QA_TEMPLATE.format(
            context_str=context_str,
            query_str=query,
        )
        
        return {
            "prompt": prompt,
            "citation_nodes": citation_nodes,
            "context_str": context_str,
            "context_overflow": context_overflow,
            "total_chunks": len(packed_chunks),
        }

    @property
    def retriever(self) -> BaseRetriever:
        """Get the retriever object."""
        return self._retriever

    def get_last_timing(self) -> Dict[str, float]:
        """
        Get timing information from the last query.
        
        Returns:
            Dict with keys: retrieval_time, generation_time, total_time (all in seconds).
            Returns empty dict if no query has been executed yet.
        """
        return getattr(self, "last_timing", {})

    def synthesize(
        self,
        query_bundle: QueryBundle,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
    ) -> RESPONSE_TYPE:
        nodes = self._create_citation_nodes(nodes)
        return self._response_synthesizer.synthesize(
            query=query_bundle,
            nodes=nodes,
            additional_source_nodes=additional_source_nodes,
        )

    async def asynthesize(
        self,
        query_bundle: QueryBundle,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
    ) -> RESPONSE_TYPE:
        nodes = self._create_citation_nodes(nodes)
        return await self._response_synthesizer.asynthesize(
            query=query_bundle,
            nodes=nodes,
            additional_source_nodes=additional_source_nodes,
        )

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """
        Answer a query with timing instrumentation.
        
        LlamaIndex Core Concepts Explained:
        ====================================
        
        1. WHAT IS self.callback_manager?
        ---------------------------------
        CallbackManager is LlamaIndex's observability/instrumentation system.
        - Inherited from BaseQueryEngine via super().__init__(callback_manager=...)
        - Manages a list of callback handlers (e.g., LlamaDebugHandler, WandbHandler)
        - Provides .event() context manager to create trackable events
        - Default: Settings.callback_manager (global singleton)
        
        Example handlers you can attach:
          from llama_index.core.callbacks import LlamaDebugHandler
          callback_manager.add_handler(LlamaDebugHandler())
        
        2. WHAT IS self.callback_manager.event() OUTPUT?
        -------------------------------------------------
        Returns a context manager (CBEventContext) that:
        - On __enter__: Emits a START event to all handlers, returns event object
        - On __exit__: Emits an END event (or you call on_end() explicitly)
        - The `query_event` object has methods like:
            - on_end(payload={...}): Signal completion with result data
            - event_id: Unique ID for this event instance
        
        3. WHAT IS query_bundle.query_str?
        ----------------------------------
        QueryBundle is a wrapper around a query with additional context:
        
        class QueryBundle:
            query_str: str           # The actual query text, e.g., "What is RAG?"
            custom_embedding_strs: List[str]  # Optional: custom strings for embedding
            embedding: List[float]   # Optional: pre-computed embedding vector
        
        query_bundle.query_str is simply the raw query string the user typed.
        
        4. WHY USE QueryBundle INSTEAD OF JUST A STRING?
        -------------------------------------------------
        QueryBundle provides flexibility for advanced use cases:
        
        a) Pre-computed embeddings: Skip embedding computation if you already have it
           bundle = QueryBundle(query_str="...", embedding=[0.1, 0.2, ...])
        
        b) Custom embedding strings: Use different text for embedding vs display
           bundle = QueryBundle(
               query_str="What is ML?",  # Shown to user/LLM
               custom_embedding_strs=["machine learning definition"]  # Used for search
           )
        
        c) Hybrid search: Carry both text and vector representations together
        
        d) Consistent interface: All retrievers/engines expect QueryBundle,
           making it easy to swap components without changing signatures
        
        Note: engine.query("string") auto-wraps it: QueryBundle(query_str="string")
        
        5. WHAT IS EventPayload.QUERY_STR?
        ----------------------------------
        EventPayload is an enum of standardized payload keys:
        
        class EventPayload(str, Enum):
            QUERY_STR = "query_str"      # The query text
            NODES = "nodes"              # Retrieved/processed nodes
            RESPONSE = "response"        # Final response object
            PROMPT = "prompt"            # LLM prompt text
            COMPLETION = "completion"    # LLM completion text
            ...
        
        Using enum keys (vs raw strings) provides:
        - Type safety and IDE autocomplete
        - Consistent naming across all LlamaIndex components
        - Easier refactoring if key names change
        
        The payload dict is passed to all callback handlers, allowing them to
        log, trace, or process the event data uniformly.
        """
        # =====================================================================
        # OUTER EVENT: QUERY - Wraps the entire query pipeline
        # This is the top-level event that encompasses retrieval + synthesis.
        # When entering this block, a QUERY_START event is emitted.
        # =====================================================================
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            
            # Time retrieval phase
            t_retrieval_start = time.perf_counter()
            
            # =================================================================
            # INNER EVENT: RETRIEVE - Wraps the retrieval/search phase
            # This nested event tracks just the retrieval portion.
            # Handlers can distinguish retrieval time from synthesis time.
            # =================================================================
            with self.callback_manager.event(
                CBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: query_bundle.query_str},
            ) as retrieve_event:
                # ---------------------------------------------------------
                # RETRIEVAL LOGIC:
                # 1. self.retrieve() calls the underlying retriever
                #    (e.g., VectorStoreRetriever) to perform semantic search
                # 2. _create_citation_nodes() splits retrieved chunks into
                #    smaller citation-friendly pieces with "Source N:" prefix
                # ---------------------------------------------------------
                nodes = self.retrieve(query_bundle)
                nodes = self._create_citation_nodes(nodes)

                # Signal end of RETRIEVE event with the retrieved nodes
                # This allows handlers to inspect what was retrieved
                retrieve_event.on_end(payload={EventPayload.NODES: nodes})
            
            retrieval_time = time.perf_counter() - t_retrieval_start

            # Time generation phase
            t_generation_start = time.perf_counter()
            
            # =================================================================
            # SYNTHESIS PHASE: LLM Generation
            # The response_synthesizer takes the query + retrieved nodes and
            # generates a response using the LLM. Internally, this triggers
            # LLM events (CBEventType.LLM) for each LLM call.
            # =================================================================
            response = self._response_synthesizer.synthesize(
                query=query_bundle,
                nodes=nodes,
            )
            
            generation_time = time.perf_counter() - t_generation_start

            # Signal end of QUERY event with the final response
            # This completes the trace for the entire query pipeline
            query_event.on_end(payload={EventPayload.RESPONSE: response})

        # Store timing in last_timing attribute for access after query
        self.last_timing = {
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_time": retrieval_time + generation_time,
        }

        return response

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            with self.callback_manager.event(
                CBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: query_bundle.query_str},
            ) as retrieve_event:
                nodes = await self.aretrieve(query_bundle)
                nodes = self._create_citation_nodes(nodes)

                retrieve_event.on_end(payload={EventPayload.NODES: nodes})

            response = await self._response_synthesizer.asynthesize(
                query=query_bundle,
                nodes=nodes,
            )

            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response