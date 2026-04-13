import os
import logging
from typing import Iterable, Tuple


class _NamespaceFilter(logging.Filter):
    """Keeps the log file scoped to the project namespaces."""

    def __init__(self, namespaces: Iterable[str] | None):
        super().__init__()
        self.namespaces: Tuple[str, ...] = tuple(ns for ns in (namespaces or ()) if ns)

    def filter(self, record: logging.LogRecord) -> bool:
        if not self.namespaces:
            return True
        if record.name in {"root", "__main__"}:
            return True
        return record.name.startswith(self.namespaces)


def setup_logging(
    run_id,
    result_dir,
    add_console_handler=False,
    verbose=False,
    allowed_namespaces=("lits", "mem0"),
    override=True,
):
    os.makedirs(result_dir, exist_ok=True)
    log_path = os.path.join(result_dir, f"{run_id}.log")
    namespace_filter = _NamespaceFilter(allowed_namespaces)

    # ～～～～～～ 文件处理器 (Begin) ～～～～～～
    # Create file handler. If the file exists, append the log.
    file_mode = 'w' if override else 'a'
    file_handler = logging.FileHandler(log_path, mode=file_mode, encoding='utf-8')
    
    
    # 文件处理器的级别设置为 logging.DEBUG，这意味着它会接收所有日志事件（只要记录器本身的级别允许）
    file_handler.setLevel(logging.DEBUG)
    
    # Formatters
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    if namespace_filter.namespaces:
        file_handler.addFilter(namespace_filter)
    # ～～～～～～ 文件处理器 (End) ～～～～～～
    
    # ～～～～～～ Logger Setup (Begin) ～～～～～～
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO) # # logging.DEBUG（会记录所有级别的消息）; logging.INFO（会记录 INFO、WARNING、ERROR、CRITICAL 级别的消息）
    logger.propagate = False # 没有实际效果，因为没有“上一级”可以继续冒泡
    logger.handlers.clear() # 清除根记录器上所有现有的处理器（Handler）。这是最关键的一步，确保每次调用该函数时，日志配置都是全新的，避免重复或冲突。
    logger.addHandler(file_handler) # Add handlers to logger
    # ～～～～～～ Logger Setup (End) ～～～～～～

    #  ～～～～～～  Console handler (Begin) ～～～～～～
    if add_console_handler:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        if namespace_filter.namespaces:
            console_handler.addFilter(namespace_filter)
        logger.addHandler(console_handler)
    #  ～～～～～～  Console handler (End) ～～～～～～
    return logger


def print_timing_summary(
    timing_results: dict,
    title: str = "Timing Summary",
    logger: logging.Logger = None,
    compact: bool = False,
) -> None:
    """
    Print a formatted timing summary to console and optionally to logger.
    
    Args:
        timing_results: Dict with timing keys. Supported keys:
            - ingestion: Document ingestion time
            - index_creation: Qdrant index/connection time
            - retrieval / engine_retrieval: Semantic search time
            - generation / engine_generation: LLM generation time
            - memory: Memory retrieval time
            - total / citation_query_total: Total query time
        title: Title for the summary section.
        logger: Optional logger to also write timing info.
        compact: If True, print single-line format without borders.
        
    Example:
        timing = {"ingestion": 2.5, "retrieval": 0.5, "generation": 3.2, "total": 6.2}
        print_timing_summary(timing, title="Chat Service Timing")
        
        # Compact mode for inline use
        print_timing_summary(timing, compact=True)
        # Output: Ingestion: 2.500s | Retrieval: 0.500s | Generation: 3.200s | Total: 6.200s
    """
    if compact:
        # Single-line compact format
        parts = []
        if timing_results.get("ingestion", 0) > 0:
            parts.append(f"Ingestion: {timing_results['ingestion']:.2f}s")
        if timing_results.get("memory", 0) > 0:
            parts.append(f"Memory: {timing_results['memory']:.2f}s")
        retrieval = timing_results.get("retrieval") or timing_results.get("engine_retrieval")
        if retrieval:
            parts.append(f"Retrieval: {retrieval:.2f}s")
        generation = timing_results.get("generation") or timing_results.get("engine_generation")
        if generation:
            parts.append(f"Generation: {generation:.2f}s")
        total = timing_results.get("total") or timing_results.get("citation_query_total")
        if total:
            parts.append(f"Total: {total:.2f}s")
        
        msg = " | ".join(parts)
        print(f"  {msg}")
        if logger:
            logger.info(msg)
        return
    
    # Full format with borders
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    
    if logger:
        logger.info("=" * 60)
        logger.info(title)
        logger.info("=" * 60)
    
    def _log(msg):
        print(f"  {msg}")
        if logger:
            logger.info(msg)
    
    # Ingestion
    if "ingestion" in timing_results and timing_results["ingestion"] > 0:
        _log(f"1) Ingestion: {timing_results['ingestion']:.3f}s")
    
    # Index creation
    if "index_creation" in timing_results:
        _log(f"2) Index Creation (Qdrant): {timing_results['index_creation']:.3f}s")
    
    # Retrieval
    retrieval = timing_results.get("retrieval") or timing_results.get("engine_retrieval")
    if retrieval is not None:
        _log(f"3) Retrieval (semantic search): {retrieval:.3f}s")
    
    # Generation
    generation = timing_results.get("generation") or timing_results.get("engine_generation")
    if generation is not None:
        _log(f"4) Generation (LLM): {generation:.3f}s")
    
    # Memory
    if "memory" in timing_results and timing_results["memory"] > 0:
        _log(f"5) Memory retrieval: {timing_results['memory']:.3f}s")
    
    # Total
    total = timing_results.get("total") or timing_results.get("citation_query_total")
    if total is not None:
        _log(f"TOTAL: {total:.3f}s")
    
    print("=" * 60)
    if logger:
        logger.info("=" * 60)
