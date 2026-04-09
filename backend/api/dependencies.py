"""
FastAPI 依赖注入：将 MemoryStore 注入到路由函数中。
"""

from repository.memory_store import MemoryStore

def get_store() -> MemoryStore:
    return MemoryStore()