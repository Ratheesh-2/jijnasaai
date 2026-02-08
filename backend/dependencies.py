from backend.config import get_settings
from backend.services.conversation_service import ConversationService


def get_conversation_service() -> ConversationService:
    return ConversationService()


def get_llm_router():
    from backend.services.llm_router import LLMRouter
    return LLMRouter(get_settings())


def get_rag_engine():
    from backend.services.rag_engine import RAGEngine
    from backend.services.vectorstore import VectorStoreManager
    settings = get_settings()
    vs = VectorStoreManager.get_instance(settings)
    return RAGEngine(settings, vs)


def get_voice_service():
    from backend.services.voice_service import VoiceService
    return VoiceService(get_settings())


def get_cost_tracker():
    from backend.services.cost_tracker import CostTracker
    return CostTracker(get_settings())
