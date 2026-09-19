"""Tiện ích gọi LLM qua Groq: khởi tạo model, thử lại khi lỗi (VD 429 rate limit).

Cố ý giữ cùng "hình dạng" với src/llm_utils.py của Hùng (get_llm / call_llm) để khi merge
2 module vào một repo chỉ cần giữ lại một file dùng chung.
"""
import time

from src import config

_llm_instance = None


def get_llm(model: str = None, temperature: float = None):
    """Tạo ChatGroq (cache 1 lần). Import trong hàm để phần chunk/embed vẫn chạy khi chưa cài langchain."""
    global _llm_instance
    if _llm_instance is None:
        from langchain_groq import ChatGroq

        if not config.GROQ_API_KEY:
            raise RuntimeError("Thiếu GROQ_API_KEY trong file .env (lấy tại https://console.groq.com/)")
        _llm_instance = ChatGroq(
            model=model or config.LLM_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=config.LLM_TEMPERATURE if temperature is None else temperature,
        )
    return _llm_instance


def call_llm(llm, prompt: str, retries: int = 3, backoff: float = 2.0) -> str:
    """Gọi LLM, trả text. Lỗi thì chờ backoff × 2^lần_thử rồi thử lại (mũ)."""
    for attempt in range(retries + 1):
        try:
            response = llm.invoke(prompt)
            return getattr(response, "content", str(response))
        except Exception:
            if attempt == retries:
                raise
            time.sleep(backoff * 2 ** attempt)
