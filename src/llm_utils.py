"""Tiện ích gọi LLM: khởi tạo model, thử lại khi lỗi, giới hạn tốc độ và đọc JSON từ output."""
import json
import re
import threading
import time

from src import config

_rate_lock = threading.Lock()
_last_call = 0.0
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def get_llm(model: str | None = None, temperature: float = 0.1):
    """Tạo ChatGroq. Import trong hàm để phần đồ thị vẫn chạy được khi chưa cài langchain."""
    from langchain_groq import ChatGroq

    if not config.GROQ_API_KEY:
        raise RuntimeError("Thiếu GROQ_API_KEY trong file .env")
    return ChatGroq(model=model or config.LLM_MODEL, api_key=config.GROQ_API_KEY, temperature=temperature)


def _wait_for_rate_limit(requests_per_minute: int):
    global _last_call
    interval = 60.0 / requests_per_minute
    with _rate_lock:
        wait = _last_call + interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()


def call_llm(llm, prompt: str, retries: int = 3, backoff: float = 2.0, throttle: bool = False) -> str:
    """Gọi LLM và trả về text. Lỗi (VD: 429 rate limit) thì chờ backoff × 2^lần_thử rồi thử lại.

    throttle=True giãn đều các lượt gọi theo LLM_REQUESTS_PER_MINUTE — dùng cho các job chạy hàng loạt.
    """
    for attempt in range(retries + 1):
        if throttle:
            _wait_for_rate_limit(config.LLM_REQUESTS_PER_MINUTE)
        try:
            response = llm.invoke(prompt)
            return getattr(response, "content", response)
        except Exception:
            if attempt == retries:
                raise
            time.sleep(backoff * 2 ** attempt)


def parse_json(text) -> dict | None:
    """Lấy object JSON đầu tiên trong output LLM (bỏ qua ```json fence và chữ thừa). Không có thì trả None."""
    if not isinstance(text, str):
        return None
    decoder = json.JSONDecoder()
    for candidate in _FENCE.findall(text) + [text]:
        for match in re.finditer(r"\{", candidate):
            try:
                obj, _ = decoder.raw_decode(candidate, match.start())
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    return None
