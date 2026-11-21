"""文本切分：长文本 → 带重叠的小片。"""


def split_text(text, size=400, overlap=80):
    """按字数切片，相邻片保留 overlap 字符重叠，避免语义被截断。"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    step = size - overlap
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return chunks
