# closing.py
# Fechamento do canal-fonte:
# - Prioriza a ÚLTIMA linha que tenha GREEN/✅ ou RED/❌
# - Extrai o ÚLTIMO dígito 1..4 dentro do ÚLTIMO par de () [] {} nessa linha
# - Fallback: procura no texto inteiro o último par com 1..4
import re
from typing import Optional

RX_PAREN_ANY = re.compile(r"[\(\[\{（【]([^)\]\}）】]*)[)\]\}）】]")
RX_LAST_PAREN_END = re.compile(
    r"[\(\[\{（【]\s*([^)\]\}）】]*?)\s*[)\]\}）】]\s*(?:✅|❌|GREEN|RED)?\s*$",
    re.I | re.M
)

GREEN_MARKERS = ("GREEN", "✅")
RED_MARKERS   = ("RED", "❌")

def _strip_noise(s: str) -> str:
    if not s: return ""
    return s.replace("\u200d","").replace("\u200c","").replace("\ufe0f","").strip()

def _extract_last_digit_1_4(chunk: str) -> Optional[int]:
    nums = re.findall(r"[1-4]", chunk or "")
    return int(nums[-1]) if nums else None

def _fallback_whole_text(text: str) -> Optional[int]:
    """Usa a regra antiga: último parêntese válido no texto completo."""
    t = _strip_noise(text)
    if not t: return None
    tail = list(RX_LAST_PAREN_END.finditer(t))
    if tail:
        d = _extract_last_digit_1_4(tail[-1].group(1))
        if d is not None:
            return d
    anyp = list(RX_PAREN_ANY.finditer(t))
    if anyp:
        d = _extract_last_digit_1_4(anyp[-1].group(1))
        if d is not None:
            return d
    return None

def extract_close_digit(text: str) -> Optional[int]:
    """
    Percorre linhas de baixo pra cima. Se a linha tiver GREEN/✅ ou RED/❌,
    tenta extrair o último dígito 1..4 em () [] {} nessa linha.
    Se não achar, cai no fallback que olha o texto inteiro.
    """
    t = _strip_noise(text or "")
    if not t: return None

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    for ln in reversed(lines):
        upper = ln.upper()
        if any(m in upper for m in GREEN_MARKERS + RED_MARKERS):
            # tenta em pares () [] {}
            parts = RX_PAREN_ANY.findall(ln)
            if parts:
                d = _extract_last_digit_1_4(parts[-1])
                if d is not None:
                    return d
            # fallback da própria linha: "( 2 )" etc.
            m = re.findall(r"\(([ \t]*([1-4])[ \t]*)\)", ln)
            if m:
                try:
                    return int(m[-1][1])
                except Exception:
                    pass

    # fallback global
    return _fallback_whole_text(t)
