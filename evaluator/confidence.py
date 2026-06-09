from typing import Sequence, Dict, Any, List, Optional, Set
import math

DEFAULT_SPECIAL_TOKENS = {"<|endoftext|>", "</s>", "<|eos|>", "<|end|>", "<|im_end|>", "<|im_start|>"}

def compute_token_confidence_from_topk(
    usage: Dict[str, Any],
    k_top: int = 1,
    exclude_special: bool = True,
    special_tokens: Optional[Set[str]] = None,
    fallback_to_chosen: bool = True,
) -> List[float]:
    """
    Returns per-token C_i list aligned with usage["tokens"] (same length).
    Definition (as in your figure):
      C_i = - (1/k) * sum_{j=1..k} log p_i(j)   using top-k token logprobs at step i.

    If usage contains "top_logprobs", we use it.
    If not, and fallback_to_chosen=True, we fallback to chosen token logprob:
      C_i = - token_logprobs[i]
    """
    tokens: Sequence[str] = usage.get("tokens", [])
    token_logprobs: Sequence[Optional[float]] = usage.get("token_logprobs", [])
    top_logprobs: Sequence[Optional[Dict[str, float]]] = usage.get("top_logprobs", None)

    if special_tokens is None:
        special_tokens = DEFAULT_SPECIAL_TOKENS

    if not tokens or len(tokens) != len(token_logprobs):
        raise ValueError("usage must contain aligned 'tokens' and 'token_logprobs'")

    if top_logprobs is not None and len(top_logprobs) != len(tokens):
        # Some servers may return empty or mismatched lengths; treat as unavailable
        top_logprobs = None

    Cis: List[float] = []
    for i, t in enumerate(tokens):
        if exclude_special and (t in special_tokens):
            Cis.append(float("nan"))
            continue

        # Prefer top-k
        if isinstance(top_logprobs, list) and top_logprobs is not None:
            d = top_logprobs[i]
            if isinstance(d, dict) and len(d) > 0:
                # take top k by logprob
                vals = sorted(d.values(), reverse=True)[: max(1, k_top)]
                Cis.append(-float(sum(vals) / len(vals)))
                continue

        # Fallback
        if not fallback_to_chosen:
            Cis.append(float("nan"))
            continue
        lp = token_logprobs[i]
        if lp is None or (isinstance(lp, float) and math.isnan(lp)):
            Cis.append(float("nan"))
        else:
            Cis.append(-float(lp))

    return Cis

from typing import Dict, Any, Optional, Set
import math

def compute_tail_confidence(
    usage: Dict[str, Any],
    K: int = 32,
    k_top: int = 1,
    exclude_special: bool = True,
    special_tokens: Optional[Set[str]] = None,
) -> float:
    """
    Tail confidence:
      C_tail = avg(C_{N-K+1}, ..., C_N)
    where C_i is token confidence computed from top-k logprobs.

    We first compute all C_i (with NaN for excluded special tokens),
    then take last K valid (non-NaN) tokens and average.
    """
    Cis = compute_token_confidence_from_topk(
        usage=usage,
        k_top=k_top,
        exclude_special=exclude_special,
        special_tokens=special_tokens,
        fallback_to_chosen=True,
    )

    # keep valid indices
    valid = [c for c in Cis if not (isinstance(c, float) and math.isnan(c))]
    if not valid:
        return float("nan")

    tail = valid[-max(1, K):]
    return float(sum(tail) / len(tail))