# dsv3.py
from typing import List, Dict, Any, Union, Tuple, Optional
import requests
from transformers import AutoTokenizer


class DSCV2InferencesHF:
    def __init__(self, checkpoint: str, url: str, decoding_params: Optional[Dict[str, Any]] = None):
        """
        Minimal client for your DSv3 router. It returns:
          - text: str if n==1, else List[str]
          - usage: {"input_token": int, "output_token": int}
        Token counts prefer server "usage" when available; otherwise we estimate locally
        using the same tokenizer as the model.
        """
        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
        self.url = url
        self.decoding_params = decoding_params or {}

    def render_text(self, prompt: str) -> str:
        """
        Important: count tokens on the exact string sent to the model.
        Keep this template consistent with your inference server.
        """
        return f"<｜User｜>{prompt}<｜Assistant｜>"

    # ---------------------------- Local counting helpers ----------------------------
    def _count_batch(self, texts: List[str], add_special_tokens: bool = True) -> List[int]:
        """
        Tokenize a batch and return per-item token lengths.
        add_special_tokens=True to mirror typical inference behavior (BOS/EOS, etc.).
        """
        enc = self.tokenizer(texts, add_special_tokens=add_special_tokens, return_length=True)
        if "length" in enc:
            return [int(x) for x in enc["length"]]
        return [len(ids) for ids in enc["input_ids"]]

    def _normalize_usage(
        self,
        raw_usage: Optional[Dict[str, Any]],
        rendered_inputs: List[str],
        outputs: List[str],
    ) -> Dict[str, int]:
        """
        Unify usage shape. Prefer server's usage; if absent, estimate locally:
        - input_token: tokens of rendered inputs
        - output_token: tokens of generated outputs
        """
        if raw_usage:
            pt = int(raw_usage.get("prompt_tokens", raw_usage.get("input_tokens", 0)) or 0)
            ct = raw_usage.get("completion_tokens", raw_usage.get("output_tokens"))
            if ct is None:
                # If only total is provided, derive completion tokens.
                tt = int(raw_usage.get("total_tokens", 0) or 0)
                ct = max(0, tt - pt)
            return {"input_token": int(pt), "output_token": int(ct)}

        # Fallback: local estimation with the same tokenizer
        in_per = self._count_batch(rendered_inputs, add_special_tokens=True) if rendered_inputs else [0]
        out_per = self._count_batch(outputs, add_special_tokens=True) if outputs else []
        return {"input_token": int(sum(in_per)), "output_token": int(sum(out_per))}
    # -------------------------------------------------------------------------------

    def generate(self, prompts: Union[str, List[str]]) -> Tuple[Union[str, List[str]], Dict[str, int]]:
        """
        Returns:
          text: str if n==1 else List[str]
          usage: {"input_token": int, "output_token": int}
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        # Compute remaining tokens using rendered inputs (more faithful to real usage).
        rendered_inputs = [self.render_text(p) for p in prompts]
        rendered_token_lens = self._count_batch(rendered_inputs, add_special_tokens=True)
        longest_prompt_len = max(rendered_token_lens) if rendered_token_lens else 0

        # Your original hard limit preserved (8180); default generation length >= 3000.
        remaining_tokens = 8180 - longest_prompt_len
        max_tokens_default = max(remaining_tokens, 3000)

        payload = {
            "model": self.checkpoint,
            "prompt": rendered_inputs,
            "max_tokens": self.decoding_params.get("max_tokens", max_tokens_default),
            "top_p": self.decoding_params.get("top_p", 1.0),
            "top_k": self.decoding_params.get("top_k", 50),
            "temperature": self.decoding_params.get("temperature", 1.0),
            "repetition_penalty": self.decoding_params.get("repetition_penalty", 1.0),
            "n": self.decoding_params.get("num_responses", 1),
        }

        try:
            resp = requests.post(self.url, json=payload, timeout=10000)
            data = resp.json()
            choices = data.get("choices", [])
            outputs = [c.get("text", "").strip() for c in choices]
            usage = self._normalize_usage(data.get("usage"), rendered_inputs, outputs)
        except Exception as e:
            # On errors, return empty text and zero usage
            print(f"Error calling LLM: {e}")
            return ("" if payload["n"] == 1 else []), {"input_token": 0, "output_token": 0}

        text = outputs if len(outputs) > 1 else (outputs[0] if outputs else "")
        return text, usage


def inference(conn: DSCV2InferencesHF, item: str, repeat: int = 10, cot: bool = True) -> Tuple[Union[str, List[str]], Dict[str, int]]:
    """
    Simple retry wrapper that returns (text, usage).
    """
    for _ in range(repeat):
        try:
            text, usage = conn.generate(item)
            if text not in (None, "", []):
                return text, usage
        except ValueError:
            pass
    return "", {"input_token": 0, "output_token": 0}
