# warning: set OPENAI_API_KEY or GOOGLE_API_KEY before calling API, if you select 'model_type'=='vllm'
# remember to set OPENAI_API_BASE="http://localhost:8000/v1"

import os
from langchain_openai import ChatOpenAI 
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing import List, Union, Literal, Optional, Dict
from transformers import AutoTokenizer

# API Keys
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', '')
DSV3_API_URL = os.environ.get('DSV3_API_URL', 'http://10.6.166.143:8000/v1/completions')
LOCAL_CHECKPOINT_PATH = os.environ.get('LOCAL_CHECKPOINT_PATH', '')

# [!!arning!!]: only dsv3 support n > 1
def call_llm(prompt: str, model_type: str = 'openai', model_name: str = 'gpt-3.5-turbo', 
             temperature: float = 0, max_tokens: int = 4096, num_responses: int = 1
             ) -> str:
    """
    Call a language model API with the given prompt and return the response.
    
    Args:
        prompt: The input prompt to send to the LLM
        model_type: Type of model API to use ('openai', 'gemini', or 'vllm')
        model_name: Name of the model to use
        temperature: Sampling temperature (0-1)
        max_tokens: Maximum number of tokens to generate    
    Returns:
        str: The generated text response from the LLM
    """
    try:
        if model_type == 'openai':
            llm = ChatOpenAI(
                model_name=model_name, 
                temperature=temperature, 
                max_tokens=max_tokens, 
                openai_api_key=OPENAI_API_KEY
            )
            return str(llm.invoke([HumanMessage(content=prompt)]).content)
        elif model_type == 'gemini':
            import google.generativeai as genai
            from google.generativeai import types as genai_types

            genai.configure(api_key=GOOGLE_API_KEY)
            model = genai.GenerativeModel(model_name)

            resp = model.generate_content(
                prompt,
                generation_config=genai_types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    candidate_count=num_responses
                )
            )
            outputs = [p.text for c in resp.candidates for p in c.content.parts]
            
            # Calculate usage - try to get from response, otherwise set to 0
            usage_metadata = getattr(resp, 'usage_metadata', None)
            if usage_metadata:
                in_tok = int(getattr(usage_metadata, 'prompt_token_count', 0))
                out_tok = int(getattr(usage_metadata, 'candidates_token_count', 0))
            else:
                in_tok = 0
                out_tok = 0
            
            usage = {"input_token": in_tok, "output_token": out_tok}
            
            if num_responses == 1:
                text = outputs[0] if outputs else ""
                return text, usage
            else:
                return outputs, usage

        elif model_type == 'dsv3':
            from .dsv3 import DSCV2InferencesHF, inference
            decoding_params = {
            "max_tokens": max_tokens,
            "top_p": 1.0,
            "top_k": -1,
            "temperature": temperature,
            "repetition_penalty": 1.0,
            "num_responses": num_responses,
            }
            _conn = DSCV2InferencesHF(checkpoint=os.environ.get('DSV3_CHECKPOINT', ''),
            url=DSV3_API_URL, 
            decoding_params=decoding_params)
            return inference(_conn, prompt)
        
        elif model_type == 'local':
            from .local import AdaptiveInferencesHF, inference
            decoding_params = {
                "max_tokens": max_tokens,
                "top_p": 1.0,
                "top_k": -1,
                "temperature": temperature,
                "repetition_penalty": 1.0,
                "num_responses": num_responses,
            }
            _conn = AdaptiveInferencesHF(
                checkpoint=LOCAL_CHECKPOINT_PATH,   # e.g. "/checkpoints/Meta-Llama-3.1-8B-Instruct" OR your DeepSeek-V3 path
                url=DSV3_API_URL,
                decoding_params=decoding_params,
                system_prompt=None,                 # optionally set a system prompt for chat-template models
            )
            return inference(_conn, prompt)

        elif model_type == "vllm_local":
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer

            # ---- Global cache (multi-model safe) ----
            global _VLLM_ENGINES, _TOKENIZERS
            if "_VLLM_ENGINES" not in globals():
                _VLLM_ENGINES = {}
            if "_TOKENIZERS" not in globals():
                _TOKENIZERS = {}

            tp_size = int(os.environ.get("TP_SIZE", 1))
            key = (model_name, tp_size)
            if key not in _VLLM_ENGINES:
                _VLLM_ENGINES[key] = LLM(
                    model=model_name,
                    tensor_parallel_size=tp_size,
                    dtype="bfloat16",
                    trust_remote_code=True,
                )
            engine = _VLLM_ENGINES[key]

            if model_name not in _TOKENIZERS:
                _TOKENIZERS[model_name] = AutoTokenizer.from_pretrained(
                    model_name, use_fast=True, trust_remote_code=True
                )
            tok = _TOKENIZERS[model_name]

            # ---- Unified rendering (no hardcoding; respects model's own chat template) ----
            system_prompt = os.environ.get("SYSTEM_PROMPT", "")  # can be left empty
            if isinstance(prompt, str):
                messages = (
                    [{"role": "system", "content": system_prompt}] if system_prompt else []
                ) + [{"role": "user", "content": prompt}]
            elif isinstance(prompt, list):
                # Also supports passing messages (list[dict]) directly from the caller
                messages = prompt
            else:
                raise ValueError("prompt must be str or list[{'role','content'}]")

            rendered = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True  # standard practice for generation
            )

            sp = SamplingParams(
                temperature=temperature,
                top_p=1.0,
                max_tokens=max_tokens,
                n=num_responses,
                stop=None,  # custom stop tokens can be passed in from outside if needed
            )
            # print(rendered)
            req_outputs = engine.generate([rendered], sp)
            req = req_outputs[0]

            # Text
            candidates = [o.text.strip() for o in req.outputs]
            text = candidates[0] if num_responses == 1 else candidates

            # ---- Usage calculation (prefer vLLM, fallback to tokenizer; sum across candidates to align with dsv3) ----
            # input tokens
            try:
                in_tok = int(len(getattr(req, "prompt_token_ids", []) or []))
            except Exception:
                enc = tok(rendered, add_special_tokens=True, return_length=True)
                in_tok = int(enc["length"][0])

            # output tokens
            try:
                if num_responses == 1:
                    out_tok = int(len(getattr(req.outputs[0], "token_ids", []) or []))
                else:
                    out_tok = int(sum(len(getattr(o, "token_ids", []) or []) for o in req.outputs))
            except Exception:
                if num_responses == 1:
                    enc = tok(text, add_special_tokens=True, return_length=True)
                    out_tok = int(enc["length"][0])
                else:
                    out_tok = int(sum(tok(t, add_special_tokens=True, return_length=True)["length"][0] for t in candidates))

            usage = {"input_token": in_tok, "output_token": out_tok}
            return text, usage

        elif model_type == 'vllm':

            llm = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                n=num_responses
            )

            if num_responses == 1:
                msg = llm.invoke([HumanMessage(content=prompt)])
                text = str(msg.content)

                raw_usage = msg.response_metadata.get("token_usage", {})
                if raw_usage:
                    in_tok = int(raw_usage.get("prompt_tokens", 0))
                    out_tok = int(raw_usage.get("completion_tokens", 0))
                else:
                    # fallback: tokenizer counting
                    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
                    rendered = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                                       tokenize=False, add_generation_prompt=True)
                    in_tok = tok(rendered, add_special_tokens=True, return_length=True)["length"][0]
                    out_tok = tok(text, add_special_tokens=True, return_length=True)["length"][0]

                usage = {"input_token": in_tok, "output_token": out_tok}
                return text, usage

            else:
                result_obj = llm.generate([[HumanMessage(content=prompt)]])
                generations = result_obj.generations[0]
                texts = [str(gen.message.content) for gen in generations]

                raw_usage = result_obj.llm_output.get("token_usage", {})
                if raw_usage:
                    in_tok = int(raw_usage.get("prompt_tokens", 0))
                    out_tok = int(raw_usage.get("completion_tokens", 0))
                else:
                    # fallback: tokenizer counting
                    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
                    rendered = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                                       tokenize=False, add_generation_prompt=True)
                    in_tok = tok(rendered, add_special_tokens=True, return_length=True)["length"][0]
                    out_tok = sum(tok(t, add_special_tokens=True, return_length=True)["length"][0] for t in texts)

                usage = {"input_token": in_tok, "output_token": out_tok}
                return texts, usage
            # if num_responses == 1:
            #     msg = llm.invoke([HumanMessage(content=prompt)])
            #     return str(msg.content)
            # else:
            #     result_obj = llm.generate([[HumanMessage(content=prompt)]])
            #     generations = result_obj.generations[0]
            #     # print([str(gen.message.content) for gen in generations])
            #     return [str(gen.message.content) for gen in generations]


            # llm = ChatOpenAI(
            #     model_name=model_name, 
            #     temperature=temperature, 
            #     max_tokens=max_tokens, 
            #     openai_api_key=OPENAI_API_KEY
            # )
            # return str(llm.invoke([HumanMessage(content=prompt)]).content)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
            
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return f"Error: {str(e)}"


def call_llm_multi_turn(input: List[Dict], model_type: str = 'openai', model_name: str = 'gpt-3.5-turbo', 
             temperature: float = 0, max_tokens: int = 4096, 
             ) -> str:
    """
    Call a language model API with the given prompt and return the response.
    
    Args:
        input: Chat history of LLM and user
        [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': 'Hello, how are you today?'}]
        model_type: Type of model API to use ('openai', 'gemini', or 'vllm')
        model_name: Name of the model to use
        temperature: Sampling temperature (0-1)
        max_tokens: Maximum number of tokens to generate
    
    Returns:
        str: The generated text response from the LLM
    """
    try:
        if model_type == 'openai':
            llm = ChatOpenAI(
                model_name=model_name, 
                temperature=temperature, 
                max_tokens=max_tokens, 
                openai_api_key=OPENAI_API_KEY
            )
            response = str(llm.invoke(input).content)
            updated_history = input + [{"role":"assistant", "content":response}]
            return updated_history, response
        elif model_type == 'gemini':
            llm = ChatGoogleGenerativeAI(
                temperature=temperature,
                model=model_name,
                max_output_tokens=max_tokens,
                google_api_key=GOOGLE_API_KEY
            )
            response = str(llm.invoke(input).content)
            updated_history = input + [{"role":"assistant", "content":response}]
            return updated_history, response

        elif model_type == 'vllm':
            llm = ChatOpenAI(
                model_name=model_name, 
                temperature=temperature, 
                max_tokens=max_tokens, 
                openai_api_key=OPENAI_API_KEY
            )
            response = str(llm.invoke(input).content)
            updated_history = input + [{"role":"assistant", "content":response}]
            return updated_history, response
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
            
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return f"Error: {str(e)}"



import os
from typing import Dict, Any, Tuple, Union, List

def call_llm_with_prob(
    prompt: str,
    model_type: str = "vllm",
    model_name: str = "qwen-7b",
    temperature: float = 0.0,
    max_tokens: int = 4096,
    num_responses: int = 1,
    logprobs_k: int = 1,
) -> Tuple[Union[str, List[str]], Dict[str, Any]]:
    if model_type != "vllm":
        raise NotImplementedError("call_llm_with_prob only implements model_type='vllm'.")

    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("Missing dependency: openai. Please `pip install openai`.") from e

    base_url = os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY") or "EMPTY"
    client = OpenAI(api_key=api_key, base_url=base_url)

    # Hardcoded Qwen2.5-7B tokenizer here; keeping the original approach
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", use_fast=True, trust_remote_code=True)

    if not isinstance(prompt, str):
        raise ValueError("prompt must be str")

    messages = [{"role": "user", "content": prompt}]
    rendered_prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    resp = client.completions.create(
        model=model_name,
        prompt=rendered_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        n=num_responses,
        logprobs=logprobs_k,
        echo=False,
    )

    usage_obj = getattr(resp, "usage", None)
    in_tok = int(getattr(usage_obj, "prompt_tokens", 0) or 0) if usage_obj else 0
    out_tok = int(getattr(usage_obj, "completion_tokens", 0) or 0) if usage_obj else 0

    def _sum_logps(token_logprobs):
        vals = [x for x in token_logprobs if x is not None]
        logP = float(sum(vals)) if vals else float("nan")
        avg_logP = (logP / len(vals)) if vals else float("nan")
        min_logP = float(min(vals)) if vals else None
        return logP, avg_logP, min_logP, vals

    if num_responses == 1:
        c = resp.choices[0]
        text = (c.text or "").strip()

        lp_obj = getattr(c, "logprobs", None)
        token_logprobs = list(getattr(lp_obj, "token_logprobs", []) or []) if lp_obj else []
        tokens = list(getattr(lp_obj, "tokens", []) or []) if lp_obj else []
        top_logprobs = list(getattr(lp_obj, "top_logprobs", []) or []) if lp_obj else []  # <-- key field

        logP, avg_logP, min_logP, _vals = _sum_logps(token_logprobs)

        if out_tok == 0 and token_logprobs:
            out_tok = len([x for x in token_logprobs if x is not None])

        stats: Dict[str, Any] = {
            "input_token": in_tok,
            "output_token": out_tok,
            "logP": logP,
            "avg_logP": avg_logP,
            "min_logP": min_logP,
            "token_logprobs": token_logprobs,
            "tokens": tokens,
            "top_logprobs": top_logprobs,          # <-- returned
            "rendered_prompt": rendered_prompt,    # <-- optional: for debugging
            "logprobs_k": logprobs_k,              # <-- recorded for reference
        }
        return text, stats

    else:
        texts: List[str] = []
        logP_list: List[float] = []
        avg_logP_list: List[float] = []
        min_logP_list: List[Optional[float]] = []
        top_logprobs_list: List[List[Dict[str, float]]] = []

        fallback_out = 0

        for c in resp.choices:
            texts.append((c.text or "").strip())
            lp_obj = getattr(c, "logprobs", None)
            token_logprobs = list(getattr(lp_obj, "token_logprobs", []) or []) if lp_obj else []
            top_logprobs = list(getattr(lp_obj, "top_logprobs", []) or []) if lp_obj else []
            top_logprobs_list.append(top_logprobs)

            logP, avg_logP, min_logP, vals = _sum_logps(token_logprobs)
            logP_list.append(logP)
            avg_logP_list.append(avg_logP)
            min_logP_list.append(min_logP)
            fallback_out += len(vals)

        if out_tok == 0:
            out_tok = fallback_out

        stats = {
            "input_token": in_tok,
            "output_token": out_tok,
            "logP_list": logP_list,
            "avg_logP_list": avg_logP_list,
            "min_logP_list": min_logP_list,
            "top_logprobs_list": top_logprobs_list,
            "rendered_prompt": rendered_prompt,
            "logprobs_k": logprobs_k,
        }
        return texts, stats

if __name__ == "__main__":

    response, usage = call_llm(
        prompt="How is everything going?",
        model_type="vllm",
        model_name="qwen-7b",
        temperature=0.3,
        num_responses=3
    )
    print(response)
    print(usage)

    # response = call_llm(
    #     prompt="What is the capital of France?",
    #     model_type="local",
    #     model_name="qwen-7b",
    #     temperature=0.3,
    #     num_responses=3
    # )
    # print(response)
    # response = call_llm(prompt="What is the capital of France?", model_type="dsv3", model_name="deepseek-v3")
    # print(response)   
#     input=[
#   { "role": "system",    "content": "You are a helpful assistant." },
#   { "role": "user",      "content": "Hello, please give me today’s weather forecast." },
#   { "role": "assistant", "content": "Sure—what city would you like the forecast for?" },
#   { "role": "user",      "content": "Beijing." }
# ]
    # updated_history, response = call_llm_multi_turn(
    #     input=input,
    #     model_type="vllm",
    #     model_name="llama-70b"
    # )
    # print(updated_history)
    # print(response)
    # updated_history, response = call_llm_multi_turn(
    #     input=input,
    #     model_type="gemini",
    #     model_name="gemini-1.5-flash"        
    # )
    # print(updated_history)
    # print(response)