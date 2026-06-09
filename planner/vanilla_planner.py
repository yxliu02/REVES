from typing import Dict, Any, Optional
from utils.apis import call_llm  # Requires OPENAI_API_KEY/GOOGLE_API_KEY, set OPENAI_API_BASE for vllm
from utils.prompt_utils import build_prompt

class Planner:
    """
    A vanilla planner that handles different task types.
    Each task loads its prompt template from the `prompt` package and uses call_llm to execute.
    LLM settings are provided via llm_config at initialization.
    """
    def __init__(
        self,
        task_name: str,
        prev_response: str = "",
        llm_config: Dict[str, Any] = None
    ):
        self.task_name = task_name
        self.prev_response = prev_response

        defaults = {
            "model_type": "gemini",
            "model_name": "gemini-1.5-flash",
            "temperature": 0.0,
            "max_tokens": 4096,
        }
        self.llm_config = {**defaults, **(llm_config or {})}
        # print(f"Generation Config: {self.llm_config}")

    def run(self, query: Dict) -> str:
        """
        Generate the plan/code by calling call_llm with the formatted prompt and stored LLM settings.
        Returns raw text from the model (planner-specific post-processing happens outside).
        """
        if self.task_name == "TravelPlanner":
            if isinstance(query, str):
                query = eval(query)
            if isinstance(query, dict) and isinstance(query.get("local_constraint"), str):
                query["local_constraint"] = eval(query["local_constraint"])
        assert self.prev_response == ""

        prompt_text = build_prompt(
            task_name=self.task_name,
            query=query,
            prev_response=self.prev_response
        )

        call_args = {**self.llm_config, "prompt": prompt_text}
        return *call_llm(**call_args), 1


if __name__ == "__main__":
    from datasets import load_dataset
    import os, sys

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, base_dir)

    data = load_dataset('osunlp/TravelPlanner', 'validation')['validation']
    first = data[-1]
    reference_information = first['reference_information']
    llm_cfg = {
        "model_type": "dsv3",
        "model_name": "deepseek-v3",
        "temperature": 0.0,
        "max_tokens": 4096
    }
    planner = Planner(task_name='TravelPlanner', llm_config=llm_cfg)
    result = planner.run(query=first, reference_info=reference_information)
    print("Generated Plan:", result)