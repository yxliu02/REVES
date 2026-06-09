import ast
import math
import random
import json
from typing import Dict, Any, List, Optional

from utils.apis import call_llm
from utils.utils import task_name2extract_func, task_name2eval_func
from planner.vanilla_planner import Planner
from utils.prompt_utils import build_prompt, issues_from_eval, build_reflection_text

def _softmax(xs: List[float], tau: float = 1.0) -> List[float]:
    """softmax with temperature."""
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp((x - m) / tau) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]


class MindEvolutionPlanner(Planner):
    
    def __init__(
        self,
        task_name: str,
        prev_response: str = "",
        llm_config: Optional[Dict[str, Any]] = None,
        hyperparams: Optional[Dict[str, int]] = None,
    ):
        super().__init__(task_name, prev_response, llm_config)

        # Use exactly the hyper-parameters you specified
        default_hp = {
            "N_gens": 8,
            "N_islands": 4,
            "N_convs": 2,
            "N_seq": 2,
            "N_parent": 3,
            "N_emigrate": 2,
            "N_reset_interval": 3, #
            "N_reset": 2, #
            "N_top": 3, #
            "softmax_tau": 1.0,
        }
        self.hp = {**default_hp, **(hyperparams or {})}
        self.extract_obj_from_raw = task_name2extract_func(self.task_name)
        self.eval_obj = task_name2eval_func(self.task_name)
        self.usage = {'input_token': 0, 'output_token': 0}



    def _llm_multi(self, prompt: str, n: int) -> List[str]:
        """Call the LLM once to get n samples. Always return a list."""
        resp, single_usage = call_llm(**{**self.llm_config, "prompt": prompt, "num_responses": n})
        self.usage['input_token'] += single_usage['input_token']
        self.usage['output_token'] += single_usage['output_token']

        return resp if isinstance(resp, list) else [resp]

    def _score_many(self, raws: List[str], query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Score a batch of raw strings."""
        return [self._score_solution(raw, query) for raw in raws]

    def _obj_key(self, obj: Any) -> str:
        """Serialize obj to a canonical json string for equality check."""
        try:
            return json.dumps(obj, sort_keys=True, ensure_ascii=False)
        except Exception:
            # Treat unparsable obj as always new (so it won't be deduped).
            # Using raw random stamp to avoid collisions.
            return f"__RAW_FALLBACK__:{random.random()}"

    def _score_solution(self, raw: str, query: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a plan and return a candidate dict."""
        obj = self.extract_obj_from_raw(raw)
        eva = self.eval_obj(query, obj)
        score = eva["score"]
        issues = issues_from_eval(self.task_name, eva)
        key = self._obj_key(obj)
        return {"raw": raw, "score": score, "obj": obj, "issues": issues, "key": key}


    def _sample_parents(self, island: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not island:
            return []

        k = random.randint(0, self.hp["N_parent"])
        if k == 0:
            return []

        k = min(k, len(island))

        scores = [c["score"] for c in island]
        probs = _softmax(scores, tau=self.hp["softmax_tau"])

        pool = list(range(len(island)))
        chosen = []
        for _ in range(k):
            w = [probs[i] for i in pool]
            pos = random.choices(range(len(pool)), weights=w, k=1)[0]
            idx = pool.pop(pos)
            chosen.append(island[idx])
        return chosen

    def _produce_child_with_reflection(
        self,
        parents: List[Dict[str, Any]],
        query: Dict[str, Any],
        reference_info: str,
    ) -> Dict[str, Any]:
        """Produce one child using the reflection text composed from parents."""
        prev_response_text = build_reflection_text(task_name=self.task_name, parents=parents)
        prompt = build_prompt(
            task_name=self.task_name,
            query=query, 
            prev_response=prev_response_text
        )
        raw, single_usage = call_llm(**{**self.llm_config, "prompt": prompt})
        self.usage['input_token'] += single_usage['input_token']
        self.usage['output_token'] += single_usage['output_token']

        return self._score_solution(raw, query)

    def _seed_once(
        self,
        parents: List[Dict[str, Any]],
        query: Dict[str, Any],
        reference_info: str,
    ) -> Dict[str, Any]:
        """Create a single seed candidate. If no parents, use vanilla prompt; otherwise use reflection."""
        if not parents:
            prompt = build_prompt(
            task_name=self.task_name,
            query=query, 
            prev_response=""
            )
            raw, single_usage = call_llm(**{**self.llm_config, "prompt": prompt})
            self.usage['input_token'] += single_usage['input_token']
            self.usage['output_token'] += single_usage['output_token']
            return self._score_solution(raw, query)
        else:
            return self._produce_child_with_reflection(parents, query, reference_info)

    def _merge_dedup_keep_best_by_obj(
        self,
        old: List[Dict[str, Any]],
        new: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate purely by obj equality (via json.dumps(sort_keys=True)).
        Keep the higher-scoring one for the same obj. No capacity cap.
        """
        best: Dict[str, Dict[str, Any]] = {}
        for cand in old + new:
            key = cand["key"]
            if key not in best or cand["score"] > best[key]["score"]:
                best[key] = cand
        return sorted(best.values(), key=lambda c: c["score"], reverse=True)

    def _evolve_one_island(
        self,
        island: List[Dict[str, Any]],
        query: Dict[str, Any],
        reference_info: str,
    ) -> List[Dict[str, Any]]:
        """
        One generation of evolution for a single island:
          1) Sample parents for N_convs seeds. Batch-generate all zero-parent seeds (same prompt) with num_responses.
          2) For each seed, refine independently for N_seq-1 rounds.
          3) Merge with old population and deduplicate by obj equality.
        """
        seeds: List[Dict[str, Any]] = []
        parents_sets: List[List[Dict[str, Any]]] = []
        zero_parent_indices: List[int] = []

        # 1) Pre-sample parents for all seeds
        for i in range(self.hp["N_convs"]):
            parents = self._sample_parents(island)
            parents_sets.append(parents)
            if not parents:
                zero_parent_indices.append(i)

        # 2) Batch-generate zero-parent seeds
        new_candidates: List[Dict[str, Any]] = []
        if zero_parent_indices:
            prompt = build_prompt(
            task_name=self.task_name,
            query=query, 
            prev_response=""
            )
            raws = self._llm_multi(prompt, len(zero_parent_indices))
            scored = self._score_many(raws, query)
            # Assign them in order of zero_parent_indices
            it = iter(zero_parent_indices)
            for cand in scored:
                next(it)
                seeds.append(cand)
                new_candidates.append(cand)

        # 3) Generate non-zero-parent seeds one-by-one
        for i in range(self.hp["N_convs"]):
            if i in zero_parent_indices:
                continue
            parents = parents_sets[i]
            cand = self._seed_once(parents, query, reference_info)
            seeds.append(cand)
            new_candidates.append(cand)

        # 4) Independent refinements (N_seq - 1)
        for seed in seeds:
            curr = seed
            for _ in range(self.hp["N_seq"] - 1):
                child = self._produce_child_with_reflection([curr], query, reference_info)
                new_candidates.append(child)
                curr = child

        # 5) Merge + dedup (by obj)
        return self._merge_dedup_keep_best_by_obj(island, new_candidates)

    def _migrate_after_island(self, islands: List[List[Dict[str, Any]]], src_idx: int):
        """Immediately migrate top N_emigrate from island src_idx to island (src_idx + 1) % N."""
        n = len(islands)
        if n <= 1 or self.hp["N_emigrate"] <= 0:
            return

        dest = (src_idx + 1) % n
        src_sorted = sorted(islands[src_idx], key=lambda c: c["score"], reverse=True)
        migrants = [c.copy() for c in src_sorted[: self.hp["N_emigrate"]]]
        islands[dest] = self._merge_dedup_keep_best_by_obj(islands[dest], migrants)

    def _reset_islands(self, islands: List[List[Dict[str, Any]]]):
        """Pick global top_k and overwrite the worst `reset_count` islands by average score."""
        all_cands = [c for isl in islands for c in isl]
        elites = sorted(all_cands, key=lambda c: c["score"], reverse=True)[: self.hp["N_top"]]

        avgs: List[float] = []
        for isl in islands:
            if isl:
                avgs.append(sum(c["score"] for c in isl) / len(isl))
            else:
                avgs.append(float("-inf"))

        worst = sorted(range(len(islands)), key=lambda i: avgs[i])[: self.hp["N_reset"]]
        for idx in worst:
            islands[idx] = [e.copy() for e in elites]

    def run(self, query: Dict[str, Any], reference_info: Optional[str] = None) -> str:
        """
        Run Mind Evolution search. Assumes TravelPlanner evaluator/templating.
        Returns the raw best plan (or the first perfect plan with score == 0.0).
        """
        if self.task_name == "TravelPlanner":
            if isinstance(query, str):
                query = ast.literal_eval(query)
            if isinstance(query.get("local_constraint"), str):
                query["local_constraint"] = ast.literal_eval(query["local_constraint"])


        # Initialize islands
        islands: List[List[Dict[str, Any]]] = []
        best_raw, best_score = None, -500.0

        for _ in range(self.hp["N_islands"]):
            # print(f"current{_}")
            pop = self._evolve_one_island([], query, reference_info)
            for c in pop:
                if c["score"] == 0.0:
                    return c["raw"], self.usage, {"island": _, "place": "init"}
                if c["score"] > best_score:
                    best_raw, best_score = c["raw"], c["score"]
            islands.append(pop)

        # Evolution loop
        for gen in range(1, self.hp["N_gens"]):
            # print(f"gen:{gen}")
            for i in range(self.hp["N_islands"]):
                islands[i] = self._evolve_one_island(islands[i], query, reference_info)

                # Early-stop and update best
                for c in islands[i]:
                    if c["score"] == 0.0:
                        return c["raw"], self.usage, {"gen": gen, "island": i}
                    if c["score"] > best_score:
                        best_raw, best_score = c["raw"], c["score"]

                # Immediate cyclic migration
                self._migrate_after_island(islands, i)

            # Periodic reset
            if gen % self.hp["N_reset_interval"] == 0:
                self._reset_islands(islands)

        return best_raw, self.usage, self.hp["N_gens"]*self.hp["N_islands"]*self.hp["N_convs"]*self.hp["N_seq"]

if __name__ == "__main__":
    from datasets import load_dataset
    import os, sys

    # base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # sys.path.insert(0, base_dir)

    data = load_dataset('osunlp/TravelPlanner', 'validation')['validation']
    first = data[-1]
    reference_information = first['reference_information']
    llm_cfg = {
        "model_type": "dsv3",
        "model_name": "deepseek-v3",
        "temperature": 1.0,
        "max_tokens": 4096
    }
    planner = MindEvolutionPlanner(task_name='TravelPlanner', llm_config=llm_cfg)
    result = planner.run(query=first, reference_info=reference_information)
    print("Generated Plan:", result)