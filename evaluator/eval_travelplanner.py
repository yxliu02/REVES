import argparse
from tqdm import tqdm
from datasets import load_dataset
from evaluator.TravelPlanner.commonsense_constraint import evaluation as commonsense_eval
from evaluator.TravelPlanner.hard_constraint import evaluation as hard_eval
from utils.utils import load_json, save_json
import os

def check_if_pass(return_info: dict):
    if return_info == None:
        return False
    for key in return_info:
        if return_info[key][0] == False:
            return False
    return True

def score(commonsense_info_box, hard_info_box):
    if commonsense_info_box is None:
        return -500.0
    elif hard_info_box is None:
        return -100.0
    else:
        score = 0.0
        for key in commonsense_info_box:
            if commonsense_info_box[key][0] == False:
                score = score - 1
        for key in hard_info_box:
            if hard_info_box[key][0] == False:
                score = score - 1
        return score

def evaluate_plan(query: dict, plan: list) -> dict:
    """
    Evaluate a single travel plan against commonsense and hard constraints.
    Assumes 'plan' and 'query["local_constraint"]' are already proper structures.
    """
    if not plan:
        commonsense_info_box = None
        hard_info_box = None
    else:
        commonsense_info_box = commonsense_eval(query, plan)
        if commonsense_info_box['is_not_absent'][0] and commonsense_info_box['is_valid_information_in_sandbox'][0]:
            hard_info_box = hard_eval(query, plan)
        else:
            hard_info_box = None

    if check_if_pass(commonsense_info_box) and check_if_pass(hard_info_box):
        assert score(commonsense_info_box, hard_info_box) == 0.0

    return {
        'commonsense_pass': check_if_pass(commonsense_info_box),
        'hard_pass':       check_if_pass(hard_info_box),
        'commonsense_details': commonsense_info_box,
        'hard_details': hard_info_box,
        'score': score(commonsense_info_box, hard_info_box)
    }


def main(file_path: str) -> None:
    dataset = load_dataset('osunlp/TravelPlanner', 'validation')['validation']
    records = load_json(file_path)
    query_data_list = [x for x in dataset]
    assert len(records) == len(query_data_list), "Query and record counts differ."

    empty_count = 0
    pass_count = 0

    for record, query in tqdm(zip(records, query_data_list), total=len(records), desc="Evaluating plans"):
        plan = record.get('plan', [])

        if type(query) == str:
            query = eval(query)
        if type(record) == str:
            record = eval(record)
        if type(query['local_constraint']) == str:
            query['local_constraint'] = eval(query['local_constraint'])

        result = evaluate_plan(query, plan)
        if not plan:
            empty_count += 1
        record.update(result)
        if result['commonsense_pass'] and result['hard_pass']:
            pass_count += 1

    base, ext = os.path.splitext(file_path)
    output_path = f"{base}_annotated{ext}"
    save_json(records, output_path)

    print(f"Total entries: {len(records)}")
    print(f"Empty plans: {empty_count}")
    print(f"Pass count (commonsense+hard): {pass_count}")
    print(f"Annotated file saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Annotate validation JSON with constraint results.")
    parser.add_argument('--file_path', help="Path to JSON/JSONL file with plans.")
    args = parser.parse_args()
    main(args.file_path)
