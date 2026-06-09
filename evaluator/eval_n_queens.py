import reasoning_gym

def evaluate_n_queens(query: str, answer: str) -> bool:
    dataset = reasoning_gym.create_dataset(
        name="n_queens",
        size=1,
        seed=42,
    )
    score = dataset.score_answer(
        answer=answer,
        entry=query,
    )
    return {"score": 0} if score == 1 else {"score": -1}

