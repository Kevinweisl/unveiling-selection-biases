import itertools

from utils.data import get_tasks, preprocess_question, preprocess_choices, process_ground_truth
from utils.experiment import experiment_per_doc, get_model


if __name__ == "__main__":
    tasks, is_validation = get_tasks()
    for task in tasks:
        for doc_id, doc in enumerate(itertools.islice(task["task_docs"], 0, None)):
            doc = preprocess_question(task["task"], doc)
            doc = preprocess_choices(task["task"], doc)
            doc = process_ground_truth(task["task"], doc)

            model = get_model(task['args'].model)

            experiment_config = {
                'task_name': task["task"],
                'model': model,
                'doc_id': doc_id,
                'doc': doc,
                'prompt_type': 'general_instruction',
                'choice_symbol': task['args'].choice_symbol,
                'validation': is_validation,
                'simplify': True,
                'temperature': 0,
                'candidate_count': 1
            }
            experiment_per_doc(**experiment_config)
