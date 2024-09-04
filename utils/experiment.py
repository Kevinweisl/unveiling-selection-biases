import itertools
import json

from llm_tool.model import PaLM2Model, GeminiModel, OpenAIModel, LlamaModel

from utils.data import get_instruction_prompt, get_question_prompt, get_choices_prompt, extract_llm_result
from utils.common import RESULTS_DIR


# choice_symbol: original, reversed
def experiment_per_doc(**kwargs):
    task_name = kwargs['task_name']
    model = kwargs['model']
    doc_id = kwargs['doc_id']
    doc = kwargs['doc']
    prompt_type = kwargs['prompt_type']
    choice_symbol = kwargs['choice_symbol']
    validation = kwargs.get('validation', False)
    simplify = kwargs.get('simplify', True)
    temperature = kwargs.get('temperature', 0)
    candidate_count = kwargs.get('candidate_count', 5)

    this_result_dir = RESULTS_DIR / f'{model.model_display_name}/{prompt_type}/{task_name}{f"_val" if validation else ""}/{choice_symbol}/{temperature}/{doc_id}'
    this_result_dir.mkdir(exist_ok=True, parents=True)

    ground_truth = doc['ground_truth']
    permutations = list(itertools.permutations(doc['this_choices']))

    for idx, each_permutation in enumerate(permutations):
        if len(each_permutation) == 2:
            simplify_lst = [0, 1]
        elif len(each_permutation) == 3:
            simplify_lst = [0, 5]
        elif len(each_permutation) == 4:
            simplify_lst = [0, 23]
        elif len(each_permutation) == 5:
            simplify_lst = [0, 119]

        if simplify and idx not in simplify_lst:
            continue

        this_result_path = this_result_dir / f'{doc_id}_{idx}.json'
        if this_result_path.exists():
            continue

        print(task_name)
        print(f'doc_id: {doc_id}, permutation_id: {idx} ground_truth: {ground_truth}')
        print(f'permutation: {each_permutation}\n====================')
        this_result_idx = each_permutation.index(ground_truth)

        # Generate prompt
        instruction_prompt = get_instruction_prompt(each_permutation, prompt_type)
        question_prompt = get_question_prompt(task_name, doc)
        choices_prompt, symbol_mapping = get_choices_prompt(each_permutation, choice_symbol)
        reversed_symbol_mapping = {v: k for k, v in symbol_mapping.items()}
        prompt = f'{instruction_prompt}\n\n{question_prompt}\n\n{choices_prompt}\n\n'

        result = model.generate(
            temperature=temperature,
            prompt=prompt,
            candidate_count=candidate_count,
            max_output_tokens=1000
        )
        answer_choice = extract_llm_result(result['result'], choice_symbol == 'lowercase')
        answer_index = reversed_symbol_mapping.get(answer_choice)

        output_info = {
            'task': task_name,
            'doc_id': doc_id,
            'permutation_id': idx,
            'prompt': prompt,
            'permutation': each_permutation,
            'ground_truth_text': ground_truth,
            'ground_truth_index': this_result_idx,
            'answer_choice': answer_choice,
            'answer_index': answer_index,
            'answer_text': each_permutation[answer_index] if answer_index is not None else '',
            'symbol_mapping': reversed_symbol_mapping,
            'details': result,
        }
        this_result_path.write_text(json.dumps(output_info, indent=4))


def get_model(model_name):
    if model_name == 'palm2':
        model = PaLM2Model(model_name='palm2')
    elif model_name == 'gemini-pro':
        model = GeminiModel(model_name='gemini-pro')
    elif model_name == 'Llama-2-7b-chat':
        model = LlamaModel(model_name='Llama-2-7b-chat')
    elif model_name == 'Llama-2-13b-chat':
        model = LlamaModel(model_name='Llama-2-13b-chat')
    elif model_name == 'Llama-2-70b-chat':
        model = LlamaModel(model_name='Llama-2-70b-chat')
    elif model_name == 'gpt-3.5-1106':
        model = OpenAIModel(model_name='gpt-3.5-1106')
    else:
        raise RuntimeError('Model not supported')

    return model
