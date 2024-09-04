import json
import re

from utils.common import parse_args, ARTIFACTS_DIR
from lm_eval import tasks, utils


def update_task_info(task_name, count):
    output_path = ARTIFACTS_DIR / 'task_info.json'
    if output_path.exists():
        task_info = json.loads(output_path.read_text())
    else:
        task_info = dict()

    if task_name in task_info:
        return

    task_info[task_name] = count
    output_path.write_text(json.dumps(task_info, indent=4))


def get_tasks():
    args = parse_args()
    task_manager = tasks.TaskManager()
    if args.tasks is None:
        task_names = task_manager.all_tasks
    else:
        task_names = utils.pattern_match(args.tasks.split(','), task_manager.all_tasks)
    print(f'Tasks: {task_names}')
    task_dict = tasks.get_task_dict(task_names)

    task_dict_items = [
        (name, task)
        for name, task in task_dict.items()
        if task.has_validation_docs() or task.has_test_docs()
    ]

    target_tasks = []
    for task_name, task in task_dict_items:

        if args.get_val:
            if task_name in ['hellaswag', 'winogrande']:
                task_doc_func = task.train_docs
            else:
                task_doc_func = task.validation_docs
        else:
            if task.has_test_docs():
                task_doc_func = task.test_docs
                task_set = 'test'
            elif task.has_validation_docs():
                task_set = 'val'
                task_doc_func = task.validation_docs
            else:
                raise RuntimeError('Task has neither test_docs nor validation_docs')

        task_docs = list(task_doc_func())
        update_task_info(task_name, len(task_docs))

        target_tasks.append({
            'task': task_name,
            'task_docs': task_docs,
            'args': args,
        })

    return target_tasks, args.get_val


def preprocess_question(task_name, doc):
    if task_name in ['arc_challenge', 'hellaswag', 'mathqa', 'openbookqa']:
        doc['this_question'] = doc['query']
    elif task_name == 'winogrande':
        doc['this_question'] = doc['sentence']
    elif task_name.startswith('hendrycksTest'):
        pattern = r'(.*?)\n([A-G]\. [^\n]+)'
        matches = re.findall(pattern, doc['query'], re.DOTALL)
        for i, pair in enumerate(matches):
            if i == 0:
                doc['this_question'] = pair[0]
    elif task_name.startswith('bigbench'):
        doc['this_question'] = doc['input']

    return doc


def preprocess_choices(task_name, doc):
    # check if the first 3 character of the choice is alphabet, dot, and space. for example, 'A. '
    def check_start_with_alpha(choice):
        return re.match(r'^[A-G]\. ', choice)

    if task_name in ['arc_challenge', 'hellaswag', 'mathqa', 'openbookqa']:
        doc['this_choices'] = doc['choices']
    elif task_name == 'winogrande':
        doc['this_choices'] = [doc['option1'], doc['option2']]
    elif task_name.startswith('hendrycksTest'):
        pattern = r'(.*?)\n([A-F]\. [^\n]+)'
        matches = re.findall(pattern, doc['query'], re.DOTALL)
        doc['this_choices'] = [pair[1][3:] for pair in matches if check_start_with_alpha(pair[1])]
        print(doc['this_choices'])
    elif task_name.startswith('bigbench'):
        doc['this_choices'] = list(doc['target_scores'].keys())

    return doc


def process_ground_truth(task_name, doc):
    if task_name.startswith('bigbench'):
        doc['ground_truth'] = None
        for key, value in doc['target_scores'].items():
            if value == 1:
                if not doc['ground_truth']:
                    doc['ground_truth'] = key
                else:
                    raise RuntimeError('Multiple ground truth found')
    elif task_name == 'winogrande':
        doc['ground_truth'] = doc['this_choices'][int(doc['answer'])-1]
    else:
        doc['ground_truth'] = doc['this_choices'][doc['gold']]

    return doc


def get_instruction_prompt(choices, prompt_type):
    prompt = '''[System]
Please carefully read the following questions and choices. Select the most suitable one. Output your final verdict by strictly following this prompt: '''

    if prompt_type == 'general_instruction':
        prompt += 'Indicate your choice by placing it inside double square brackets, with a single character representing the chosen option. For example, [[<single_character>]].'

    return prompt


def get_question_prompt(task_name, doc):
    question = doc['this_question']
    prompt = f'[The start of question]\n{question}\n[The end of question]'

    return prompt


def get_choices_prompt(choices, choice_symbol):
    num_choices = len(choices)
    if choice_symbol == 'original':
        symbol_mapping = {
            i: chr(ord('A') + i)
            for i in range(num_choices)
        }
    elif choice_symbol == 'reversed':
        symbol_mapping = {
            i: chr(ord('A') + num_choices - 1 - i)
            for i in range(num_choices)
        }

    prompt = ''
    for i in range(len(choices)):
        prompt += f'\n\n[The start of choice {symbol_mapping[i]}]\n{choices[i]}\n[The end of choice {symbol_mapping[i]}]'

    return prompt, symbol_mapping


def extract_llm_result(s, lowercase=False):
    if s is None:
        return None
    if lowercase:
        s_adjusted = s
    else:
        s_adjusted = s.upper()

    double_pattern = re.compile(r'\[\[([^\]]+)\]\]')
    double_matches = double_pattern.findall(s_adjusted)

    single_pattern = re.compile(r'\[([^\]]+)\]')
    single_matches = single_pattern.findall(s_adjusted)

    try:
        candidate = double_matches[0]
        if len(candidate) == 1:
            return candidate
        else:
            return candidate.lower()
    except:
        try:
            candidate = single_matches[0]
            if len(candidate) == 1:
                return candidate
            else:
                return candidate.lower()
        except:
            return None
