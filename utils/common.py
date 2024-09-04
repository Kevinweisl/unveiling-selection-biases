import argparse
from pathlib import Path

from llm_tool.config import MODEL_NAME_MAPPING


ARTIFACTS_DIR = Path('artifacts')
ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
ANALYSIS_DIR = ARTIFACTS_DIR / 'RQ1' / 'analysis'
ANALYSIS_DIR.mkdir(exist_ok=True, parents=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks", default=None
    )
    parser.add_argument(
        "--model", default='palm2', type=str, choices=list(MODEL_NAME_MAPPING.keys())
    )
    parser.add_argument(
        "--choice_symbol", default='original', type=str, choices=['original', 'reversed']
    )
    parser.add_argument(
        "--get_val", default=False, type=bool, choices=[True, False]
    )

    return parser.parse_args()

mmlu_tasks = [
    'hendrycksTest-abstract_algebra',
    'hendrycksTest-anatomy',
    'hendrycksTest-astronomy',
    'hendrycksTest-business_ethics',
    'hendrycksTest-clinical_knowledge',
    'hendrycksTest-college_biology',
    'hendrycksTest-college_chemistry',
    'hendrycksTest-college_computer_science',
    'hendrycksTest-college_mathematics',
    'hendrycksTest-college_medicine',
    'hendrycksTest-college_physics',
    'hendrycksTest-computer_security',
    'hendrycksTest-conceptual_physics',
    'hendrycksTest-econometrics',
    'hendrycksTest-electrical_engineering',
    'hendrycksTest-elementary_mathematics',
    'hendrycksTest-formal_logic',
    'hendrycksTest-global_facts',
    'hendrycksTest-high_school_biology',
    'hendrycksTest-high_school_chemistry',
    'hendrycksTest-high_school_computer_science',
    'hendrycksTest-high_school_european_history',
    'hendrycksTest-high_school_geography',
    'hendrycksTest-high_school_government_and_politics',
    'hendrycksTest-high_school_macroeconomics',
    'hendrycksTest-high_school_mathematics',
    'hendrycksTest-high_school_microeconomics',
    'hendrycksTest-high_school_physics',
    'hendrycksTest-high_school_psychology',
    'hendrycksTest-high_school_statistics',
    'hendrycksTest-high_school_us_history',
    'hendrycksTest-high_school_world_history',
    'hendrycksTest-human_aging',
    'hendrycksTest-human_sexuality',
    'hendrycksTest-international_law',
    'hendrycksTest-jurisprudence',
    'hendrycksTest-logical_fallacies',
    'hendrycksTest-machine_learning',
    'hendrycksTest-management',
    'hendrycksTest-marketing',
    'hendrycksTest-medical_genetics',
    'hendrycksTest-miscellaneous',
    'hendrycksTest-moral_disputes',
    'hendrycksTest-moral_scenarios',
    'hendrycksTest-nutrition',
    'hendrycksTest-philosophy',
    'hendrycksTest-prehistory',
    'hendrycksTest-professional_accounting',
    'hendrycksTest-professional_law',
    'hendrycksTest-professional_medicine',
    'hendrycksTest-professional_psychology',
    'hendrycksTest-public_relations',
    'hendrycksTest-security_studies',
    'hendrycksTest-sociology',
    'hendrycksTest-us_foreign_policy',
    'hendrycksTest-virology',
    'hendrycksTest-world_religions'
]

all_tasks = mmlu_tasks + ['openbookqa', 'arc_challenge', 'mathqa', 'winogrande', 'hellaswag']

category_subcategory_mapping = {
    'abstract_algebra': 'STEM',
    'anatomy': 'STEM',
    'astronomy': 'STEM',
    'business_ethics': 'Other',
    'clinical_knowledge': 'Other',
    'college_biology': 'STEM',
    'college_chemistry': 'STEM',
    'college_computer_science': 'STEM',
    'college_mathematics': 'STEM',
    'college_medicine': 'Other',
    'college_physics': 'STEM',
    'computer_security': 'STEM',
    'conceptual_physics': 'STEM',
    'econometrics': 'Social Sciences',
    'electrical_engineering': 'STEM',
    'elementary_mathematics': 'STEM',
    'formal_logic': 'Humanities',
    'global_facts': 'Other',
    'high_school_biology': 'STEM',
    'high_school_chemistry': 'STEM',
    'high_school_computer_science': 'STEM',
    'high_school_european_history': 'Humanities',
    'high_school_geography': 'Social Sciences',
    'high_school_government_and_politics': 'Social Sciences',
    'high_school_macroeconomics': 'Social Sciences',
    'high_school_mathematics': 'STEM',
    'high_school_microeconomics': 'Social Sciences',
    'high_school_physics': 'STEM',
    'high_school_psychology': 'Social Sciences',
    'high_school_statistics': 'STEM',
    'high_school_us_history': 'Humanities',
    'high_school_world_history': 'Humanities',
    'human_aging': 'Other',
    'human_sexuality': 'Social Sciences',
    'international_law': 'Humanities',
    'jurisprudence': 'Humanities',
    'logical_fallacies': 'Humanities',
    'machine_learning': 'STEM',
    'management': 'Other',
    'marketing': 'Other',
    'medical_genetics': 'Other',
    'miscellaneous': 'Other',
    'moral_disputes': 'Humanities',
    'moral_scenarios': 'Humanities',
    'nutrition': 'Other',
    'philosophy': 'Humanities',
    'prehistory': 'Humanities',
    'professional_accounting': 'Other',
    'professional_law': 'Humanities',
    'professional_medicine': 'Other',
    'professional_psychology': 'Social Sciences',
    'public_relations': 'Social Sciences',
    'security_studies': 'Social Sciences',
    'sociology': 'Social Sciences',
    'us_foreign_policy': 'Social Sciences',
    'virology': 'Other',
    'world_religions': 'Humanities'
}
