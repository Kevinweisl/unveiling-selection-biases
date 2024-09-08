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
    'mmlu_abstract_algebra',
    'mmlu_anatomy',
    'mmlu_astronomy',
    'mmlu_business_ethics',
    'mmlu_clinical_knowledge',
    'mmlu_college_biology',
    'mmlu_college_chemistry',
    'mmlu_college_computer_science',
    'mmlu_college_mathematics',
    'mmlu_college_medicine',
    'mmlu_college_physics',
    'mmlu_computer_security',
    'mmlu_conceptual_physics',
    'mmlu_econometrics',
    'mmlu_electrical_engineering',
    'mmlu_elementary_mathematics',
    'mmlu_formal_logic',
    'mmlu_global_facts',
    'mmlu_high_school_biology',
    'mmlu_high_school_chemistry',
    'mmlu_high_school_computer_science',
    'mmlu_high_school_european_history',
    'mmlu_high_school_geography',
    'mmlu_high_school_government_and_politics',
    'mmlu_high_school_macroeconomics',
    'mmlu_high_school_mathematics',
    'mmlu_high_school_microeconomics',
    'mmlu_high_school_physics',
    'mmlu_high_school_psychology',
    'mmlu_high_school_statistics',
    'mmlu_high_school_us_history',
    'mmlu_high_school_world_history',
    'mmlu_human_aging',
    'mmlu_human_sexuality',
    'mmlu_international_law',
    'mmlu_jurisprudence',
    'mmlu_logical_fallacies',
    'mmlu_machine_learning',
    'mmlu_management',
    'mmlu_marketing',
    'mmlu_medical_genetics',
    'mmlu_miscellaneous',
    'mmlu_moral_disputes',
    'mmlu_moral_scenarios',
    'mmlu_nutrition',
    'mmlu_philosophy',
    'mmlu_prehistory',
    'mmlu_professional_accounting',
    'mmlu_professional_law',
    'mmlu_professional_medicine',
    'mmlu_professional_psychology',
    'mmlu_public_relations',
    'mmlu_security_studies',
    'mmlu_sociology',
    'mmlu_us_foreign_policy',
    'mmlu_virology',
    'mmlu_world_religions'
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
