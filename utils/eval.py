import numpy as np


def evaluate_by_section(evaluator, gt_df, model_df):
    """
    Avalia os custos de forma agregada por sectionName.
    
    Agrupa os DataFrames por 'sectionName', soma os valores de 'rowTotalCostUsd' para cada
    seção e retorna o score calculado pelo evaluator sobre os arrays agregados.
    """
    gt_group = gt_df.groupby("sectionName")["rowTotalCostUsd"].sum()
    model_group = model_df.groupby("sectionName")["rowTotalCostUsd"].sum()
    all_sections = set(gt_group.index).union(set(model_group.index))
    gt_vals = []
    model_vals = []
    for section in all_sections:
        gt_val = gt_group.get(section, 0)
        model_val = model_group.get(section, 0)
        gt_vals.append(gt_val)
        model_vals.append(model_val)
    gt_array = np.array(gt_vals)
    model_array = np.array(model_vals)
    return evaluator.evaluate(gt_array, model_array)

def evaluate_by_section_per_section(evaluator, gt_df, model_df):
    """
    Avalia os custos por sectionName.
    
    Para cada sectionName, soma os valores de 'rowTotalCostUsd' do ground truth e do modelo e 
    retorna o score calculado pelo evaluator. Retorna um dicionário com cada sectionName e seu score.
    """
    gt_group = gt_df.groupby("sectionName")["rowTotalCostUsd"].sum()
    model_group = model_df.groupby("sectionName")["rowTotalCostUsd"].sum()
    all_sections = set(gt_group.index).union(set(model_group.index))
    section_scores = {}
    for section in all_sections:
        gt_val = gt_group.get(section, 0)
        model_val = model_group.get(section, 0)
        # Passamos arrays de um elemento para manter a interface do evaluator
        score = evaluator.evaluate(np.array([gt_val]), np.array([model_val]))
        section_scores[section] = score
    return section_scores