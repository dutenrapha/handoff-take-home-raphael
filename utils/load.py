import os
import logging
from adapters.json_adapter import JSONAdapter

def load_all_ground_truths(ground_truth_dir):
    """
    Carrega todos os arquivos de ground truth do diretório informado.
    
    Retorna:
        dict: Mapeia o nome base (sem extensão) para um dicionário contendo:
              - "adapter": instância do JSONAdapter
              - "df": DataFrame extraído da chave 'rows'
              - "total": totalCostUsd se disponível ou a soma dos rowTotalCostUsd
    """
    gt_map = {}
    for file_name in os.listdir(ground_truth_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(ground_truth_dir, file_name)
            try:
                adapter = JSONAdapter(file_path)
                df = adapter.to_dataframe()
            except Exception as e:
                logging.warning(f"Não foi possível carregar {file_path}: {e}")
                continue
            if "totalCostUsd" in adapter.data:
                total = adapter.data["totalCostUsd"]
            else:
                total = df["rowTotalCostUsd"].sum() if "rowTotalCostUsd" in df.columns else None
            base_name = os.path.splitext(file_name)[0]
            gt_map[base_name] = {"adapter": adapter, "df": df, "total": total}
    return gt_map