import streamlit as st
import pandas as pd
import numpy as np
import json

REPORT_GLOBAL_PATH = "reports/20250214_evaluation_report_global.json"
REPORT_BY_SECTION_PATH = "reports/20250214_evaluation_report_by_section.json"

def reorder_columns_by_std_desc(df: pd.DataFrame) -> pd.DataFrame:
    """Reordena as colunas pelo desvio padrão (descendente)."""
    stds = df.std(axis=0, numeric_only=True)
    sorted_cols = stds.sort_values(ascending=False).index
    return df[sorted_cols]

def reorder_columns_by_mean_desc(df: pd.DataFrame) -> pd.DataFrame:
    """Reordena as colunas pela média (descendente)."""
    means = df.mean(axis=0, numeric_only=True)
    sorted_cols = means.sort_values(ascending=False).index
    return df[sorted_cols]

def get_best_model_by_section(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada seção, encontra o modelo com o menor valor da métrica.
    Retorna um DataFrame com três colunas: 'sectionName', 'best_model', 'score'.
    """
    df = df.dropna(subset=["score"])  # Removendo NaN antes de calcular idxmin()

    if df.empty:
        return pd.DataFrame(columns=["sectionName", "best_model", "score"])  # Retorna vazio se não houver dados válidos

    best_models = df.loc[df.groupby("sectionName")["score"].idxmin(), ["sectionName", "model_file", "score"]]
    return best_models.rename(columns={"model_file": "best_model"}).reset_index(drop=True)

def add_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona uma coluna de ranking ao DataFrame, ordenando os modelos do melhor (menor score) para o pior."""
    df = df.dropna(subset=["score"]).copy()  # Remover NaN antes do rankeamento

    if df.empty:
        return df  # Se estiver vazio, retorna como está

    df["ranking"] = df["score"].rank(method="dense", ascending=True).astype(int)
    df = df.sort_values("ranking").reset_index(drop=True)
    return df

def main():
    st.title("Take home Raphael - Handoff")

    with open(REPORT_GLOBAL_PATH, 'r') as f:
        data_global = json.load(f)
    with open(REPORT_BY_SECTION_PATH, 'r') as f:
        data_section = json.load(f)

    df_global = pd.DataFrame(data_global)
    df_section = pd.DataFrame(data_section)

    if df_global.empty or df_section.empty:
        st.error("The JSON files are empty or contain no data.")
        return

    metrics_global = df_global['metric'].unique()
    metrics_section = df_section['metric'].unique()
    all_metrics = sorted(set(metrics_global).union(set(metrics_section)))

    selected_metric = st.selectbox("Select Metric:", all_metrics, index=0)

    df_global_filtered = df_global[df_global["metric"] == selected_metric]
    df_section_filtered = df_section[df_section["metric"] == selected_metric]

    st.subheader(f"Global Table - Metric '{selected_metric}'")
    if df_global_filtered.empty:
        st.warning(f"No data found for metric '{selected_metric}' in evaluation_report_global.json.")
    else:
        df_global_filtered = add_ranking(df_global_filtered)
        st.dataframe(df_global_filtered.reset_index(drop=True))

    st.subheader(f"Table by Section - Metric '{selected_metric}'")
    if df_section_filtered.empty:
        st.warning(f"No data found for metric '{selected_metric}' in evaluation_report_by_section.json.")
    else:
        pivot_by_section = df_section_filtered.pivot_table(
            index='model_file',
            columns='sectionName',
            values='score',
            aggfunc='first'
        ).fillna(np.nan)

        order_choice = st.radio(
            "Order columns by:",
            ("Standard Deviation (desc)", "Mean (desc)"),
            index=0
        )

        if order_choice == "Standard Deviation (desc)":
            pivot_by_section = reorder_columns_by_std_desc(pivot_by_section)
        elif order_choice == "Mean (desc)":
            pivot_by_section = reorder_columns_by_mean_desc(pivot_by_section)

        df_styled = pivot_by_section.style.background_gradient(
            cmap='Blues_r',
            axis=1
        )

        html_table = df_styled.to_html()
        html_container = f"<div style='overflow-x: auto; max-width: 100%; border: 1px solid #333; padding: 10px;'>{html_table}</div>"
        st.markdown(html_container, unsafe_allow_html=True)

        st.subheader(f"Best Model by Section - Metric '{selected_metric}'")
        if df_section_filtered.empty:
            st.warning(f"No data found for metric '{selected_metric}' in evaluation_report_by_section.json.")
        else:
            best_models_df = get_best_model_by_section(df_section_filtered)
            st.dataframe(best_models_df)

if __name__ == "__main__":
    main()
