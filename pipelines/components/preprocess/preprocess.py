import pandas as pd
import argparse
import os
from sklearn.preprocessing import StandardScaler
import numpy as np
from kfp.v2.dsl import component, Output, OutputPath
from typing import NamedTuple

@component(
    base_image='python:3.9-slim',
    packages_to_install=['pandas', 'scikit-learn', 'pyarrow', 'numpy']
)
def preprocess_data(input_path: str, output_path: str) -> NamedTuple("Outputs", [("output_file", str)]):
    """
    Função real para pré-processamento:
    - Carrega os dados
    - Trata valores nulos
    - Normaliza features numéricas
    - Engenharia de features (cria novas variáveis)
    """
    print(f"[PREPROCESS] Carregando dados de {input_path}")
    
    df = pd.read_parquet(input_path)
    
    # 1. Tratamento de nulos
    df.fillna({
        'consumo_kwh': df['consumo_kwh'].median(),
        'tensao_media': df['tensao_media'].mean(),
        'variação_consumo': 0
    }, inplace=True)
    
    # 2. Normalização
    scaler = StandardScaler()
    numeric_cols = ['consumo_kwh', 'tensao_media', 'variação_consumo']
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # 3. Engenharia de features
    df['consumo_tensao_ratio'] = df['consumo_kwh'] / (df['tensao_media'] + 1e-6)
    df['consumo_diff'] = df['consumo_kwh'] - df['consumo_medio_historico']
    
    # 4. Filtro de segurança
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Salva os dados processados
    output_file = os.path.join(output_path, "dados_processados.parquet")
    df.to_parquet(output_file, index=False)
    
    print(f"[PREPROCESS] Dados processados salvos em {output_file}")
    return (output_file,)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    output_file = preprocess_data(args.input, args.output)
    print(f"Arquivo processado: {output_file}")