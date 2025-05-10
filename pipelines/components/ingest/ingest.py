import pandas as pd
import argparse
import os
from kfp.v2.dsl import component, Output, OutputPath
from typing import NamedTuple

@component(
    base_image='python:3.9-slim',
    packages_to_install=['pandas', 'pyarrow']
)
def ingest_data(input_path: str, 
                output_path: str) -> NamedTuple("Outputs", [("output_file", str)]):
    """
    Função real para ingestão de dados:
    - Lê o CSV de entrada
    - Faz validação básica
    - Salva em formato parquet (mais eficiente)
    """
    print(f"[INGEST] Lendo dados de {input_path}")
    
    # Verifica se o arquivo existe
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {input_path}")
    
    df = pd.read_csv(input_path)
    
    # Validações básicas
    required_columns = ['id_medidor', 'consumo_kwh', 'tensao_media', 'fraude']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Coluna obrigatória faltando: {col}")
    
    # Conversão para formato mais eficiente
    output_file = os.path.join(output_path, "dados_medidores.parquet")
    df.to_parquet(output_file, index=False)
    print(f"[INGEST] Dados salvos em {output_file}")
    
    return (output_file,)  # Retorna o caminho do arquivo processado

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    output_file = ingest_data(args.input, args.output)
    print(f"Saída gerada em: {output_file}")