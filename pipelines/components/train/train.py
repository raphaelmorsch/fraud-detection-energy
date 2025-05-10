import argparse
import joblib
import pandas as pd
import os
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import mlflow
from kfp.v2.dsl import component

@component(
    base_image='python:3.9-slim',
    packages_to_install=['scikit-learn', 'joblib', 'mlflow', 'pandas', 'pyarrow']
)
def train_model(input_path: str, model_path: str):
    """
    Função real para treinamento:
    - Carrega dados processados
    - Separa em treino/teste
    - Treina Isolation Forest
    - Registra métricas no MLflow
    - Salva o modelo treinado
    """
    print(f"[TRAIN] Carregando dados de {input_path}")
    
    df = pd.read_parquet(input_path)
    
    # Separa features e target
    X = df[['consumo_kwh', 'variação_consumo', 'tensao_media', 'consumo_tensao_ratio']]
    y = df['fraude']
    
    # Separa em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size= 0.2, random_state=42
    )
    
    # Configuração do modelo
    model = IsolationForest(
        n_estimators=150,
        max_samples='auto',
        contamination=0.05,
        random_state=42,
        verbose=1
    )
    
    print("[TRAIN] Iniciando treinamento...")
    model.fit(X_train)
    
    # Avaliação
    train_score = model.score_samples(X_train).mean()
    test_score = model.score_samples(X_test).mean()
    
    print(f"[TRAIN] Train score: {train_score:.4f}")
    print(f"[TRAIN] Test score: {test_score:.4f}")
    
    # Registro no MLflow
    with mlflow.start_run():
        mlflow.log_param("n_estimators", 150)
        mlflow.log_metric("train_score", train_score)
        mlflow.log_metric("test_score", test_score)
        mlflow.sklearn.log_model(model, "model")
        
        # Salva o modelo também localmente
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
    
    print(f"[TRAIN] Modelo salvo em {model_path}")
    return model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    
    model_path = train_model(args.input, args.model)
    print(f"Modelo treinado salvo em: {model_path}")