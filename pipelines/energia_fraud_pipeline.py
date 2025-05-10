import kfp
from kfp import dsl


# 1. Carregar funções reais dos componentes
from pipelines.components.ingest.ingest import ingest_data
from pipelines.components.preprocess.preprocess import preprocess_data
from pipelines.components.train.train import train_model



# 2. Definir o pipeline (igual ao anterior, mas agora com funções reais)
@dsl.pipeline(
    name="energia-fraud-detection",
    description="Pipeline completo para detecção de fraudes em energia"
)
def energia_fraud_pipeline(
    input_path: str = "/data/raw/dados_medidores_energia.csv",
    processed_path: str = "/data/processed/",
    model_path: str = "/model/isolation_forest.joblib"
):
    # Passos do pipeline
    ingest_task = ingest_data(
        input_path=input_path,
        output_path=processed_path
    )
    
    preprocess_task = preprocess_data(
        input_path=ingest_task.outputs["output_file"],
        output_path=processed_path
    )
    
    train_model(
        input_path=preprocess_task.outputs["output_file"],
        model_path=model_path
    )

# 4. Compilar o pipeline
if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=energia_fraud_pipeline,
        package_path="energia_fraud_pipeline.yaml"
    )