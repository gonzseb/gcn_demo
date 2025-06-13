import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from train import train_model
from evaluate import evaluate

def main():
    """
    Aquí entrenamos el modelo GCN y luego lo se evalúa mostrando resultados en consola y gráficamente.
    """
    print("🦾 Entrenando modelo GCN...")
    model, data = train_model()
    print("✅ Modelo entrenado. Evaluando...")
    evaluate(model, data)

if __name__ == "__main__":
    main()
    