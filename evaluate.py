import torch
from utils.visualize import plot_graph
from config import Config

"""
def evaluate(model, data):
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=1)
    accuracy = (pred == data.y).sum().item() / data.num_nodes
    
    # Visualización
    plot_graph(data, data.y.numpy(), "Grafo Real (LDA=rojo, CSH=amarillo)")
    plot_graph(data, pred.numpy(), "Predicciones del Modelo")
    
    print(f"Accuracy: {accuracy:.2%}")
"""

def evaluate(model, data):
    """
    Esta función evalúa el modelo entrenado.
    Calcula las predicciones, estima la precisión del modelo,
    y visualiza el grafo real junto con el grafo con las predicciones.
    """
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)
    accuracy = (pred == data.y).sum().item() / data.num_nodes
    
    plot_graph(data, data.y.numpy(), "Distribución Real de Clases del Dataset KarateClub\n(CSH = Amarillo, LDA = Rojo)")
    plot_graph(data, pred.numpy(), "Predicciones del Modelo GCN\nTras el Entrenamiento")

    print(f"Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    from model.gcn_model import GCN
    from train import train_model
    
    model, data = train_model()
    evaluate(model, data)
    