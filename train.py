from model.gcn_model import GCN
from config import Config
from torch_geometric.datasets import KarateClub
import torch.nn.functional as F
import torch

def train_model():
    """
    Esta función entrena el modelo GCN usando el dataset KarateClub.
    Primero adapta las etiquetas a una clasificación binaria (0 y 1).
    Luego inicializa el modelo con los parámetros definidos en la clase Config.
    Se entrena el modelo usando descenso por gradiente con la función de pérdida de entropía cruzada.
    Al final, devuelve el modelo entrenado y los datos del grafo.
    """
    # Cargar dataset
    dataset = KarateClub()
    data = dataset[0]
    data.y = torch.where(data.y == 0, 0, 1)  # Adaptar etiquetas
    
    # Inicializar modelo
    model = GCN(
        num_features=dataset.num_features,
        hidden_channels=Config.hidden_channels,
        num_classes=2
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)
    
    # Entrenamiento
    for epoch in range(Config.epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
    
    return model, data

if __name__ == "__main__":
    model, data = train_model()
    torch.save(model.state_dict(), "model.pth")