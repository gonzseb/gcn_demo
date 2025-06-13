import torch
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    """
    Esta clase define la arquitectura del modelo GCN.
    Se compone de dos capas de convolución sobre grafos (GCNConv).
    La primera capa aplica ReLU como función de activación.
    La segunda capa produce las probabilidades para cada clase usando softmax.
    """
    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
    
    def forward(self, x, edge_index):
        """
        Este método define el paso hacia adelante del modelo.
        Recibe las características de los nodos y la estructura del grafo (aristas).
        Devuelve una matriz de probabilidades por clase para cada nodo.
        """
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return torch.softmax(x, dim=1)
