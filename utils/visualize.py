import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from config import Config

def plot_graph(data, colors, title):
    """
    Esta función se encarga de visualizar el grafo.
    Convierte los datos a un grafo de NetworkX, calcula la posición de los nodos,
    asigna colores según las clases, y muestra el grafo con un título.
    Usa un objeto Axes para asegurarse de que el título se muestre correctamente.
    """
    G = to_networkx(data, to_undirected=True)
    pos = nx.spring_layout(G, seed=42)
    node_colors = [Config.colors[color] for color in colors]

    fig, ax = plt.subplots(figsize=(6, 5))
    nx.draw(
        G,
        pos=pos,
        ax=ax,  # <<< Usar ese eje
        node_color=node_colors,
        with_labels=True,
        edge_color="gray",
        node_size=300,
        font_color="black"
    )

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
