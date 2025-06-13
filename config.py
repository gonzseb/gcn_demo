# Hiperparámetros y configuraciones
class Config:
    # Dataset
    dataset_name = "KarateClub"
    
    # Modelo
    hidden_channels = 4
    learning_rate = 0.01
    epochs = 100
    
    # Visualización
    colors = {
        0: "red",     # LDA: rojo
        1: "yellow"   # CSH: amarillo
    }