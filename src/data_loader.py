import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

def load_dataset(path: str, batch_size: int = 32):
    df = pd.read_csv(path)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, scaler, X_tensor, df.columns