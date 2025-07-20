import torch
import pandas as pd
import yaml
from src.vae import VAE
from src.data_loader import load_dataset

def generate_synthetic_data(config_path: str, model_path: str, n_samples: int = 500):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    input_dim = config["input_dim"]
    latent_dim = config["latent_dim"]

    # Load trained model
    model = VAE(input_dim, latent_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Generate synthetic samples
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim)
        synthetic = model.decode(z).numpy()

    # Inverse scale
    _, scaler,_, df = load_dataset(config["data_path"], config["batch_size"], return_df=True)
    synthetic_original = scaler.inverse_transform(synthetic)
    columns = df.drop("Class", axis=1).columns

    synthetic_df = pd.DataFrame(synthetic_original, columns=columns)
    synthetic_df["Class"] = 1  # label as fraud
    return synthetic_df
