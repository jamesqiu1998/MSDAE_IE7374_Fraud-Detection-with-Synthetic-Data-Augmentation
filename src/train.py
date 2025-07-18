import os
import yaml
import torch
from torch import optim
from src.vae import VAE
from utils.helpers import vae_loss
from src.data_loader import load_dataset

def train_model(config):

    # Load dataset
    dataloader, _, X_tensor, _ = load_dataset(config["data_path"])
    input_dim = X_tensor.shape[1]
    latent_dim = config["latent_dim"]

    # Model and optimizer
    model = VAE(input_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    epochs = config["epochs"]
    losses = []

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x_batch = batch[0]
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x_batch)
            loss = vae_loss(recon_x, x_batch, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader.dataset)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # Save model
    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "outputs/vae_model.pth")

if __name__ == "__main__":
    train()