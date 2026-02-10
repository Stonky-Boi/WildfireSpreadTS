import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataloader.FireSpreadDataset import FireSpreadDataset
from src.dataloader.FireSpreadDataModule import FireSpreadDataModule

# -----------------------------
# Model
# -----------------------------

class TemporalViTPixel(nn.Module):
    """
    Pixel-wise temporal Vision Transformer.
    Each pixel is treated as a patch sequence over time.
    """

    def __init__(
        self,
        n_channels: int,
        temporal_length: int,
        embedding_dimension: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        patch_size: int = 1
    ):
        super().__init__()

        self.patch_size = patch_size
        self.temporal_length = temporal_length

        # patch embedding (here patch_size=1 for pixel-wise)
        self.input_projection = nn.Linear(n_channels * patch_size * patch_size, embedding_dimension)

        # learnable positional embedding along temporal dimension
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, temporal_length, embedding_dimension)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dimension,
            nhead=num_heads,
            dim_feedforward=4 * embedding_dimension,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # output projection back to single channel
        self.output_projection = nn.Linear(embedding_dimension, 1)

        nn.init.trunc_normal_(self.positional_embedding, std=0.02)

    def forward(self, x):
        """
        x: (B, T, C, H, W)
        return: (B, 1, H, W)
        """
        B, T, C, H, W = x.shape

        # rearrange to (B, H*W, T, C)
        x = x.permute(0, 3, 4, 1, 2).reshape(B, H * W, T, C)

        # project patches
        x = self.input_projection(x)  # (B, H*W, T, D)
        x = x + self.positional_embedding  # broadcast (1, T, D)

        # reshape for transformer: (B*H*W, T, D)
        x = x.reshape(B * H * W, T, -1)
        x = self.transformer(x)

        # temporal pooling
        x = x.mean(dim=1)  # (B*H*W, D)

        # project to output
        x = self.output_projection(x)  # (B*H*W, 1)

        # reshape back to image
        x = x.reshape(B, H, W, 1).permute(0, 3, 1, 2)  # (B, 1, H, W)
        return x

# -----------------------------
# Training script
# -----------------------------

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- Dataset config --------
    data_dir = "/home/arnav/WildfireSpread/hdf5_tiny/"
    batch_size = 1
    num_workers = 4

    n_leading_observations = 3
    crop_side_length = 64
    load_from_hdf5 = True
    remove_duplicate_features = False
    features_to_keep = None
    return_doy = False
    data_fold_id = 0

    # -------- Build datasets --------
    train_years, val_years, test_years = FireSpreadDataModule.split_fires(data_fold_id)

    train_dataset = FireSpreadDataset(
        data_dir=data_dir,
        included_fire_years=train_years,
        n_leading_observations=n_leading_observations,
        n_leading_observations_test_adjustment=None,
        crop_side_length=crop_side_length,
        load_from_hdf5=load_from_hdf5,
        is_train=True,
        remove_duplicate_features=remove_duplicate_features,
        features_to_keep=features_to_keep,
        return_doy=return_doy,
        stats_years=train_years
    )

    val_dataset = FireSpreadDataset(
        data_dir=data_dir,
        included_fire_years=val_years,
        n_leading_observations=n_leading_observations,
        n_leading_observations_test_adjustment=None,
        crop_side_length=crop_side_length,
        load_from_hdf5=load_from_hdf5,
        is_train=False,
        remove_duplicate_features=remove_duplicate_features,
        features_to_keep=features_to_keep,
        return_doy=return_doy,
        stats_years=train_years
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # -------- Infer channels --------
    sample_batch = train_dataset[0]
    sample_x = sample_batch[0]
    n_channels = sample_x.shape[1]

    # -------- Model --------
    model = TemporalViTPixel(
        n_channels=n_channels,
        temporal_length=n_leading_observations,
        embedding_dimension=64,
        num_layers=2,  # reduced for smaller memory footprint
        num_heads=2
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            x = x.to(device)
            y = y.to(device).unsqueeze(1).float()

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                x = x.to(device)
                y = y.to(device).unsqueeze(1).float()
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")


if __name__ == "__main__":
    main()