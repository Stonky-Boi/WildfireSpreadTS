import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataloader.FireSpreadDataset import FireSpreadDataset
from src.dataloader.FireSpreadDataModule import FireSpreadDataModule

# -----------------------------
# Model
# -----------------------------

class TemporalTransformerPixel(nn.Module):
    """
    Pixel-wise temporal transformer.
    Each pixel is an independent temporal sequence.
    """

    def __init__(
        self,
        n_channels: int,
        temporal_length: int,
        embedding_dimension: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            n_channels,
            embedding_dimension
        )

        self.positional_embedding = nn.Parameter(
            torch.zeros(1, temporal_length, embedding_dimension)
        )

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

        self.output_projection = nn.Linear(
            embedding_dimension,
            1
        )

        nn.init.trunc_normal_(self.positional_embedding, std=0.02)

    def forward(self, x):
        """
        x: (B, T, C, H, W)
        return: (B, 1, H, W)
        """
        B, T, C, H, W = x.shape

        # (B, H, W, T, C)
        x = x.permute(0, 3, 4, 1, 2)

        # (B*H*W, T, C)
        x = x.reshape(B * H * W, T, C)

        x = self.input_projection(x)
        x = x + self.positional_embedding

        x = self.transformer(x)

        # temporal pooling
        x = x.mean(dim=1)

        x = self.output_projection(x)

        # back to image
        x = x.reshape(B, H, W, 1).permute(0, 3, 1, 2)

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
    train_years, val_years, test_years = FireSpreadDataModule.split_fires(
        data_fold_id
    )

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # -------- Infer channels --------
    sample_batch = train_dataset[0]
    sample_x = sample_batch[0]
    n_channels = sample_x.shape[1]

    # -------- Model --------
    model = TemporalTransformerPixel(
        n_channels=n_channels,
        temporal_length=n_leading_observations,
        embedding_dimension=64,
        num_layers=4,
        num_heads=4
    ).to(device)

    # -------- Loss / Optim --------
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # -------- Training loop --------
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

        # -------- Validation --------
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

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
        )


if __name__ == "__main__":
    main()