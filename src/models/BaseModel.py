import math
from abc import ABC
from typing import Any, Literal, Optional, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import wandb
import matplotlib.pyplot as plt
from segmentation_models_pytorch.losses import (DiceLoss, JaccardLoss,
                                                LovaszLoss)
from torchvision.ops import sigmoid_focal_loss


class BaseModel(pl.LightningModule, ABC):
    """Base model class for all models in this project."""
    
    def __init__(
        self,
        n_channels: int,
        flatten_temporal_dimension: bool,
        pos_class_weight: float,
        loss_function: Literal["BCE", "Focal", "Lovasz", "Jaccard", "Dice"],
        use_doy: bool = False,
        required_img_size: Optional[List[int]] = None,
        *args: Any,
        **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        if required_img_size is not None:
            self.hparams.required_img_size = torch.Size(required_img_size)

        if self.hparams.loss_function == "Focal" and self.hparams.pos_class_weight > 1:
            self.hparams.pos_class_weight /= 1 + self.hparams.pos_class_weight

        self.loss = self.get_loss()

        self.train_f1 = torchmetrics.F1Score("binary")
        self.val_f1 = self.train_f1.clone()
        
        # --- Metrics Configuration ---
        # 1. Lightweight metrics: Keep as modules so they live on GPU for fast batch updates
        self.test_f1 = torchmetrics.F1Score("binary")
        self.test_precision = torchmetrics.Precision("binary")
        self.test_recall = torchmetrics.Recall("binary")
        self.test_iou = torchmetrics.JaccardIndex("binary")

        # 2. Heavy metrics buffer
        # Buffer to store predictions on CPU RAM to avoid GPU OOM
        self.test_outputs_buffer = []

    def forward(self, x, doys=None):
        if self.hparams.flatten_temporal_dimension and len(x.shape) == 5:
            x = x.flatten(start_dim=1, end_dim=2)
        return self.model(x)

    def get_pred_and_gt(self, batch):
        if self.hparams.use_doy:
            x, y, doys = batch
        else:
            x, y = batch
            doys = None

        if self.hparams.required_img_size is not None:
            B, T, C, H, W = x.shape
            req_size = tuple(self.hparams.required_img_size)
            
            if x.shape[-2:] != req_size:
                if B != 1:
                    raise ValueError("Not implemented: repeated cropping for batch size > 1.")
                
                H_req, W_req = req_size
                n_H = math.ceil(H / H_req)
                n_W = math.ceil(W / W_req)

                agg_output = torch.zeros(B, H, W, device=self.device)

                for i in range(n_H):
                    for j in range(n_W):
                        if i == n_H - 1:
                            H1, H2 = H - H_req, H
                        else:
                            H1, H2 = i * H_req, (i + 1) * H_req
                        
                        if j == n_W - 1:
                            W1, W2 = W - W_req, W
                        else:
                            W1, W2 = j * W_req, (j + 1) * W_req

                        x_crop = x[:, :, :, H1:H2, W1:W2]
                        agg_output[:, H1:H2, W1:W2] = self(x_crop, doys).squeeze(1)

                return agg_output, y

        y_hat = self(x, doys).squeeze(1)
        return y_hat, y

    def training_step(self, batch, batch_idx):
        y_hat, y = self.get_pred_and_gt(batch)
        loss = self.compute_loss(y_hat, y)
        self.train_f1(y_hat, y)
        self.log("train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_f1", self.train_f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y = self.get_pred_and_gt(batch)
        loss = self.compute_loss(y_hat, y)
        self.val_f1(y_hat, y)
        self.log("val_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        y_hat, y = self.get_pred_and_gt(batch)
        loss = self.compute_loss(y_hat, y)
        
        # 1. Update lightweight metrics (Inputs are on GPU, Metric is on GPU -> OK)
        self.test_f1(y_hat, y)
        self.test_precision(y_hat, y)
        self.test_recall(y_hat, y)
        self.test_iou(y_hat, y)

        # 2. Buffer predictions to CPU for heavy metrics later
        # We assume y_hat is varying size (B, H, W).
        self.test_outputs_buffer.append({
            "y_hat": y_hat.detach().cpu(),
            "y": y.detach().cpu()
        })

        self.log("test_loss", loss.item(), sync_dist=True)

    def on_test_epoch_end(self) -> None:
        # 1. Log accumulated lightweight metrics
        self.log("test_f1", self.test_f1.compute())
        self.log("test_precision", self.test_precision.compute())
        self.log("test_recall", self.test_recall.compute())
        self.log("test_iou", self.test_iou.compute())

        # 2. Compute heavy metrics on CPU
        if not self.test_outputs_buffer:
            return

        print("Aggregating test results for global metrics...")
        
        # FIX: Flatten tensors to 1D to handle varying image sizes across batches
        # y_hat is (B, H, W) -> flatten to (Pixels,)
        # y is (B, H, W) -> flatten to (Pixels,)
        # .float() ensures consistency, .long() for targets
        try:
            all_y_hat = torch.cat([x["y_hat"].flatten().float() for x in self.test_outputs_buffer])
            all_y = torch.cat([x["y"].flatten().long() for x in self.test_outputs_buffer])
            
            self.test_outputs_buffer.clear() # Free memory

            # Instantiate metrics LOCALLY (Default is CPU)
            print("Computing Average Precision and PR Curve on CPU...")
            avg_prec_metric = torchmetrics.AveragePrecision("binary")
            pr_curve_metric = torchmetrics.PrecisionRecallCurve("binary", thresholds=100)
            conf_mat_metric = torchmetrics.ConfusionMatrix("binary")

            # Compute
            ap_score = avg_prec_metric(all_y_hat, all_y)
            self.log("test_AP", ap_score)

            # Plots
            # Confusion Matrix
            cm = conf_mat_metric(all_y_hat, all_y).numpy()
            wandb_table = wandb.Table(data=cm, columns=["PredictedBackground", "PredictedFire"])
            wandb.log({"Test confusion matrix": wandb_table})

            # PR Curve
            pr_curve_metric.update(all_y_hat, all_y)
            fig, ax = pr_curve_metric.plot(score=True)
            wandb.log({"Test PR Curve": wandb.Image(fig)})
            plt.close()

        except Exception as e:
            print(f"Warning: Failed to compute global heavy metrics: {e}")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        x_af = x[:, :, -1, :, :]
        y_hat = self(x).squeeze(1)
        return x_af, y, y_hat

    def get_loss(self):
        if self.hparams.loss_function == "BCE":
            return nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([self.hparams.pos_class_weight], device=self.device))
        elif self.hparams.loss_function == "Focal":
            return sigmoid_focal_loss
        elif self.hparams.loss_function == "Lovasz":
            return LovaszLoss(mode="binary")
        elif self.hparams.loss_function == "Jaccard":
            return JaccardLoss(mode="binary")
        elif self.hparams.loss_function == "Dice":
            return DiceLoss(mode="binary")

    def compute_loss(self, y_hat, y):
        if self.hparams.loss_function == "Focal":
            return self.loss(y_hat, y.float(), alpha=1 - self.hparams.pos_class_weight, gamma=2, reduction="mean")
        else:
            return self.loss(y_hat, y.float())