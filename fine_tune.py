import os
import argparse
import datetime
import math

import torch
import pytorch_lightning as pl

import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Qt5Agg')
from sklearn.metrics import classification_report
from torchmetrics import MetricCollection
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from torchmetrics.classification import Accuracy, MulticlassF1Score, MulticlassConfusionMatrix
from torchmetrics.regression import MeanSquaredError
from datalaoders.train_dataloader import get_datasets
from model.model import Transformer_bkbone
from utils import save_copy_of_files, str2bool, get_rul_report, scoring_function_v2

# ==================== Model Wrapper ====================
class Model(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = Transformer_bkbone(args)
        if args.task_type == 'FD':
            self.loss_fn = torch.nn.CrossEntropyLoss()
        elif args.task_type == 'RUL':
            self.loss_fn = torch.nn.MSELoss()
        if args.task_type == 'FD':
            self.train_metrics = MetricCollection({
                "acc": Accuracy(task="multiclass", num_classes=args.num_classes),
                "f1": MulticlassF1Score(num_classes=args.num_classes, average="macro")
            })
            self.val_metrics = MetricCollection({
                "acc": Accuracy(task="multiclass", num_classes=args.num_classes),
                "f1": MulticlassF1Score(num_classes=args.num_classes, average="macro")
            })
            self.test_f1 = MulticlassF1Score(num_classes=args.num_classes, average="macro")
            self.confusion_matrix = MulticlassConfusionMatrix(num_classes=args.num_classes)
        elif args.task_type == 'RUL':
            self.train_metrics = MetricCollection({
                "rmse": MeanSquaredError(squared=False)
            })
            self.val_metrics = MetricCollection({
                "rmse": MeanSquaredError(squared=False)
            })
            self.test_rmse = MeanSquaredError(squared=False)



        self.total_steps = args.num_epochs * args.tl_length
        self.num_warmup_steps = int(0.1 * self.total_steps)  # 2048

        self.test_preds = []
        self.test_targets = []

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.wt_decay)

        scheduler = {
            'scheduler': self.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.num_warmup_steps,
                                                              num_training_steps=self.total_steps),
            'name': 'learning_rate', 'interval': 'step', 'frequency': 1,
        }
        return [optimizer], [scheduler]


    def get_cosine_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _shared_step(self, batch, stage):
        x, y = batch

        if self.args.task_type == "FD":
            # ---- target: force (B,) long ----
            if y.ndim > 1:
                # one-hot (B,K) OR (B,1)
                if y.size(-1) == 1:
                    y = y.view(-1)
                else:
                    y = torch.argmax(y, dim=1)
            y = y.long()

            # ---- forward gives features/tokens ----
            feats = self(x)

            # ---- predict() must output class logits (B,num_classes) ----
            class_logits = self.model.predict(feats)

            # fix the B=1 squeeze case: (num_classes,) -> (1,num_classes)
            if class_logits.ndim == 1:
                class_logits = class_logits.unsqueeze(0)

            # if predict returns (B,1,num_classes) etc, flatten to (B,num_classes)
            if class_logits.ndim > 2:
                class_logits = class_logits.view(class_logits.size(0), -1)

            loss = self.loss_fn(class_logits, y)

            # for metrics use class indices (B,)
            preds = torch.argmax(class_logits, dim=1)

        elif self.args.task_type == "RUL":
            feats = self(x)
            preds = self.model.predict(feats)
            if preds.ndim > 1:
                preds = preds.view(-1)
            y = y.view(-1).float()
            loss = self.loss_fn(preds, y)

        # ---- metrics/logging ----
        if stage == "train":
            self.train_metrics.update(preds, y)
            self.log_dict({f"train_{k}": v for k, v in self.train_metrics.compute().items()},
                        on_epoch=True, prog_bar=True)
            self.log("train_loss", loss, on_epoch=True, prog_bar=True)

        elif stage == "val":
            self.val_metrics.update(preds, y)
            self.log_dict({f"val_{k}": v for k, v in self.val_metrics.compute().items()},
                        on_epoch=True, prog_bar=True)
            self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        elif stage == "test":
            if self.args.task_type == "FD":
                self.test_f1.update(preds, y)
                self.confusion_matrix.update(preds, y)
                self.test_preds.extend(preds.cpu().numpy())
                self.test_targets.extend(y.cpu().numpy())

                acc = Accuracy(task="multiclass", num_classes=self.args.num_classes).to(preds.device)(preds, y)
                self.log("test_accuracy", acc)

            else:
                self.test_rmse.update(preds, y)
                self.test_preds.extend(preds.detach().float().cpu().numpy())
                self.test_targets.extend(y.detach().float().cpu().numpy())
                score = scoring_function_v2(np.array(self.test_preds), np.array(self.test_targets))
                self.log("test_score", score)

            self.log("test_loss", loss)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def on_train_epoch_end(self):
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        if self.args.task_type == 'FD':
            f1_score = self.test_f1.compute()
            self.log("test_f1", f1_score)
            self.test_f1.reset()

            fig, ax = self.confusion_matrix.plot()
            fig.tight_layout()
            fig.savefig(f"{self.args.ckpt_dir}/confusion_matrix.png", bbox_inches="tight")
            print("Test Confusion Matrix saved.")
            self.confusion_matrix.reset()

            labels = list(range(self.args.num_classes))  # [0,1,2,3]
            print("unique y_true:", np.unique(self.test_targets, return_counts=True))
            print("unique y_pred:", np.unique(self.test_preds, return_counts=True))
            print("args.num_classes:", self.args.num_classes)
            print("class_names:", self.args.class_names)

            report = classification_report(
                self.test_targets,
                self.test_preds,
                labels=labels,
                target_names=self.args.class_names,
                digits=4,
                zero_division=0
            )
            print("=== Classification Report ===")
            print(report)

            with open(f"{self.args.ckpt_dir}/classification_report.txt", "w") as f:
                f.write(report)

        elif self.args.task_type == 'RUL':
            rmse = self.test_rmse.compute()
            self.log("test_rmse", rmse)
            self.test_rmse.reset()

            # Calculate final score
            score = scoring_function_v2(np.array(self.test_preds), np.array(self.test_targets))

            # print(self.test_predss)
            # print(self.test_targets)
            # print(score)

            self.log("test_score", score)

            # Save predictions and targets
            np.save(f"{self.args.ckpt_dir}/test_preds.npy", np.array(self.test_preds))
            np.save(f"{self.args.ckpt_dir}/test_targets.npy", np.array(self.test_targets))
            report = f"""=== RUL Prediction Report ===
            Evaluation Metrics:
            - RMSE: {rmse:.8f}
            - Score: {score:.8f}

            Prediction Statistics:
            - Min True RUL: {np.min(self.test_targets):.2f}
            - Max True RUL: {np.max(self.test_targets):.2f}
            - Mean True RUL: {np.mean(self.test_targets):.2f}
            - Std True RUL: {np.std(self.test_targets):.2f}

            - Min Predicted RUL: {np.min(self.test_preds):.2f}
            - Max Predicted RUL: {np.max(self.test_preds):.2f}
            - Mean Predicted RUL: {np.mean(self.test_preds):.2f}
            - Std Predicted RUL: {np.std(self.test_preds):.2f}

            First 10 predictions (True, Predicted):
            """
            for i in range(min(10, len(self.test_targets))):
                report += f"{self.test_targets[i]:.2f}, {self.test_preds[i]:.2f}\n"

            print(report)
            with open(f"{self.args.ckpt_dir}/rul_report.txt", "w") as f:
                f.write(report)
            # Plot predictions vs targets
            plt.figure(figsize=(10, 6))
            plt.scatter(self.test_targets, self.test_preds, alpha=0.5)
            plt.plot([min(self.test_targets), max(self.test_targets)],
                     [min(self.test_targets), max(self.test_targets)], 'r--')
            plt.xlabel('True RUL')
            plt.ylabel('Predicted RUL')
            plt.title(f'RUL Prediction\nRMSE: {rmse:.2f}, Score: {score:.2f}')
            plt.tight_layout()
            plt.savefig(f"{self.args.ckpt_dir}/rul_prediction.png", bbox_inches="tight")
            plt.close()

        self.test_preds = []
        self.test_targets = []


# ==================== Callbacks ====================
def construct_experiment_dir(args):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_description = "FT" if args.load_from_pretrained else "Supervised"
    run_description += f"_{args.model_type}"
    run_description += f"_{args.data_id}_from{args.pretraining_epoch_id}_{args.model_id}"
    run_description += f"_bs{args.batch_size}_lr{args.lr}_seed{args.random_seed}_{timestamp}"
    return run_description


def plot_metrics(metrics, ckpt_dir, task_type):
    plt.figure()
    plt.plot(metrics["train_loss"], label="Train Loss")
    plt.plot(metrics["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.tight_layout()
    plt.savefig(f"{ckpt_dir}/loss.png", bbox_inches="tight")

    plt.figure()
    if task_type == 'FD':
        plt.plot(metrics["train_acc"], label="Train Acc")
        plt.plot(metrics["val_acc"], label="Val Acc")
        plt.legend()
        plt.title("Accuracy Curve")
    elif task_type == 'RUL':
        plt.plot(metrics["train_rmse"], label="Train RMSE")
        plt.plot(metrics["val_rmse"], label="Val RMSE")
        plt.legend()
        plt.title("RMSE Curve")
    plt.tight_layout()
    plt.savefig(f"{ckpt_dir}/performance_metric.png", bbox_inches="tight")


class MetricTrackerCallback(pl.Callback):
    def __init__(self, task_type):
        super().__init__()
        self.task_type = task_type
        self.losses = {"train_loss": [], "val_loss": []}
        if task_type == 'FD':
            self.accuracies = {"train_acc": [], "val_acc": []}
        elif task_type == 'RUL':
            self.rmses = {"train_rmse": [], "val_rmse": []}

    def on_validation_epoch_end(self, trainer, pl_module):
        self.losses["val_loss"].append(trainer.callback_metrics["val_loss"].item())
        if self.task_type == 'FD':
            self.accuracies["val_acc"].append(trainer.callback_metrics["val_acc"].item())
        elif self.task_type == 'RUL':
            self.rmses["val_rmse"].append(trainer.callback_metrics["val_rmse"].item())

    def on_train_epoch_end(self, trainer, pl_module):
        self.losses["train_loss"].append(trainer.callback_metrics["train_loss"].item())
        if self.task_type == 'FD':
            self.accuracies["train_acc"].append(trainer.callback_metrics["train_acc"].item())
        elif self.task_type == 'RUL':
            self.rmses["train_rmse"].append(trainer.callback_metrics["train_rmse"].item())

# ==================== Main ====================
def main(args):
    pl.seed_everything(args.random_seed)
    train_loader, val_loader, test_loader = get_datasets(args)

    # args extracted from the running dataset
    if args.task_type == 'FD':
        args.num_classes = len(np.unique(train_loader.dataset.y_data))
        args.class_names = [str(i) for i in range(args.num_classes)]
    else:
        args.num_classes = 1
    args.seq_len = train_loader.dataset.x_data.shape[-1]
    args.num_channels = train_loader.dataset.x_data.shape[1]
    args.tl_length = len(train_loader)

    # Callbacks
    run_description = construct_experiment_dir(args)
    print(f"========== {run_description} ===========")
    ckpt_dir = f"checkpoints/{run_description}"

    args.ckpt_dir = ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    # Set monitoring metric based on task type
    if args.task_type == 'FD':
        checkpoint = ModelCheckpoint(monitor="train_f1_epoch", mode="max", save_top_k=1, dirpath=ckpt_dir,
                                     filename="best")
        early_stop = EarlyStopping(monitor="train_f1_epoch", patience=args.patience, mode="max")
    elif args.task_type == 'RUL':
        checkpoint = ModelCheckpoint(monitor="val_rmse", mode="min", save_top_k=1, dirpath=ckpt_dir, filename="best")
        early_stop = EarlyStopping(monitor="val_rmse", patience=args.patience, mode="min")

    tracker = MetricTrackerCallback(args.task_type)

    save_copy_of_files(checkpoint)

    model = Model(args)

    # Optional load pretrained weights
    if args.load_from_pretrained and args.pretrained_model_type != 'mae':
        path = os.path.join(args.pretrained_model_dir, f"pretrain-epoch={args.pretraining_epoch_id}.ckpt")
        checkpoint_data = torch.load(path, map_location='cuda', weights_only=False)


        # Filter and count matching keys with the same shape
        matched_weights = {
            k: v for k, v in checkpoint_data['state_dict'].items()
            if k in model.state_dict() and model.state_dict()[k].size() == v.size()
        }

        total_pretrained = len(checkpoint_data['state_dict'])
        model.load_state_dict(matched_weights, strict=False)

        print(f"Loaded pretrained weights from {path}")
        print(f"Matched weights: {len(matched_weights)}/{len(model.state_dict())} model parameters matched "
              f"(from {total_pretrained} pretrained parameters)")
        print("")

    elif args.load_from_pretrained:  #
        path = os.path.join(args.pretrained_model_dir, f"pretrain-epoch={args.pretraining_epoch_id}.ckpt")
        checkpoint_data = torch.load(path, map_location='cuda', weights_only=False)
        checkpoint_state = checkpoint_data['state_dict']
        model_state = model.state_dict()

        remapped_weights = {}
        for ckpt_key, ckpt_value in checkpoint_state.items():
            # Fix the redundant nesting: "model.encoder.encoder." → "model.encoder."
            if ckpt_key.startswith("model.encoder.encoder."):
                new_key = "model.encoder." + ckpt_key[len("model.encoder.encoder."):]
            else:
                new_key = ckpt_key

            # Match if key exists and shape is the same
            if new_key in model_state and model_state[new_key].shape == ckpt_value.shape:
                remapped_weights[new_key] = ckpt_value

        model.load_state_dict(remapped_weights, strict=False)

        print(f"Loaded pretrained weights from {path}")
        print(f"Matched weights: {len(remapped_weights)}/{len(model_state)} model parameters matched "
              f"(from {len(checkpoint_state)} pretrained parameters)")

    trainer = pl.Trainer(
        default_root_dir=ckpt_dir,
        max_epochs=args.num_epochs,
        callbacks=[checkpoint, early_stop, tracker, TQDMProgressBar(refresh_rate=500)],
        accelerator="auto",
        precision='bf16-mixed',
        devices=[args.gpu_id],
        num_sanity_val_steps=0,
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader, ckpt_path="best")

    if args.task_type == 'FD':
        plot_metrics(
            {"train_loss": tracker.losses["train_loss"], "val_loss": tracker.losses["val_loss"],
             "train_acc": tracker.accuracies["train_acc"], "val_acc": tracker.accuracies["val_acc"]},
            args.ckpt_dir,
            args.task_type
        )
    elif args.task_type == 'RUL':
        plot_metrics(
            {"train_loss": tracker.losses["train_loss"], "val_loss": tracker.losses["val_loss"],
             "train_rmse": tracker.rmses["train_rmse"], "val_rmse": tracker.rmses["val_rmse"]},
            args.ckpt_dir,
            args.task_type
        )

def apply_model_config(args):
    config_map = {
        'tiny':  {'embed_dim': 128, 'heads': 4,  'depth': 4},
        'small': {'embed_dim': 256, 'heads': 8,  'depth': 8},
        'base':  {'embed_dim': 512, 'heads': 12, 'depth': 16},
    }
    config = config_map[args.model_type]
    for k, v in config.items():
        setattr(args, k, v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default=r'./dataset/')
    parser.add_argument('--data_id', type=str, default=r'M01', help= 'choose [M01, M02,M03] for FD task and [FEMTO] for RUL task')
    parser.add_argument('--data_percentage', type=str, default="1")
    parser.add_argument('--model_id', type=str, default="CNC_FT", help= 'CNC_FT or FEMTO_FT')

    parser.add_argument('--model_type', type=str, choices=['tiny', 'small', 'base'], default='tiny')
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--use_moe', type=str2bool, default=False, help='[use MoE or default]')

    parser.add_argument('--load_from_pretrained', type=str2bool, default=True)
    parser.add_argument('--pretrained_model_dir', type=str, default="pretrained_models/Tiny")
    parser.add_argument('--pretraining_epoch_id', type=int, default=1)
    parser.add_argument('--pretrained_model_type', type=str, default='normal', help='model can be [normal, mae]')

    parser.add_argument('--num_epochs', type=int, default=600)
    parser.add_argument('--patience', type=int, default=50, help="For early stopping")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4) #1e-3
    parser.add_argument('--wt_decay', type=float, default=1e-4)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--task_type',type=str,default='FD',choices=['FD', 'RUL'])
    args = parser.parse_args()
    apply_model_config(args)
    main(args)

 # Tabular Regression   
# python fine_tune.py --task_type RUL --data_path C:\Users\ngyx\Desktop\Common_Model_framework\tabular_dataset --data_id regression_splits_S3 --data_percentage 1 --model_id regression_FT --model_type tiny --pretrained_model_dir pretrained_models/Tiny --pretraining_epoch_id 1 --batch_size 16 --num_epochs 100 --lr 3e-4 --gpu_id 0

# Tabular Classification
# python fine_tune.py --task_type FD --data_path C:\Users\ngyx\Desktop\Common_Model_framework\tabular_dataset --data_id classification_splits --data_percentage 1 --model_id classification_FT --model_type tiny --pretrained_model_dir pretrained_models/Tiny --pretraining_epoch_id 1 --batch_size 16 --num_epochs 100 --lr 3e-4 --gpu_id 0

# Time Series Regression
# python fine_tune.py --task_type RUL --data_path C:\Users\ngyx\Desktop\Common_Model_framework\time_series_dataset --data_id FEMTO_Regression/splits --data_percentage 1 --model_id FEMTO_FT --model_type tiny --pretrained_model_dir pretrained_models/Tiny --pretraining_epoch_id 1 --batch_size 16 --num_epochs 5 --lr 3e-4 --gpu_id 0