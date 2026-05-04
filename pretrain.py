import argparse
import datetime
import math
import os
import pytorch_lightning as L
import torch
import torch.optim as optim
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.callbacks import Callback

from model.model import Transformer_bkbone
from model.mae import MaskedAutoencoderTimeSeries
from datalaoders.pretraining_dataloader import get_datasets
from utils import save_copy_of_files, NTXentLoss, str2bool


class create_model(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        if args.train_strategy == "contrastive":
            self.model = Transformer_bkbone(args)
        else:
            self.model = MaskedAutoencoderTimeSeries(args)
        # self.criterion = torch.nn.MSELoss()
        self.contrastive_criterion = NTXentLoss(device='cuda', batch_size=args.batch_size, temperature=0.1,
                                                use_cosine_similarity=True)
        self.args = args

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.wt_decay)
        total_steps = self.args.num_pretrain_epochs * args.tl_length
        num_warmup_steps = int(0.1 * total_steps)
        scheduler = {
            'scheduler': self.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                              num_training_steps=total_steps),
            'name': 'learning_rate',
            'interval': 'step',
            'frequency': 1,
        }
        return [optimizer], [scheduler]

    def get_cosine_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _calculate_loss(self, data, mode="train"):
        if self.args.train_strategy == "contrastive":
            feat1, feat2 = self.model.cl_pretrain(data)
            loss = self.contrastive_criterion(feat1, feat2)
        else:
            loss, pred, mask = self.model(data)

        # Logging for both step and epoch
        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        self.log("train_loss_step", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="val")
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="test")
        return loss

class LossPlotCallback(L.Callback):
    def __init__(self, checkpoint_path):
        self.losses = []
        self.checkpoint_path = checkpoint_path

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Extract loss from outputs
        if isinstance(outputs, dict):
            loss = outputs.get("loss", None)
            if loss is None:
                return  # or raise warning
        else:
            loss = outputs

        if isinstance(loss, torch.Tensor):
            self.losses.append(loss.detach().cpu().item())

    def on_train_end(self, trainer, pl_module):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        plt.plot(self.losses)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.grid(True)
        plt.tight_layout()
        save_path = os.path.join(self.checkpoint_path, "loss_curve.png")
        plt.savefig(save_path)
        print(f"Saved loss curve to {save_path}")
        plt.close()


def pretrain_model(args):
    loss_plot_callback = LossPlotCallback(CHECKPOINT_PATH)
    trainer = L.Trainer(
        default_root_dir=CHECKPOINT_PATH,
        accelerator="auto",
        devices=[args.gpu_id],
        gradient_clip_val=1.0,
        precision='16-mixed',
        max_epochs=PRETRAIN_MAX_EPOCHS,
        callbacks=[
            pretrain_checkpoint_callback,
            LearningRateMonitor("epoch"),
            TQDMProgressBar(refresh_rate=500),
            loss_plot_callback
        ],
    )
    trainer.logger._log_graph = False  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    L.seed_everything(42)  # To be reproducible
    model = create_model(args)
    trainer.fit(model, train_loader, val_loader)

    return model, pretrain_checkpoint_callback.best_model_path


def apply_model_config(args):
    config_map = {
        'tiny': {'embed_dim': 128, 'heads': 4, 'depth': 4, 'project_head_dim': 16},
        'small': {'embed_dim': 256, 'heads': 8, 'depth': 8, 'project_head_dim': 32},
        'base': {'embed_dim': 512, 'heads': 12, 'depth': 16, 'project_head_dim': 16},
    }
    config = config_map[args.model_type]
    for k, v in config.items():
        setattr(args, k, v)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='minmax_prj_head_32', help='')

    parser.add_argument('--data_path', type=str,
                        default=r'/mnt/hdd/fl_server/datasets/new_phm_preprocessing_1ch/min_max_norm/')  # z_score_norm #min_max_norm
    parser.add_argument('--data_ids', nargs='*', default=["CWRU", "JNU", "XJTUSY", "HUST_B", "HITSM_SpectraQuest", "HITSM_self_built", "UNSW",
                                 "TORINO_FD", "UO", "PU", "IMS_FD", "KAIST1", "KAIST2", "KAIST3", "MFPT"])

    parser.add_argument('--include_mixup_files', type=str2bool, default=True)
    parser.add_argument('--mixup_percentage_included', type=int, default=100)

    # Model parameters
    parser.add_argument('--model_type', type=str, choices=['tiny', 'small', 'base'], default='tiny')
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--dropout', type=int, default=0.1)

    parser.add_argument('--train_strategy', type=str, default='contrastive', help='[contrastive or MAE]')

    # MAE parameters
    parser.add_argument('--decoder_embed_dim', type=int, default=128, help='Dec embdedding dimension of the Transformer')
    parser.add_argument('--decoder_depth', type=int, default=2, help='Dec embdedding dimension of the Transformer')
    parser.add_argument('--decoder_num_heads', type=int, default=4, help='Dec embdedding dimension of the Transformer')
    parser.add_argument('--masking_ratio', type=float, default=0.5, help='for the pretraining')

    # Training parameters
    parser.add_argument('--num_pretrain_epochs', type=int, default=5)

    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wt_decay', type=float, default=1e-5)
    parser.add_argument('--gpu_id', type=int, default=0)

    args = parser.parse_args()
    apply_model_config(args)
    DATASET_PATH = args.data_path
    PRETRAIN_MAX_EPOCHS = args.num_pretrain_epochs
    print(f"==== Loading datasets from {DATASET_PATH}")

    # load from checkpoint
    datasets = args.data_ids[0] if len(args.data_ids) == 1 else str(len(args.data_ids)) + "_datasets"
    with_mix = "withMixup" if args.include_mixup_files else "withOUTMixup"
    run_description = f"PRETRAIN_{args.model_type}_{datasets}_{args.model_id}_{args.train_strategy}_{with_mix}"
    run_description += f"_{args.num_pretrain_epochs}ep_bs{args.batch_size}_lr{args.lr}_wd{args.wt_decay}"
    run_description += f"_{datetime.datetime.now().strftime('%H_%M')}"
    print(f"========== {run_description} ===========")
    CHECKPOINT_PATH = f"lightning_logs/{run_description}"

    pretrain_checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
        save_top_k=-1,
        filename='pretrain-{epoch}',
        every_n_epochs=1
    )

    # Save a copy of this file and configs file as a backup
    save_copy_of_files(pretrain_checkpoint_callback)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load datasets ...
    train_loader, val_loader = get_datasets(args)
    print("Dataset loaded ...")

    args.num_classes = args.project_head_dim  # This is the projection head size, not the #classes
    # args.class_names = [str(i) for i in range(args.num_classes)]
    args.num_channels = train_loader.dataset[0].shape[0]
    args.tl_length = len(train_loader)

    pretrained_model, best_model_path = pretrain_model(args)
