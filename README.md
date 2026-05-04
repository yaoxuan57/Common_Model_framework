# UniFault Foundation Model 

This repository provides a PyTorch-based training pipeline for fine-tuning 
the UniFault Foundation model on time-series data.

---

## 📁 Directory Structure

```
.
├── fine_tune.py         # Main fine-tuning script
├── model/
│   └── model.py                # Main Model (SSL-training and fine-tuning) implementation
│   └── Transformer_utils.py    # Transformer model implementation
├── datalaoders/
│   └── pretraining_dataloader.py   # Data loading during SSL-based pretraining
│   └── train_dataloader.py         # Data loading in supervised or fine-tuning
├── utils.py                # Helper functions
├── checkpoints/            # Output directory for checkpoints and metrics
├── lightning_logs/         # Directory to store pretrained model checkpoints
```

---

## 🚀 Quick Start

### 🧪 Supervised Training From Scratch (Start from randomly initialized weights)

```bash
python fine_tune.py --load_from_pretrained False
```

### ♻️ Fine-Tune a Pretrained Model

```bash
python fine_tune.py \
  --load_from_pretrained True \
  --pretrained_model_dir <your_pretrain_dir> \
  --pretraining_epoch_id <epoch_id>
```

---

## ⚙️ Command-Line Arguments
### General Args
| Argument                 | Description                                                  | Default   |
| ------------------------ |--------------------------------------------------------------|-----------|
| `--data_path`            | Path to dataset                                              | -         |
| `--data_id`              | Identifier for the dataset variant                           | Ex: `IMS` |
| `--data_percentage`      | Fraction of data used (e.g., `1` = 1% and `100` = full data) | `1`       |
| `--gpu_id`               | GPU device ID                                                | `0`       |


### Model Args
| Model Argument | Description                                                  | Note                                  |
|----------------|--------------------------------------------------------------|---------------------------------------|
| `--model_id`   | Identifier for model variant (used in naming logs)           | Change each run                       |
| `--embed_dim`  | Embedding dimension for Transformer                          | Change for (Tiny-Small-Base) variants |
| `--heads`      | Number of attention heads                                    | Change for (Tiny-Small-Base) variants |
| `--depth`      | Number of transformer blocks                                 | Change for (Tiny-Small-Base) variants |
| `--patch_size` | Patch size for ViT input                                     | KEEP FIXED                            |
| `--dropout`    | Dropout rate                                                 | KEEP FIXED                                 |

### Loading from a pretrained model
| Argument                 | Description                                                                                                                                |
| ------------------------ |--------------------------------------------------------------------------------------------------------------------------------------------|
| `--load_from_pretrained` | Boolean flag to use pretrained weights                                                                                                     |
| `--pretrained_model_dir` | Directory containing pretrained checkpoint (No need to add checkpoint name -- Just its directory)                                          |
| `--pretraining_epoch_id` | Epoch number of pretrained checkpoint (because we save all the pretraining epochs, so this one is to select which one do you want to load) |

| Argument                 | Description                                                  | Default  |
| ------------------------ |--------------------------------------------------------------| -------- |
| `--num_epochs`           | Number of training epochs                                    | `10`     |
| `--batch_size`           | Batch size                                                   | `64`     |
| `--lr`                   | Learning rate                                                | `3e-4`   |
| `--wt_decay`             | Weight decay                                                 | `1e-4`   |
| `--random_seed`          | Seed for reproducibility                                     | `42`     |

---

## 📝 Outputs

After training, the following files are generated in `checkpoints/<run_name>/`:

* `confusion_matrix.png` – Visualization of model predictions
* `classification_report.txt` – Detailed precision/recall/F1 report
* `loss.png`, `accuracy.png` – Training and validation curves
* `best.ckpt` – Best checkpoint based on validation F1 score

---

## 🧠 How Pretrained Loading Works

If `--load_from_pretrained` is set:

* A checkpoint is loaded from `lightning_logs/<pretrained_model_dir>/pretrain-epoch=<id>.ckpt`
* Only layers with matching names and sizes are loaded (via filtered `state_dict`)

---

## 🧪 Example for Reproducibility

Train from scratch:

```bash
python fine_tune.py \
  --data_path "/path/to/your/data" \
  --data_id "YourDatasetID" \
  --model_id "scratch_run" \
  --load_from_pretrained False \
  --num_epochs 20
```

Fine-tune a model:

```bash
python fine_tune.py \
  --load_from_pretrained True \
  --pretrained_model_dir PRETRAIN_BASE_dim256_depth8 \
  --pretraining_epoch_id 1 \
  --model_id "finetune_run"
```