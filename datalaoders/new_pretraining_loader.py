import glob

import torch
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import pyarrow.parquet as pq
import torch.nn.functional as F


class StreamingPHMDataset(Dataset):
    def __init__(self, data_file, target_seq_len=1024):
        self.data_file = data_file
        self.target_seq_len = target_seq_len
        parquet_file = pq.ParquetFile(data_file)
        self.row_count = parquet_file.metadata.num_rows

    def __len__(self):
        return self.row_count

    def __getitem__(self, idx):
        # Load only a single row at a time to minimize memory usage
        row = pq.read_table(self.data_file, columns=['samples'], use_threads=False, row_groups=[idx // 10000])
        sample = row.column('samples')[idx % 10000].as_py()
        sample_tensor = torch.tensor(sample, dtype=torch.bfloat16)

        return self.adjust_sequence_length(sample_tensor)

    def adjust_sequence_length(self, x):
        # Example method, adjust according to your own logic
        num_channels, seq_length = x.shape
        window_size = self.target_seq_len // num_channels
        remainder = self.target_seq_len % num_channels

        if self.target_seq_len < num_channels * seq_length:
            sliding_stride = window_size // 2
            splits = []
            for start in range(0, seq_length - window_size + 1, sliding_stride):
                splits.append(x[:, start:start + window_size])
            x_processed = torch.cat(splits, dim=-1)  # 2d still, stack channelwise, channel A/B first window to channel A/B last  window
        else:
            x_processed = x.flatten()  # becomes 1d, first channel ends then start with 2nd channel

        if remainder != 0:
            padding = torch.zeros(remainder, dtype=x_processed.dtype)
            x_processed = torch.cat((x_processed, padding))

        return x_processed.unsqueeze(0)

    def _windowed_chunk_sampling(self, x, seq_length, window_size, remainder):
        """
        Helper function to perform windowed chunk sampling on the input tensor.

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Target tensor.
            num_channels (int): Number of channels in the input tensor.
            seq_length (int): Original sequence length.
            window_size (int): Size of each sliding window.
            remainder (int): Remainder to determine if padding is needed.

        Returns:
            torch.Tensor: Processed version of x.
            torch.Tensor: Corresponding target tensor.
        """
        sliding_stride = window_size // 2
        x_splits = []

        # Create sliding window splits
        for start in range(0, seq_length - window_size + 1, sliding_stride):
            end = start + window_size
            x_splits.append(x[:, :, start:end])

        # Concatenate the splits and replicate targets
        x_processed = torch.cat(x_splits, dim=0).flatten(start_dim=1)

        # Pad the remainder if necessary
        if remainder != 0:
            x_processed = F.pad(input=x_processed, pad=(0, remainder), mode='constant', value=0)

        # Add a new dimension for compatibility
        x_processed = x_processed.unsqueeze(dim=1)

        return x_processed  # from (60, 2, 512) --> shape: (batch, target_Seq_len) (60,1024), channel A followed by channel B

def get_datasets(args):

    train_files = []
    val_files = []
    for subdir in args.data_ids:
        subdir_path = os.path.join(args.data_path, subdir)
        train_file = os.path.join(subdir_path, "train.parquet")
        train_files.append(train_file)

    val_file = os.path.join(subdir_path, "val.parquet")
    val_files.append(val_file)

    if args.include_mixup_files:
        mixup_files = os.path.join(args.data_path, "mixed")
        total_files = os.listdir(mixup_files)
        # total_files = [
        #     # HITSM OK
        #     "mixup_file0_HITSM_KAIST.parquet","mixup_file1_HITSM_MFPT.parquet","mixup_file2_HITSM_IMS.parquet",
        #     "mixup_file3_HITSM_TORINO.parquet","mixup_file4_HITSM_UO.parquet","mixup_file5_HITSM_FEMTO.parquet",
        #     "mixup_file6_HITSM_XJUST.parquet", "mixup_file7_HITSM_PU.parquet","mixup_file8_HITSM_CWRU.parquet",
        #     # KAIST missing 8
        #     "mixup_file0_KAIST_MFPT.parquet", "mixup_file1_KAIST_TORINO.parquet", "mixup_file2_KAIST_UO.parquet",
        #     "mixup_file3_KAIST_FEMTO.parquet", "mixup_file4_KAIST_XJTUSY.parquet", "mixup_file5_KAIST_PU.parquet",
        #     "mixup_file6_KAIST_CWRU.parquet","mixup_file0_KAIST_IMS.parquet",
        #     # MFPT OK
        #     "mixup_file0_MFPT_IMS.parquet","mixup_file1_MFPT_TORINO.parquet","mixup_file2_MFPT_UO.parquet",
        #     "mixup_file3_MFPT_FEMTO.parquet","mixup_file4_MFPT_XJTUSY.parquet","mixup_file5_MFPT_PU.parquet",
        #     "mixup_file6_MFPT_CWRU.parquet",
        #     # IMS OK
        #     "mixup_file7_IMS_TORINO.parquet","mixup_file8_IMS_UO.parquet","mixup_file9_IMS_FEMTO.parquet",
        #     "mixup_file10_IMS_XJTUSY.parquet","mixup_file11_IMS_PU.parquet","mixup_file12_IMS_CWRU.parquet",
        #     # TORINO
        #     "mixup_file0_TORINO_UO.parquet","mixup_file1_TORINO_FEMTO.parquet","mixup_file2_TORINO_XJTUSY.parquet",
        #     "mixup_file3_TORINO_PU.parquet","mixup_file4_TORINO_CWRU.parquet",
        #     #UO OK
        #     "mixup_file0_UO_FEMTO.parquet", "mixup_file1_UO_XJTUSY.parquet",
        #     "mixup_file2_UO_PU.parquet","mixup_file3_UO_CWRU.parquet",
        #     #FEMTO
        #     "mixup_file4_FEMTO_XJTUSY.parquet", "mixup_file5_FEMTO_PU.parquet", "mixup_file6_FEMTO_CWRU.parquet",
        #     #XJTUSY
        #     "mixup_file7_XJTUSY_PU.parquet", "mixup_file8_XJTUSY_CWRU.parquet",
        #     #PU
        #     "mixup_file9_PU_CWRU.parquet"]

        total_files_10_per = ["mixup_file1_HITSM_MFPT.parquet", "mixup_file3_HITSM_TORINO.parquet", "mixup_file1_MFPT_TORINO.parquet"]
        total_files_30_per = ["mixup_file1_HITSM_MFPT.parquet", "mixup_file3_HITSM_TORINO.parquet", "mixup_file7_HITSM_PU.parquet", "mixup_file8_HITSM_CWRU.parquet",
                              "mixup_file1_MFPT_TORINO.parquet", "mixup_file5_MFPT_PU.parquet", "mixup_file6_MFPT_CWRU.parquet",
                              "mixup_file3_TORINO_PU.parquet","mixup_file4_TORINO_CWRU.parquet",
                              "mixup_file9_PU_CWRU.parquet"]

        total_files_50_per = ["mixup_file1_HITSM_MFPT.parquet", "mixup_file3_HITSM_TORINO.parquet", "mixup_file7_HITSM_PU.parquet", "mixup_file8_HITSM_CWRU.parquet","mixup_file6_HITSM_XJUST.parquet",
                              "mixup_file1_MFPT_TORINO.parquet", "mixup_file5_MFPT_PU.parquet", "mixup_file6_MFPT_CWRU.parquet","mixup_file4_MFPT_XJTUSY.parquet",
                              "mixup_file3_TORINO_PU.parquet","mixup_file4_TORINO_CWRU.parquet","mixup_file2_TORINO_XJTUSY.parquet",
                              "mixup_file9_PU_CWRU.parquet", "mixup_file7_XJTUSY_PU1.parquet", "mixup_file7_XJTUSY_PU2.parquet","mixup_file8_XJTUSY_CWRU.parquet",]

        total_files_80_per = ["mixup_file1_HITSM_MFPT.parquet", "mixup_file3_HITSM_TORINO.parquet", "mixup_file7_HITSM_PU.parquet", "mixup_file8_HITSM_CWRU.parquet","mixup_file6_HITSM_XJUST.parquet",
                              "mixup_file4_HITSM_UO.parquet","mixup_file5_HITSM_FEMTO.parquet",
                              "mixup_file1_MFPT_TORINO.parquet", "mixup_file5_MFPT_PU.parquet", "mixup_file6_MFPT_CWRU.parquet",
                              "mixup_file4_MFPT_XJTUSY.parquet","mixup_file2_MFPT_UO.parquet","mixup_file3_MFPT_FEMTO.parquet",
                              "mixup_file3_TORINO_PU.parquet","mixup_file4_TORINO_CWRU.parquet","mixup_file2_TORINO_XJTUSY.parquet","mixup_file0_TORINO_UO.parquet","mixup_file1_TORINO_FEMTO.parquet",
                              "mixup_file9_PU_CWRU.parquet", "mixup_file7_XJTUSY_PU1.parquet", "mixup_file7_XJTUSY_PU2.parquet","mixup_file8_XJTUSY_CWRU.parquet",
                              "mixup_file0_UO_FEMTO.parquet", "mixup_file1_UO_XJTUSY.parquet",
                              "mixup_file2_UO_PU.parquet", "mixup_file3_UO_CWRU.parquet",
                              "mixup_file4_FEMTO_XJTUSY.parquet", "mixup_file5_FEMTO_PU.parquet",
                              "mixup_file6_FEMTO_CWRU.parquet",
                              ]

        if args.mixup_percentage_included == 10:
            data_files = total_files_10_per
        elif args.mixup_percentage_included == 30:
            data_files = total_files_30_per
        elif args.mixup_percentage_included == 50:
            data_files = total_files_50_per
        elif args.mixup_percentage_included == 80:
            data_files = total_files_80_per
        elif args.mixup_percentage_included == 100:
            data_files = total_files
        else:
            print("NOT IMPLEMENTED")

        for file in data_files:
            train_files.append(os.path.join(mixup_files, file))

    train_dataset = ConcatDataset([
        StreamingPHMDataset(data_file=f, target_seq_len=args.seq_len) for f in train_files
    ])


    val_dataset = ConcatDataset([
        StreamingPHMDataset(data_file=f, target_seq_len=args.seq_len) for f in val_files
    ])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader
