import glob
import torch
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import pyarrow.parquet as pq
import torch.nn.functional as F


class PHMDataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, data_file, data_type):
        super(PHMDataset, self).__init__()

        data_file = pq.read_table(data_file)
        x_np_list = data_file['samples'].to_pylist()

        x_np = np.array(x_np_list)   # Expected dimension: [num_samples, num_channels, seq_length]
        x_data = torch.tensor(x_np)

        x_data = self.adjust_sequence_length(x_data)
        print(f"New data shapes: {x_data.shape}")

        # Update class attributes
        x_data = x_data.to(torch.bfloat16)
        self.x_data = x_data

        self.len = x_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index]

    def __len__(self):
        return self.len


    def adjust_sequence_length(self, x, target_seq_len=1024):
        """
        Processes input tensor x and target y to match a specified target sequence length.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, num_channels, sequence_length].
            y (torch.Tensor): Target tensor.
            target_seq_len (int): Desired sequence length for the output.

        Returns:
            torch.Tensor: Processed version of x with the desired sequence length.
            torch.Tensor: Corresponding target tensor.
        """
        batch_size, num_channels, seq_length = x.shape
        window_size = target_seq_len // num_channels
        remainder = target_seq_len % num_channels

        if target_seq_len < num_channels * seq_length and window_size < seq_length:
            return self._windowed_chunk_sampling(x, seq_length, window_size, remainder)
        elif target_seq_len == num_channels * seq_length:
            return x.flatten(start_dim=1).unsqueeze(dim=1)
        else:
            raise NotImplementedError("Other cases are not implemented yet.")

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

        return x_processed

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
        
        data_files = total_files


        for file in data_files:
            train_files.append(os.path.join(mixup_files, file))

    train_dataset = ConcatDataset(
        [PHMDataset(data_file=data_file, data_type='train') for data_file in train_files]
    )

    val_dataset = ConcatDataset(
        [PHMDataset(data_file=data_file, data_type='val') for data_file in val_files]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader