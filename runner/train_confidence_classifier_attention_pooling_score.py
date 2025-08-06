# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Attribution-NonCommercial 4.0 International
# License (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the
# License at
#
#     https://creativecommons.org/licenses/by-nc/4.0/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import wandb
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
#from protenix.model.modules.attentionpooling import AttentionPooling
import glob
from pathlib import Path
import math

import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    def __init__(self, n_in: int, hidden_dim: int = 128):
        super().__init__()
        self.attn_mlp = nn.Sequential(
            nn.Linear(n_in, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):  # input: (L², n_in)

        attn_scores = self.attn_mlp(x).squeeze(-1)  # (L²,)
        attn_weights = F.softmax(attn_scores, dim=0)  # (L²,)
        pooled = torch.sum(attn_weights.unsqueeze(1) * x, dim=0)  # (n_in,)
        return pooled  # shape: (n_in,)

def create_position_encoding(N_tokens, encoding_dim=64):
    """
    Create position encoding for token pairs.
    
    Args:
        N_tokens (int): Number of tokens
        encoding_dim (int): Dimension of position encoding
        
    Returns:
        torch.Tensor: Position encoding of shape [N_tokens, N_tokens, encoding_dim]
    """
    # Create position indices for i and j tokens
    pos_i = torch.arange(N_tokens, dtype=torch.float32).unsqueeze(1)  # [N_tokens, 1]
    pos_j = torch.arange(N_tokens, dtype=torch.float32).unsqueeze(0)  # [1, N_tokens]
    
    # Create relative position encoding
    rel_pos = pos_i - pos_j  # [N_tokens, N_tokens]
    
    # Create sinusoidal encoding
    encoding = torch.zeros(N_tokens, N_tokens, encoding_dim)
    for k in range(encoding_dim):
        if k % 2 == 0:
            encoding[:, :, k] = torch.sin(rel_pos / (10000 ** (k / encoding_dim)))
        else:
            encoding[:, :, k] = torch.cos(rel_pos / (10000 ** ((k-1) / encoding_dim)))
    
    return encoding


class PositionEncodedTokenPairFeatures:
    """
    Class to handle position encoding for token pair features.
    """
    def __init__(self, encoding_dim=64):
        self.encoding_dim = encoding_dim
        self.position_encodings = {}
    
    def get_position_encoding(self, N_tokens, device):
        """
        Get or create position encoding for given number of tokens.
        
        Args:
            N_tokens (int): Number of tokens
            device (torch.device): Device to place encoding on
            
        Returns:
            torch.Tensor: Position encoding of shape [N_tokens, N_tokens, encoding_dim]
        """
        if N_tokens not in self.position_encodings:
            pos_enc = create_position_encoding(N_tokens, self.encoding_dim)
            self.position_encodings[N_tokens] = pos_enc.to(device)
        
        return self.position_encodings[N_tokens]
    
    def encode_token_pair_features(self, features, device):
        """
        Add position encoding to token pair features.
        
        Args:
            features (torch.Tensor): Token pair features of shape [N_tokens, N_tokens]
            device (torch.device): Device to place encoding on
            
        Returns:
            torch.Tensor: Features with position encoding of shape [N_tokens*N_tokens, 1+encoding_dim]
        """
        N_tokens = features.shape[0]
        
        # Get position encoding
        pos_enc = self.get_position_encoding(N_tokens, device)
        
        # Flatten features and add position encoding
        features_flat = features.flatten().unsqueeze(-1)  # [N_tokens*N_tokens, 1]
        pos_enc_flat = pos_enc.reshape(-1, self.encoding_dim)  # [N_tokens*N_tokens, encoding_dim]
        
        # Concatenate features with position encoding
        encoded_features = torch.cat([features_flat, pos_enc_flat], dim=-1)  # [N_tokens*N_tokens, 1+encoding_dim]
        
        return encoded_features


def find_feature_files(data_dir):
    """Find all feature and label files in the data directory."""
    data_dir = Path(data_dir)
    atom_plddt_files = sorted(data_dir.glob("**/atom_plddt*.pt"))
    contact_probs_files = sorted(data_dir.glob("**/contact_probs*.pt"))
    token_pair_pae_files = sorted(data_dir.glob("**/token_pair_pae*.pt"))
    token_pair_pde_files = sorted(data_dir.glob("**/token_pair_pde*.pt"))
    label_files = sorted(data_dir.glob("**/labels*.pt"))
    
    print(f"Found {len(atom_plddt_files)} atom_plddt files:")
    for f in atom_plddt_files:
        print(f"  {f}")
    print(f"\nFound {len(contact_probs_files)} contact_probs files:")
    for f in contact_probs_files:
        print(f"  {f}")
    print(f"\nFound {len(token_pair_pae_files)} token_pair_pae files:")
    for f in token_pair_pae_files:
        print(f"  {f}")
    print(f"\nFound {len(token_pair_pde_files)} token_pair_pde files:")
    for f in token_pair_pde_files:
        print(f"  {f}")
    print(f"\nFound {len(label_files)} label files:")
    for f in label_files:
        print(f"  {f}")
    
    return [str(f) for f in atom_plddt_files], [str(f) for f in contact_probs_files], [str(f) for f in token_pair_pae_files], [str(f) for f in token_pair_pde_files], [str(f) for f in label_files]

def parse_args():
    parser = argparse.ArgumentParser(description="Train Confidence Classifier")
    # Replace individual file paths with data directory
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing feature and label files')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--output', type=str, default='confidence_classifier.pt', help='Output model path')
    parser.add_argument('--hidden_channels', type=int, default=64, help='Hidden channels in DeepSet transformer')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads in DeepSet transformer')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience (epochs)')
    parser.add_argument('--pretrained_model', type=str, default=None, help='Path to a pre-trained model to skip training and only run evaluation/plotting')
    parser.add_argument('--load_all_in_memory', action='store_true', help='Load all data into memory at once (faster but uses more memory)')
    parser.add_argument('--pos_encoding_dim', type=int, default=64, help='Dimension of position encoding for token pair features')
    return parser.parse_args()

class SequentialFileBatchSampler:
    """BatchSampler that processes one file at a time to minimize I/O"""
    def __init__(self, dataset, batch_size, file_sizes, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.file_sizes = file_sizes
        self.shuffle = shuffle
        
    def __iter__(self):
        # Process one file at a time
        start_idx = 0
        for file_size in self.file_sizes:
            # Get indices for current file
            indices = list(range(start_idx, start_idx + file_size))
            
            # Shuffle indices within this file
            if self.shuffle:
                np.random.shuffle(indices)
            
            # Yield batches from this file
            for i in range(0, len(indices), self.batch_size):
                yield indices[i:i + self.batch_size]
            
            start_idx += file_size
    
    def __len__(self):
        return sum(math.ceil(size / self.batch_size) for size in self.file_sizes)

class InMemoryDataset(Dataset):
    def __init__(self, atom_plddt_files, contact_probs_files, token_pair_pae_files, token_pair_pde_files, label_files, sample_indices, device='cuda', shared_data=None):
        """
        Dataset that loads all data into memory at once.
        sample_indices: list of (file_idx, sample_idx) tuples
        shared_data: Optional shared data from another dataset instance
        """
        self.device = device
        self.sample_indices = sample_indices
        
        if shared_data is not None:
            # Use shared data from another instance
            self.all_atom_plddt = shared_data['atom_plddt']
            self.all_contact_probs = shared_data['contact_probs']
            self.all_token_pair_pae = shared_data['token_pair_pae']
            self.all_token_pair_pde = shared_data['token_pair_pde']
            self.all_labels = shared_data['labels']
            self.index_mapping = shared_data['index_mapping']
            print(f"Using shared data with {len(self.sample_indices)} samples")
        else:
            # Load all files into memory
            print("Loading all data into memory...")
            
            self.all_atom_plddt = []
            self.all_contact_probs = []
            self.all_token_pair_pae = []
            self.all_token_pair_pde = []
            self.all_labels = []
            
            # Calculate cumulative sizes for mapping
            cumulative_sizes = [0]
            for file_idx, atom_plddt_file in enumerate(atom_plddt_files):
                print(f"Loading file {file_idx + 1}/{len(atom_plddt_files)}")
                atom_plddt_data = torch.load(atom_plddt_file, map_location='cpu')
                contact_probs_data = torch.load(contact_probs_files[file_idx], map_location='cpu')
                token_pair_pae_data = torch.load(token_pair_pae_files[file_idx], map_location='cpu')
                token_pair_pde_data = torch.load(token_pair_pde_files[file_idx], map_location='cpu')
                label_data = torch.load(label_files[file_idx], map_location='cpu')
                
                # Convert to list format if needed
                if not isinstance(atom_plddt_data, list):
                    atom_plddt_data = [atom_plddt_data]
                    contact_probs_data = [contact_probs_data]
                    token_pair_pae_data = [token_pair_pae_data]
                    token_pair_pde_data = [token_pair_pde_data]
                    label_data = [label_data]
                
                # Store all samples from this file
                self.all_atom_plddt.extend(atom_plddt_data)
                self.all_contact_probs.extend(contact_probs_data)
                self.all_token_pair_pae.extend(token_pair_pae_data)
                self.all_token_pair_pde.extend(token_pair_pde_data)
                self.all_labels.extend(label_data)
                
                # Update cumulative sizes
                cumulative_sizes.append(cumulative_sizes[-1] + len(atom_plddt_data))
            
            print(f"Loaded {len(self.all_atom_plddt)} total samples into memory")
            
            # Convert to tensors and move to GPU directly
            print("Converting to tensors and moving to GPU...")
            self.all_atom_plddt = [feat.to(device) for feat in self.all_atom_plddt]  # Move to GPU
            self.all_contact_probs = [feat.to(device) for feat in self.all_contact_probs]  # Move to GPU
            self.all_token_pair_pae = [feat.to(device) for feat in self.all_token_pair_pae]  # Move to GPU
            self.all_token_pair_pde = [feat.to(device) for feat in self.all_token_pair_pde]  # Move to GPU
            self.all_labels = torch.tensor(self.all_labels, device=device)  # Move to GPU
            
            # Create mapping from sample_indices to global indices
            self.index_mapping = []
            for file_idx, sample_idx in sample_indices:
                global_idx = cumulative_sizes[file_idx] + sample_idx
                self.index_mapping.append(global_idx)
            
            print("Data loading complete!")
    
    def get_shared_data(self):
        """Return shared data for other instances"""
        return {
            'atom_plddt': self.all_atom_plddt,
            'contact_probs': self.all_contact_probs,
            'token_pair_pae': self.all_token_pair_pae,
            'token_pair_pde': self.all_token_pair_pde,
            'labels': self.all_labels,
            'index_mapping': self.index_mapping
        }
    
    def __len__(self):
        return len(self.sample_indices)
    
    def __getitem__(self, idx):
        global_idx = self.index_mapping[idx]
        return (
            self.all_atom_plddt[global_idx],  # Already on GPU
            self.all_contact_probs[global_idx],  # Already on GPU
            self.all_token_pair_pae[global_idx],  # Already on GPU
            self.all_token_pair_pde[global_idx],  # Already on GPU
            self.all_labels[global_idx]  # Already on GPU
        )

class SampleBasedDataset(Dataset):
    def __init__(self, atom_plddt_files, contact_probs_files, token_pair_pae_files, token_pair_pde_files, label_files, sample_indices, device='cuda'):
        """
        Dataset that works with sample indices rather than file indices.
        sample_indices: list of (file_idx, sample_idx) tuples
        """
        self.atom_plddt_files = atom_plddt_files
        self.contact_probs_files = contact_probs_files
        self.token_pair_pae_files = token_pair_pae_files
        self.token_pair_pde_files = token_pair_pde_files
        self.label_files = label_files
        self.sample_indices = sample_indices
        self.device = device
        
        # Cache for loaded files (optional)
        self.file_cache = {}
        self.max_cache_size = 2  # Keep only 2 files in memory at once
    
    def __len__(self):
        return len(self.sample_indices)
    
    def load_sample_from_file(self, file_idx, sample_idx):
        """Load a single sample from a specific file"""
        # Check if file is cached
        if file_idx not in self.file_cache:
            # Load file if not cached
            atom_plddt_data = torch.load(self.atom_plddt_files[file_idx], map_location='cpu')
            contact_probs_data = torch.load(self.contact_probs_files[file_idx], map_location='cpu')
            token_pair_pae_data = torch.load(self.token_pair_pae_files[file_idx], map_location='cpu')
            token_pair_pde_data = torch.load(self.token_pair_pde_files[file_idx], map_location='cpu')
            label_data = torch.load(self.label_files[file_idx], map_location='cpu')
            
            # Convert to list format if needed
            if not isinstance(atom_plddt_data, list):
                atom_plddt_data = [atom_plddt_data]
                contact_probs_data = [contact_probs_data]
                token_pair_pae_data = [token_pair_pae_data]
                token_pair_pde_data = [token_pair_pde_data]
                label_data = [label_data]
            
            # Cache the file
            self.file_cache[file_idx] = {
                'atom_plddt': atom_plddt_data,
                'contact_probs': contact_probs_data,
                'token_pair_pae': token_pair_pae_data,
                'token_pair_pde': token_pair_pde_data,
                'labels': label_data
            }
            
            # Remove oldest cache entry if cache is full (but not the current file!)
            if len(self.file_cache) > self.max_cache_size:
                # Find the oldest key that's not the current file
                oldest_key = None
                for key in self.file_cache.keys():
                    if key != file_idx:
                        if oldest_key is None or key < oldest_key:
                            oldest_key = key
                if oldest_key is not None:
                    del self.file_cache[oldest_key]
        
        # Get the specific sample
        return (
            self.file_cache[file_idx]['atom_plddt'][sample_idx].to(self.device),
            self.file_cache[file_idx]['contact_probs'][sample_idx].to(self.device),
            self.file_cache[file_idx]['token_pair_pae'][sample_idx].to(self.device),
            self.file_cache[file_idx]['token_pair_pde'][sample_idx].to(self.device),
            torch.tensor(self.file_cache[file_idx]['labels'][sample_idx], device=self.device)
        )
    
    def __getitem__(self, idx):
        file_idx, sample_idx = self.sample_indices[idx]
        return self.load_sample_from_file(file_idx, sample_idx)


def confidence_collate_fn(batch):
    atom_plddt, contact_probs, token_pair_pae, token_pair_pde, labels = zip(*batch)
    labels = torch.tensor(labels)
    return list(atom_plddt), list(contact_probs), list(token_pair_pae), list(token_pair_pde), labels

class FourFeatureAttentionPoolingClassifier(torch.nn.Module):
    def __init__(self, atom_plddt_dim=1, contact_probs_dim=1, token_pair_pae_dim=1, token_pair_pde_dim=1, hidden_channels=64, dropout_rate=0.1, pos_encoding_dim=64):
        super(FourFeatureAttentionPoolingClassifier, self).__init__()
        # Position encoding for token pair features
        self.pos_encoder = PositionEncodedTokenPairFeatures(encoding_dim=pos_encoding_dim)
        
        # Attention pooling for atom pLDDT features
        self.atom_plddt_pooling = AttentionPooling(
            n_in=1,
            hidden_dim=hidden_channels,
        )
        # Attention pooling for contact probabilities (with position encoding)
        self.contact_probs_pooling = AttentionPooling(
            n_in=1 + pos_encoding_dim,  # Original feature + position encoding
            hidden_dim=hidden_channels,
        )
        # Attention pooling for token pair PAE features (with position encoding)
        self.token_pair_pae_pooling = AttentionPooling(
            n_in=1 + pos_encoding_dim,  # Original feature + position encoding
            hidden_dim=hidden_channels,
        )
        # Attention pooling for token pair PDE features (with position encoding)
        self.token_pair_pde_pooling = AttentionPooling(
            n_in=1 + pos_encoding_dim,  # Original feature + position encoding
            hidden_dim=hidden_channels,
        )
        # No dropout layers
        # Final classification layer - combine all four pooled features
        total_dim = 3 * pos_encoding_dim + 4
        self.classifier = torch.nn.Linear(total_dim, 1)

    def forward(self, atom_plddt_list, contact_probs_list, token_pair_pae_list, token_pair_pde_list):
        batch_size = len(atom_plddt_list)
        device = next(self.parameters()).device
        
        # Process each sample in the batch
        pooled_features = []
        for i in range(batch_size):
            # Process atom pLDDT features
            atom_plddt = atom_plddt_list[i]
            # Remove extra dimensions if present
            while atom_plddt.dim() > 2:
                atom_plddt = atom_plddt.squeeze(0)
            # Now atom_plddt should be [N_atoms, 1]
            if atom_plddt.dim() == 1:
                atom_plddt = atom_plddt.unsqueeze(-1)  # Add feature dim -> [N_atoms, 1]
            
            # Process contact probabilities with position encoding
            contact_probs = contact_probs_list[i]
            # Remove extra dimensions if present
            while contact_probs.dim() > 2:
                contact_probs = contact_probs.squeeze(0)
            # Now contact_probs should be [N_tokens, N_tokens]
            if contact_probs.dim() == 2:
                # Add position encoding and flatten
                contact_probs = self.pos_encoder.encode_token_pair_features(contact_probs, device)
            elif contact_probs.dim() == 1:
                contact_probs = contact_probs.unsqueeze(-1)  # Add feature dim -> [N_tokens, 1]
            
            # Process token pair PAE features with position encoding
            token_pair_pae = token_pair_pae_list[i]
            # Remove extra dimensions if present
            while token_pair_pae.dim() > 2:
                token_pair_pae = token_pair_pae.squeeze(0)
            # Now token_pair_pae should be [N_tokens, N_tokens]
            if token_pair_pae.dim() == 2:
                # Add position encoding and flatten
                token_pair_pae = self.pos_encoder.encode_token_pair_features(token_pair_pae, device)
            elif token_pair_pae.dim() == 1:
                token_pair_pae = token_pair_pae.unsqueeze(-1)  # Add feature dim -> [N_tokens, 1]
            
            # Process token pair PDE features with position encoding
            token_pair_pde = token_pair_pde_list[i]
            # Remove extra dimensions if present
            while token_pair_pde.dim() > 2:
                token_pair_pde = token_pair_pde.squeeze(0)
            # Now token_pair_pde should be [N_tokens, N_tokens]
            if token_pair_pde.dim() == 2:
                # Add position encoding and flatten
                token_pair_pde = self.pos_encoder.encode_token_pair_features(token_pair_pde, device)
            elif token_pair_pde.dim() == 1:
                token_pair_pde = token_pair_pde.unsqueeze(-1)  # Add feature dim -> [N_tokens, 1]
            
            # Get pooled features without dropout
            atom_plddt_pooled = self.atom_plddt_pooling(atom_plddt)  # [atom_plddt_dim]
            contact_probs_pooled = self.contact_probs_pooling(contact_probs)  # [contact_probs_dim]
            token_pair_pae_pooled = self.token_pair_pae_pooling(token_pair_pae)  # [token_pair_pae_dim]
            token_pair_pde_pooled = self.token_pair_pde_pooling(token_pair_pde)  # [token_pair_pde_dim]
            
            # Combine all features
            combined = torch.cat([atom_plddt_pooled, contact_probs_pooled, token_pair_pae_pooled, token_pair_pde_pooled], dim=-1)
            pooled_features.append(combined)
        
        # Stack all samples in the batch
        pooled_features = torch.stack(pooled_features, dim=0)  # [batch_size, total_dim]
        
        # Final classification
        return self.classifier(pooled_features)

def main():
    args = parse_args()

    # Initialize wandb
    wandb.init(
        project="confidence-classifier",  # Your project name
        name=args.output.replace('.pt', ''),  # Use output filename as run name
        config={
            "data_dir": args.data_dir,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "hidden_channels": args.hidden_channels,
            "num_heads": args.num_heads,
            "patience": args.patience,
            "output": args.output,
            "pretrained_model": args.pretrained_model,
            "pos_encoding_dim": args.pos_encoding_dim,
        },
        tags=["protein", "confidence", "deepset"],
        notes="Training confidence classifier for protein binding prediction"
    )

    # Find all feature and label files
    atom_plddt_paths, contact_probs_paths, token_pair_pae_paths, token_pair_pde_paths, label_paths = find_feature_files(args.data_dir)
    
    if not (atom_plddt_paths and contact_probs_paths and token_pair_pae_paths and token_pair_pde_paths and label_paths):
        raise ValueError(f"Could not find required files in {args.data_dir}")

    # Create sample-based split instead of file-based split
    print("\nCreating sample-based train/eval/test split...")
    
    # First, calculate total samples and create sample indices
    total_samples = 0
    sample_indices = []  # List of (file_idx, sample_idx) tuples
    
    print("Calculating total samples across all files...")
    for file_idx, atom_plddt_file in enumerate(atom_plddt_paths):
        print(f"Checking file {file_idx + 1}/{len(atom_plddt_paths)}")
        # Quick check without loading full data
        sample_data = torch.load(atom_plddt_file, map_location='cpu')
        if isinstance(sample_data, list):
            file_size = len(sample_data)
        else:
            file_size = sample_data.size(0) if sample_data.dim() == 3 else 1
        
        # Add all samples from this file
        for sample_idx in range(file_size):
            sample_indices.append((file_idx, sample_idx))
        total_samples += file_size
        del sample_data
    
    print(f"Total samples: {total_samples}")
    
    # Shuffle sample indices
    np.random.shuffle(sample_indices)
    
    # Split based on samples: 70% train, 15% eval, 15% test
    train_size = int(0.7 * total_samples)
    eval_size = int(0.15 * total_samples)
    
    train_indices = sample_indices[:train_size]
    eval_indices = sample_indices[train_size:train_size + eval_size]
    test_indices = sample_indices[train_size + eval_size:]
    
    print(f"Split samples: {len(train_indices)} train, {len(eval_indices)} eval, {len(test_indices)} test")
    print(f"Percentages: {len(train_indices)/total_samples*100:.1f}% train, {len(eval_indices)/total_samples*100:.1f}% eval, {len(test_indices)/total_samples*100:.1f}% test")

    # Create datasets using sample indices
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Choose dataset class based on memory flag
    if args.load_all_in_memory:
        print("Using InMemoryDataset (loading all data into memory)")
        DatasetClass = InMemoryDataset
        
        # Create train dataset first (this will load all data)
        train_dataset = DatasetClass(
            atom_plddt_paths,
            contact_probs_paths,
            token_pair_pae_paths,
            token_pair_pde_paths,
            label_paths,
            train_indices,
            device=device
        )
        
        # Get shared data from train dataset
        shared_data = train_dataset.get_shared_data()
        
        # Create eval and test datasets using shared data
        eval_dataset = DatasetClass(
            atom_plddt_paths,
            contact_probs_paths,
            token_pair_pae_paths,
            token_pair_pde_paths,
            label_paths,
            eval_indices,
            device=device,
            shared_data=shared_data
        )
        
        test_dataset = DatasetClass(
            atom_plddt_paths,
            contact_probs_paths,
            token_pair_pae_paths,
            token_pair_pde_paths,
            label_paths,
            test_indices,
            device=device,
            shared_data=shared_data
        )
    else:
        print("Using SampleBasedDataset (loading data on-demand)")
        DatasetClass = SampleBasedDataset
        
        train_dataset = DatasetClass(
            atom_plddt_paths,
            contact_probs_paths,
            token_pair_pae_paths,
            token_pair_pde_paths,
            label_paths,
            train_indices,
            device=device
        )
        
        eval_dataset = DatasetClass(
            atom_plddt_paths,
            contact_probs_paths,
            token_pair_pae_paths,
            token_pair_pde_paths,
            label_paths,
            eval_indices,
            device=device
        )
        
        test_dataset = DatasetClass(
            atom_plddt_paths,
            contact_probs_paths,
            token_pair_pae_paths,
            token_pair_pde_paths,
            label_paths,
            test_indices,
            device=device
        )

    # Create data loaders with standard samplers (no need for file-based samplers)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=confidence_collate_fn,
        num_workers=0
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=confidence_collate_fn,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=confidence_collate_fn,
        num_workers=0
    )

    print(f"Created data loaders with {len(train_dataset)} training, {len(eval_dataset)} evaluation, and {len(test_dataset)} test samples")

    # Now you can use train_loader and eval_loader in your training loop
    # Each batch will be (list_of_a_feats, list_of_z_pair_feats, labels)

    # Create model - FIX: Use the correct class name
    model = FourFeatureAttentionPoolingClassifier(
        atom_plddt_dim=1,
        contact_probs_dim=1,
        token_pair_pae_dim=1,
        token_pair_pde_dim=1,
        hidden_channels=args.hidden_channels,
        pos_encoding_dim=args.pos_encoding_dim,  # Position encoding dimension for token pair features
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Loss and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Add learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.8, patience=15, verbose=True
    # )

    # Load pretrained model if provided
    if args.pretrained_model is not None:
        model.load_state_dict(torch.load(args.pretrained_model, map_location=device))
        print(f"Loaded pretrained model from {args.pretrained_model}")
        print("Continuing training from pretrained weights...")
    
    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
        for atom_plddt_batch, contact_probs_batch, token_pair_pae_batch, token_pair_pde_batch, yb in train_loader:
            # Data is already on GPU, just ensure correct dtype
            atom_plddt_batch = [a.to(dtype=torch.float32) for a in atom_plddt_batch]
            contact_probs_batch = [c.to(dtype=torch.float32) for c in contact_probs_batch]
            token_pair_pae_batch = [p.to(dtype=torch.float32) for p in token_pair_pae_batch]
            token_pair_pde_batch = [d.to(dtype=torch.float32) for d in token_pair_pde_batch]
            yb = yb.to(device=device, dtype=torch.float32)
            
            optimizer.zero_grad()
            out = model(atom_plddt_batch, contact_probs_batch, token_pair_pae_batch, token_pair_pde_batch)
            # Ensure consistent shapes
            out = out.view(-1)  # Flatten to [batch_size]
            yb = yb.float().view(-1)  # Flatten to [batch_size]
            loss = criterion(out, yb)
            loss.backward()
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Calculate training accuracy
            with torch.no_grad():
                probs = torch.sigmoid(out)
                preds = (probs > 0.5).long()
                train_correct += (preds == yb.long()).sum().item()
                train_total += yb.size(0)
        
        # Calculate average loss and accuracy
        avg_loss = total_loss / len(train_loader)  # Average per batch
        train_acc = train_correct / train_total
        
        if (epoch + 1) % 100 == 0 or (epoch + 1) == args.epochs:
            print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_loss:.4f} - Train Acc: {train_acc:.4f}")
        wandb.log({
            "train_loss": avg_loss,
            "train_acc": train_acc,
            "epoch": epoch + 1
        }, step=epoch+1)

        # Eval
        model.eval()
        eval_loss = 0
        correct, total = 0, 0
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for atom_plddt_batch, contact_probs_batch, token_pair_pae_batch, token_pair_pde_batch, yb in eval_loader:
                # Data is already on GPU, just ensure correct dtype
                atom_plddt_batch = [a.to(dtype=torch.float32) for a in atom_plddt_batch]
                contact_probs_batch = [c.to(dtype=torch.float32) for c in contact_probs_batch]
                token_pair_pae_batch = [p.to(dtype=torch.float32) for p in token_pair_pae_batch]
                token_pair_pde_batch = [d.to(dtype=torch.float32) for d in token_pair_pde_batch]
                yb = yb.to(device=device, dtype=torch.float32)
                
                out = model(atom_plddt_batch, contact_probs_batch, token_pair_pae_batch, token_pair_pde_batch)
                # Ensure consistent shapes
                out = out.view(-1)  # Flatten to [batch_size]
                yb = yb.float().view(-1)  # Flatten to [batch_size]
                
                # Calculate evaluation loss
                loss = criterion(out, yb)
                eval_loss += loss.item()
                
                probs = torch.sigmoid(out)
                preds = (probs > 0.5).long()
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())
                correct += (preds == yb.long()).sum().item()
                total += yb.size(0)
        
        # Calculate average evaluation loss and accuracy
        avg_eval_loss = eval_loss / len(eval_loader)  # Average per batch
        acc = correct / total
        
        if (epoch + 1) % 100 == 0 or (epoch + 1) == args.epochs:
            print(f"Epoch {epoch+1}/{args.epochs} - Eval Loss: {avg_eval_loss:.4f} - Eval Acc: {acc:.4f}")
        wandb.log({
            "eval_loss": avg_eval_loss,
            "eval_acc": acc,
            "epoch": epoch + 1
        }, step=epoch+1)

        # Compute and log AUC
        try:
            area_under_curve = roc_auc_score(all_labels, all_probs)
            if (epoch + 1) % 100 == 0 or (epoch + 1) == args.epochs:
                print(f"Epoch {epoch+1}/{args.epochs} - Eval AUC: {area_under_curve:.4f}")
            wandb.log({"eval_auc": area_under_curve, "epoch": epoch + 1}, step=epoch+1)
        except Exception as e:
            if (epoch + 1) % 100 == 0 or (epoch + 1) == args.epochs:
                print(f"AUC calculation failed: {e}")
            wandb.log({"eval_auc": None, "epoch": epoch + 1}, step=epoch+1)

        # Early stopping and best model saving
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), args.output)
            print(f"[Best] Model saved to {args.output} at epoch {epoch+1} with acc {acc:.4f}")
            wandb.save(args.output)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}. Best acc: {best_acc:.4f} at epoch {best_epoch}")
                break
        
        # Update learning rate based on validation accuracy
        #scheduler.step(avg_eval_loss)
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({"learning_rate": current_lr, "epoch": epoch + 1}, step=epoch+1)

    print(f"Best model saved to {args.output} with acc {best_acc:.4f} at epoch {best_epoch}")
    # Load best model for evaluation/plotting
    model.load_state_dict(torch.load(args.output, map_location=device))
    model.eval()

    # --- Evaluate on test set using the best or loaded model ---
    print("\n=== Test Set Evaluation ===")
    model.eval()
    test_loss = 0
    test_correct, test_total = 0, 0
    test_all_labels = []
    test_all_probs = []
    
    with torch.no_grad():
        for atom_plddt_batch, contact_probs_batch, token_pair_pae_batch, token_pair_pde_batch, yb in test_loader:
            atom_plddt_batch = [a.to(dtype=torch.float32) for a in atom_plddt_batch]
            contact_probs_batch = [c.to(dtype=torch.float32) for c in contact_probs_batch]
            token_pair_pae_batch = [p.to(dtype=torch.float32) for p in token_pair_pae_batch]
            token_pair_pde_batch = [d.to(dtype=torch.float32) for d in token_pair_pde_batch]
            yb = yb.to(device = device, dtype=torch.float32)
            
            out = model(atom_plddt_batch, contact_probs_batch, token_pair_pae_batch, token_pair_pde_batch)
            # Ensure consistent shapes
            out = out.view(-1)  # Flatten to [batch_size]
            yb = yb.float().view(-1)  # Flatten to [batch_size]
            
            # Calculate test loss
            loss = criterion(out, yb)
            test_loss += loss.item()
            
            probs = torch.sigmoid(out)
            preds = (probs > 0.5).long()
            test_all_probs.extend(probs.cpu().numpy())
            test_all_labels.extend(yb.cpu().numpy())
            test_correct += (preds == yb.long()).sum().item()
            test_total += yb.size(0)
    
    # Calculate test metrics
    avg_test_loss = test_loss / len(test_loader)
    test_acc = test_correct / test_total
    
    # Compute test AUC
    try:
        test_auc = roc_auc_score(test_all_labels, test_all_probs)
        print(f"Test Loss: {avg_test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        wandb.log({
            "test_loss": avg_test_loss,
            "test_acc": test_acc,
            "test_auc": test_auc
        })
    except Exception as e:
        print(f"Test AUC calculation failed: {e}")
        wandb.log({
            "test_loss": avg_test_loss,
            "test_acc": test_acc,
            "test_auc": None
        })

    # --- Draw ROC curve using the best or loaded model on eval set ---
    print("\n=== Evaluation Set ROC Curve ===")
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for atom_plddt_batch, contact_probs_batch, token_pair_pae_batch, token_pair_pde_batch, yb in eval_loader:
            atom_plddt_batch = [a.to(dtype=torch.float32) for a in atom_plddt_batch]
            contact_probs_batch = [c.to(dtype=torch.float32) for c in contact_probs_batch]
            token_pair_pae_batch = [p.to(dtype=torch.float32) for p in token_pair_pae_batch]
            token_pair_pde_batch = [d.to(dtype=torch.float32) for d in token_pair_pde_batch]
            yb = yb.to(device=device, dtype=torch.float32)
            out = model(atom_plddt_batch, contact_probs_batch, token_pair_pae_batch, token_pair_pde_batch)
            if out.shape[-1] == 1:
                probs = torch.sigmoid(out.squeeze())
                if probs.ndim == 0:
                    probs = probs.unsqueeze(0)
                all_probs.extend(probs.cpu().numpy().reshape(-1))
                all_labels.extend(yb.cpu().numpy().reshape(-1))
            else:
                probs = torch.softmax(out, dim=1)
                if yb.ndim > 1:
                    yb = yb.argmax(dim=1)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(yb.cpu().numpy().tolist())
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    plt.figure()
    # For binary classification, we don't need to index the second column
    # Because we only have one probability value per sample
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    # Save ROC curve data for later replotting
    np.savez('./output/roc_curve_classifier_only.npz', 
         fpr=fpr, tpr=tpr, thresholds=thresholds, roc_auc=roc_auc)

    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Evaluation Set)')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig('./output/roc_curve_confidence_classifier_eval.png')
    wandb.log({"eval_roc_curve": wandb.Image('./output/roc_curve_confidence_classifier_eval.png')})
    plt.close()
    
    # --- Draw ROC curve using the best or loaded model on test set ---
    print("\n=== Test Set ROC Curve ===")
    test_all_labels = np.array(test_all_labels)
    test_all_probs = np.array(test_all_probs)
    plt.figure()
    test_fpr, test_tpr, test_thresholds = roc_curve(test_all_labels, test_all_probs)
    test_roc_auc = auc(test_fpr, test_tpr)

    # Save test ROC curve data for later replotting
    np.savez('./output/roc_curve_classifier_test.npz', 
         fpr=test_fpr, tpr=test_tpr, thresholds=test_thresholds, roc_auc=test_roc_auc)

    plt.plot(test_fpr, test_tpr, color='red', lw=2, label=f'Test ROC curve (area = {test_roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Test Set)')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig('./output/roc_curve_confidence_classifier_test.png')
    wandb.log({"test_roc_curve": wandb.Image('./output/roc_curve_confidence_classifier_test.png')})
    plt.close()
    
    wandb.finish()


if __name__ == "__main__":
    main()
