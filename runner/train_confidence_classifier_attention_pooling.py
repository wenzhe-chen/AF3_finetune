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
from protenix.model.modules.attentionpooling import AttentionPooling
import glob
from pathlib import Path
import math


def find_feature_files(data_dir):
    """Find all feature and label files in the data directory."""
    data_dir = Path(data_dir)
    a_feats_files = sorted(data_dir.glob("**/a_feats*.pt"))
    z_pair_feats_files = sorted(data_dir.glob("**/z_pair_feats*.pt"))
    label_files = sorted(data_dir.glob("**/labels*.pt"))
    
    print(f"Found {len(a_feats_files)} a_feats files:")
    for f in a_feats_files:
        print(f"  {f}")
    print(f"\nFound {len(z_pair_feats_files)} z_pair_feats files:")
    for f in z_pair_feats_files:
        print(f"  {f}")
    print(f"\nFound {len(label_files)} label files:")
    for f in label_files:
        print(f"  {f}")
    
    return [str(f) for f in a_feats_files], [str(f) for f in z_pair_feats_files], [str(f) for f in label_files]

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
    def __init__(self, a_feats_files, z_pair_feats_files, label_files, sample_indices, device='cuda', shared_data=None):
        """
        Dataset that loads all data into memory at once.
        sample_indices: list of (file_idx, sample_idx) tuples
        shared_data: Optional shared data from another dataset instance
        """
        self.device = device
        self.sample_indices = sample_indices
        
        if shared_data is not None:
            # Use shared data from another instance
            self.all_a_feats = shared_data['a_feats']
            self.all_z_pair_feats = shared_data['z_pair_feats']
            self.all_labels = shared_data['labels']
            self.index_mapping = shared_data['index_mapping']
            print(f"Using shared data with {len(self.sample_indices)} samples")
        else:
            # Load all files into memory
            print("Loading all data into memory...")
            
            self.all_a_feats = []
            self.all_z_pair_feats = []
            self.all_labels = []
            
            # Calculate cumulative sizes for mapping
            cumulative_sizes = [0]
            for file_idx, a_file in enumerate(a_feats_files):
                print(f"Loading file {file_idx + 1}/{len(a_feats_files)}")
                a_data = torch.load(a_file, map_location='cpu')
                z_data = torch.load(z_pair_feats_files[file_idx], map_location='cpu')
                label_data = torch.load(label_files[file_idx], map_location='cpu')
                
                # Convert to list format if needed
                if not isinstance(a_data, list):
                    a_data = [a_data]
                    z_data = [z_data]
                    label_data = [label_data]
                
                # Store all samples from this file
                self.all_a_feats.extend(a_data)
                self.all_z_pair_feats.extend(z_data)
                self.all_labels.extend(label_data)
                
                # Update cumulative sizes
                cumulative_sizes.append(cumulative_sizes[-1] + len(a_data))
            
            print(f"Loaded {len(self.all_a_feats)} total samples into memory")
            
            # Convert to tensors but keep on CPU (move to device during training)
            print("Converting to tensors (keeping on CPU)...")
            self.all_a_feats = [feat for feat in self.all_a_feats]  # Keep on CPU
            self.all_z_pair_feats = [feat for feat in self.all_z_pair_feats]  # Keep on CPU
            self.all_labels = torch.tensor(self.all_labels)  # Keep on CPU
            
            # Create mapping from sample_indices to global indices
            self.index_mapping = []
            for file_idx, sample_idx in sample_indices:
                global_idx = cumulative_sizes[file_idx] + sample_idx
                self.index_mapping.append(global_idx)
            
            print("Data loading complete!")
    
    def get_shared_data(self):
        """Return shared data for other instances"""
        return {
            'a_feats': self.all_a_feats,
            'z_pair_feats': self.all_z_pair_feats,
            'labels': self.all_labels,
            'index_mapping': self.index_mapping
        }
    
    def __len__(self):
        return len(self.sample_indices)
    
    def __getitem__(self, idx):
        global_idx = self.index_mapping[idx]
        return (
            self.all_a_feats[global_idx].to(self.device),
            self.all_z_pair_feats[global_idx].to(self.device),
            self.all_labels[global_idx].to(self.device)
        )

class SampleBasedDataset(Dataset):
    def __init__(self, a_feats_files, z_pair_feats_files, label_files, sample_indices, device='cuda'):
        """
        Dataset that works with sample indices rather than file indices.
        sample_indices: list of (file_idx, sample_idx) tuples
        """
        self.a_feats_files = a_feats_files
        self.z_pair_feats_files = z_pair_feats_files
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
            a_data = torch.load(self.a_feats_files[file_idx], map_location='cpu')
            z_data = torch.load(self.z_pair_feats_files[file_idx], map_location='cpu')
            label_data = torch.load(self.label_files[file_idx], map_location='cpu')
            
            # Convert to list format if needed
            if not isinstance(a_data, list):
                a_data = [a_data]
                z_data = [z_data]
                label_data = [label_data]
            
            # Cache the file
            self.file_cache[file_idx] = {
                'a_feats': a_data,
                'z_pair_feats': z_data,
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
            self.file_cache[file_idx]['a_feats'][sample_idx].to(self.device),
            self.file_cache[file_idx]['z_pair_feats'][sample_idx].to(self.device),
            torch.tensor(self.file_cache[file_idx]['labels'][sample_idx], device=self.device)
        )
    
    def __getitem__(self, idx):
        file_idx, sample_idx = self.sample_indices[idx]
        return self.load_sample_from_file(file_idx, sample_idx)

class FileChunkedDataset(Dataset):
    def __init__(self, a_feats_files, z_pair_feats_files, label_files, device='cuda'):
        self.a_feats_files = a_feats_files
        self.z_pair_feats_files = z_pair_feats_files
        self.label_files = label_files
        self.device = device
        
        # Just store file paths, don't load anything yet
        self.file_sizes = []
        self.total_size = 0
        print("\nCalculating dataset size...")
        for i, a_file in enumerate(a_feats_files, 1):
            print(f"Checking file {i}/{len(a_feats_files)}")
            # Quick check without loading full data
            sample_data = torch.load(a_file, map_location='cpu')
            if isinstance(sample_data, list):
                size = len(sample_data)
            else:
                size = sample_data.size(0) if sample_data.dim() == 3 else 1
            self.file_sizes.append(size)
            self.total_size += size
            del sample_data
        
        # Cache for loaded files (optional)
        self.file_cache = {}
        self.max_cache_size = 2  # Keep only 2 files in memory at once
    
    def __len__(self):
        return self.total_size
    
    def load_sample_from_file(self, file_idx, sample_idx):
        """Load a single sample from a specific file"""
        # Check if file is cached
        if file_idx not in self.file_cache:
            # Load file if not cached
            a_data = torch.load(self.a_feats_files[file_idx], map_location='cpu')
            z_data = torch.load(self.z_pair_feats_files[file_idx], map_location='cpu')
            label_data = torch.load(self.label_files[file_idx], map_location='cpu')
            
            # Convert to list format if needed
            if not isinstance(a_data, list):
                a_data = [a_data]
                z_data = [z_data]
                label_data = [label_data]
            
            # Cache the file
            self.file_cache[file_idx] = {
                'a_feats': a_data,
                'z_pair_feats': z_data,
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
            self.file_cache[file_idx]['a_feats'][sample_idx].to(self.device),
            self.file_cache[file_idx]['z_pair_feats'][sample_idx].to(self.device),
            torch.tensor(self.file_cache[file_idx]['labels'][sample_idx], device=self.device)
        )
    
    def __getitem__(self, idx):
        # Find which file and position this index corresponds to
        file_idx = 0
        pos = idx
        while pos >= self.file_sizes[file_idx]:
            pos -= self.file_sizes[file_idx]
            file_idx += 1
        
        # Debug: Print the mapping
        #print(f"Index {idx} -> File {file_idx}, Position {pos}")
        #print(f"File sizes: {self.file_sizes}")
        
        # Load only the specific sample needed
        return self.load_sample_from_file(file_idx, pos)

def confidence_collate_fn(batch):
    a_feats, z_pair_feats, labels = zip(*batch)
    labels = torch.tensor(labels)
    return list(a_feats), list(z_pair_feats), labels

class DualAttentionPoolingClassifier(torch.nn.Module):
    def __init__(self, a_feat_dim, z_pair_feat_dim, hidden_channels=64, dropout_rate=0.1):
        super(DualAttentionPoolingClassifier, self).__init__()
        # Attention pooling for atom-level features (a_feats)
        self.a_pooling = AttentionPooling(
            n_in=a_feat_dim,
            hidden_dim=hidden_channels,
        )
        # Attention pooling for pairwise features (z_pair_feats)
        self.z_pair_pooling = AttentionPooling(
            n_in=z_pair_feat_dim,
            hidden_dim=hidden_channels,
        )
        # Dropout layers
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.dropout2 = torch.nn.Dropout(dropout_rate)
        # Final classification layer - combine both pooled features
        self.classifier = torch.nn.Linear(a_feat_dim + z_pair_feat_dim, 1)

    def forward(self, a_feats_list, z_pair_feats_list):
        batch_size = len(a_feats_list)
        device = next(self.parameters()).device
        
        # Process each sample in the batch
        pooled_features = []
        for i in range(batch_size):
            # Pool atom features
            a_feat = a_feats_list[i]
            # Remove extra dimensions if present
            while a_feat.dim() > 2:
                a_feat = a_feat.squeeze(0)
            # Now a_feat should be [N_atoms, feat_dim]
            if a_feat.dim() == 1:
                a_feat = a_feat.unsqueeze(0)  # Add atom dim -> [1, feat_dim]
            
            # Handle z_pair features
            z_pair = z_pair_feats_list[i]
            # Remove extra dimensions if present
            while z_pair.dim() > 2:
                z_pair = z_pair.squeeze(0)
            # Now z_pair should be [N*N, feat_dim]
            if z_pair.dim() == 1:
                z_pair = z_pair.unsqueeze(0)  # Add pair dim -> [1, feat_dim]
            
            # Get pooled features with dropout
            a_pooled = self.dropout1(self.a_pooling(a_feat))  # [a_feat_dim]
            z_pair_pooled = self.dropout2(self.z_pair_pooling(z_pair))  # [z_pair_feat_dim]
            
            # Combine scores
            combined = torch.cat([a_pooled, z_pair_pooled], dim=-1)  # [a_feat_dim + z_pair_feat_dim]
            pooled_features.append(combined)
        
        # Stack all samples in the batch
        pooled_features = torch.stack(pooled_features, dim=0)  # [batch_size, a_feat_dim + z_pair_feat_dim]
        
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
        },
        tags=["protein", "confidence", "deepset"],
        notes="Training confidence classifier for protein binding prediction"
    )

    # Find all feature and label files
    a_feats_paths, z_pair_feats_paths, label_paths = find_feature_files(args.data_dir)
    
    if not (a_feats_paths and z_pair_feats_paths and label_paths):
        raise ValueError(f"Could not find required files in {args.data_dir}")

    # Get feature dimensions from first samples
    a_sample = torch.load(a_feats_paths[0], map_location='cpu')
    if isinstance(a_sample, list):
        a_feat_dim = a_sample[0].shape[-1]
    else:
        a_feat_dim = a_sample.shape[-1]
    del a_sample

    z_sample = torch.load(z_pair_feats_paths[0], map_location='cpu')
    if isinstance(z_sample, list):
        z_pair_feat_dim = z_sample[0].shape[-1]
    else:
        z_pair_feat_dim = z_sample.shape[-1]
    del z_sample

    # Create sample-based split instead of file-based split
    print("\nCreating sample-based train/eval/test split...")
    
    # First, calculate total samples and create sample indices
    total_samples = 0
    sample_indices = []  # List of (file_idx, sample_idx) tuples
    
    print("Calculating total samples across all files...")
    for file_idx, a_file in enumerate(a_feats_paths):
        print(f"Checking file {file_idx + 1}/{len(a_feats_paths)}")
        # Quick check without loading full data
        sample_data = torch.load(a_file, map_location='cpu')
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
            a_feats_paths,
            z_pair_feats_paths,
            label_paths,
            train_indices,
            device=device
        )
        
        # Get shared data from train dataset
        shared_data = train_dataset.get_shared_data()
        
        # Create eval and test datasets using shared data
        eval_dataset = DatasetClass(
            a_feats_paths,
            z_pair_feats_paths,
            label_paths,
            eval_indices,
            device=device,
            shared_data=shared_data
        )
        
        test_dataset = DatasetClass(
            a_feats_paths,
            z_pair_feats_paths,
            label_paths,
            test_indices,
            device=device,
            shared_data=shared_data
        )
    else:
        print("Using SampleBasedDataset (loading data on-demand)")
        DatasetClass = SampleBasedDataset
        
        train_dataset = DatasetClass(
            a_feats_paths,
            z_pair_feats_paths,
            label_paths,
            train_indices,
            device=device
        )
        
        eval_dataset = DatasetClass(
            a_feats_paths,
            z_pair_feats_paths,
            label_paths,
            eval_indices,
            device=device
        )
        
        test_dataset = DatasetClass(
            a_feats_paths,
            z_pair_feats_paths,
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
    model = DualAttentionPoolingClassifier(
        a_feat_dim=a_feat_dim,
        z_pair_feat_dim=z_pair_feat_dim,
        hidden_channels=args.hidden_channels,
        dropout_rate=0.1
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Loss and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=15, verbose=True
    )

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
        for a_feats_batch, z_pair_feats_batch, yb in train_loader:
            # Move each tensor in the lists to device
            a_feats_batch = [a.to(device, dtype=torch.float32) for a in a_feats_batch]
            z_pair_feats_batch = [z.to(device, dtype=torch.float32) for z in z_pair_feats_batch]
            yb = yb.to(device)
            
            optimizer.zero_grad()
            out = model(a_feats_batch, z_pair_feats_batch)
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
        })

        # Eval
        model.eval()
        eval_loss = 0
        correct, total = 0, 0
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for a_feats_batch, z_pair_feats_batch, yb in eval_loader:
                # Move each tensor in the lists to device
                a_feats_batch = [a.to(device, dtype=torch.float32) for a in a_feats_batch]
                z_pair_feats_batch = [z.to(device, dtype=torch.float32) for z in z_pair_feats_batch]
                yb = yb.to(device)
                
                out = model(a_feats_batch, z_pair_feats_batch)
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
        })

        # Compute and log AUC
        try:
            area_under_curve = roc_auc_score(all_labels, all_probs)
            if (epoch + 1) % 100 == 0 or (epoch + 1) == args.epochs:
                print(f"Epoch {epoch+1}/{args.epochs} - Eval AUC: {area_under_curve:.4f}")
            wandb.log({"eval_auc": area_under_curve, "epoch": epoch + 1})
        except Exception as e:
            if (epoch + 1) % 100 == 0 or (epoch + 1) == args.epochs:
                print(f"AUC calculation failed: {e}")
            wandb.log({"eval_auc": None, "epoch": epoch + 1})

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
        scheduler.step(avg_eval_loss)
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({"learning_rate": current_lr, "epoch": epoch + 1})

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
        for a_feats_batch, z_pair_feats_batch, yb in test_loader:
            a_feats_batch = [a.to(device, dtype=torch.float32) for a in a_feats_batch]
            z_pair_feats_batch = [z.to(device, dtype=torch.float32) for z in z_pair_feats_batch]
            yb = yb.to(device)
            
            out = model(a_feats_batch, z_pair_feats_batch)
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
        for a_feats_batch, z_pair_feats_batch, yb in eval_loader:
            a_feats_batch = [a.to(device, dtype=torch.float32) for a in a_feats_batch]
            z_pair_feats_batch = [z.to(device, dtype=torch.float32) for z in z_pair_feats_batch]
            yb = yb.to(device)
            out = model(a_feats_batch, z_pair_feats_batch)
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
