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
from protenix.model.modules.deepsettransformer import DeepSetTransformerPooling
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

class FileChunkedDataset(Dataset):
    def __init__(self, a_feats_files, z_pair_feats_files, label_files, device='cuda', chunk_size=1000):
        """
        Dataset that loads and processes one file at a time.
        """
        self.a_feats_files = a_feats_files
        self.z_pair_feats_files = z_pair_feats_files
        self.label_files = label_files
        self.device = device
        self.chunk_size = chunk_size
        
        # Get total size and create index mapping
        self.file_sizes = []
        self.total_size = 0
        print("\nCalculating dataset size...")
        for i, a_file in enumerate(a_feats_files, 1):
            print(f"Checking file {i}/{len(a_feats_files)}")
            size = torch.load(a_file, map_location='cpu')
            size = len(size) if isinstance(size, list) else 1
            self.file_sizes.append(size)
            self.total_size += size
        
        # Current loaded file info
        self.current_file_idx = -1
        self.current_data = None
        
    def __len__(self):
        return self.total_size

    def load_file(self, file_idx):
        """Load a complete file"""
        if self.current_file_idx == file_idx:
            return
            
        # Clear previous file
        if self.current_data is not None:
            del self.current_data
            torch.cuda.empty_cache()

        # Load new file
        print(f"\nLoading file {file_idx + 1}/{len(self.a_feats_files)}")
        a_data = torch.load(self.a_feats_files[file_idx], map_location='cpu')
        z_data = torch.load(self.z_pair_feats_files[file_idx], map_location='cpu')
        label_data = torch.load(self.label_files[file_idx], map_location='cpu')
        
        if not isinstance(a_data, list):
            a_data = [a_data]
            z_data = [z_data]
            label_data = [label_data]
            
        self.current_data = {
            'a_feats': a_data,
            'z_pair_feats': z_data,
            'labels': label_data
        }
        self.current_file_idx = file_idx

    def __getitem__(self, idx):
        # Find which file and position this index corresponds to
        file_idx = 0
        pos = idx
        while pos >= self.file_sizes[file_idx]:
            pos -= self.file_sizes[file_idx]
            file_idx += 1

        # Load the appropriate file if needed
        self.load_file(file_idx)
        
        # Get the item from the current file
        return (
            self.current_data['a_feats'][pos].to(self.device),
            self.current_data['z_pair_feats'][pos].to(self.device),
            torch.tensor(self.current_data['labels'][pos], device=self.device)
        )

def confidence_collate_fn(batch):
    a_feats, z_pair_feats, labels = zip(*batch)
    labels = torch.tensor(labels)
    return list(a_feats), list(z_pair_feats), labels

class DualDeepSetClassifier(torch.nn.Module):
    def __init__(self, a_feat_dim, z_pair_feat_dim, hidden_channels=64, num_heads=8):
        super(DualDeepSetClassifier, self).__init__()
        # DeepSet for atom-level features (a_feats)
        self.a_pooling = DeepSetTransformerPooling(
            n_in=a_feat_dim,
            n_hidden_channels=hidden_channels,
            num_heads=num_heads
        )
        # DeepSet for pairwise features (z_pair_feats)
        self.z_pair_pooling = DeepSetTransformerPooling(
            n_in=z_pair_feat_dim,
            n_hidden_channels=hidden_channels,
            num_heads=num_heads
        )
        # Final classification layer
        self.classifier = torch.nn.Linear(2*hidden_channels, 1)

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
            a_feat = a_feat.unsqueeze(0)  # Add single batch dim -> [1, N_atoms, feat_dim]
            
            # Handle z_pair features
            z_pair = z_pair_feats_list[i]
            # Remove extra dimensions if present
            while z_pair.dim() > 1:
                z_pair = z_pair.squeeze(0)
            # Now z_pair should be [N*N*feat_dim]
            
            # Get the feature dimension from a_feat
            feat_dim = a_feat.size(-1)
            # Calculate N from the total size and feature dimension
            total_size = z_pair.size(0)
            N = int(torch.sqrt(torch.tensor(total_size / feat_dim)))
            
            # Reshape z_pair to proper dimensions
            z_pair = z_pair.view(N*N, feat_dim)  # First get [N*N, feat_dim]
            z_pair = z_pair.unsqueeze(0)  # Add batch dim -> [1, N*N, feat_dim]
            
            print(f"After reshape:")
            print(f"a_feat shape: {a_feat.shape}")
            print(f"z_pair shape: {z_pair.shape}")
            
            # Get pooled features
            a_score = self.a_pooling(a_feat)  # [1, hidden_channels]
            z_pair_score = self.z_pair_pooling(z_pair)  # [1, hidden_channels]
            
            # Combine scores
            combined = torch.cat([a_score, z_pair_score], dim=-1)  # [1, 2*hidden_channels]
            pooled_features.append(combined)
        
        # Stack all samples in the batch
        pooled_features = torch.cat(pooled_features, dim=0)  # [batch_size, 2*hidden_channels]
        
        # Final classification
        return self.classifier(pooled_features)

def main():
    args = parse_args()

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

    # Split indices for train/eval without stratification first
    indices = np.arange(len(a_feats_paths))
    
    # Simple random split of files
    train_size = int(0.8 * len(indices))
    np.random.shuffle(indices)
    train_idx, eval_idx = indices[:train_size], indices[train_size:]
    
    print(f"\nSplit data into {len(train_idx)} training files and {len(eval_idx)} evaluation files")

    # Create datasets
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = FileChunkedDataset(
        [a_feats_paths[i] for i in train_idx],
        [z_pair_feats_paths[i] for i in train_idx],
        [label_paths[i] for i in train_idx],
        device=device
    )
    
    eval_dataset = FileChunkedDataset(
        [a_feats_paths[i] for i in eval_idx],
        [z_pair_feats_paths[i] for i in eval_idx],
        [label_paths[i] for i in eval_idx],
        device=device
    )

    # Create sequential batch samplers
    train_sampler = SequentialFileBatchSampler(
        train_dataset, 
        args.batch_size, 
        train_dataset.file_sizes,
        shuffle=True
    )
    
    eval_sampler = SequentialFileBatchSampler(
        eval_dataset, 
        args.batch_size, 
        eval_dataset.file_sizes,
        shuffle=False
    )

    # Create data loaders with the sequential samplers
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=confidence_collate_fn,
        num_workers=0
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_sampler=eval_sampler,
        collate_fn=confidence_collate_fn,
        num_workers=0
    )

    print(f"Created data loaders with {len(train_dataset)} training and {len(eval_dataset)} evaluation samples")

    # Now you can use train_loader and eval_loader in your training loop
    # Each batch will be (list_of_a_feats, list_of_z_pair_feats, labels)

    # Create model
    model = DualDeepSetClassifier(
        a_feat_dim=a_feat_dim,
        z_pair_feat_dim=z_pair_feat_dim,
        hidden_channels=args.hidden_channels,  # You can make this configurable via args
        num_heads=args.num_heads         # You can make this configurable via args
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Loss and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.pretrained_model is not None:
        model.load_state_dict(torch.load(args.pretrained_model, map_location=device))
        print(f"Loaded pretrained model from {args.pretrained_model}")
    else:
        best_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            for a_feats_batch, z_pair_feats_batch, yb in train_loader:
                # Move each tensor in the lists to device
                a_feats_batch = [a.to(device, dtype=torch.float32) for a in a_feats_batch]
                z_pair_feats_batch = [z.to(device, dtype=torch.float32) for z in z_pair_feats_batch]
                yb = yb.to(device)
                
                optimizer.zero_grad()
                out = model(a_feats_batch, z_pair_feats_batch)
                yb = yb.float()
                loss = criterion(out.squeeze(), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(yb)
            
            avg_loss = total_loss / len(train_loader.dataset)
            if (epoch + 1) % 100 == 0 or (epoch + 1) == args.epochs:
                print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_loss:.4f}")
            wandb.log({"train_loss": avg_loss, "epoch": epoch + 1})

            # Eval
            model.eval()
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
                    probs = torch.sigmoid(out.squeeze())
                    preds = (probs > 0.5).long()
                    all_probs.extend(probs.cpu().numpy().reshape(-1))
                    all_labels.extend(yb.cpu().numpy().reshape(-1))
                    correct += (preds == yb).sum().item()
                    total += yb.size(0)
            
            acc = correct / total
            if (epoch + 1) % 100 == 0 or (epoch + 1) == args.epochs:
                print(f"Epoch {epoch+1}/{args.epochs} - Eval Acc: {acc:.4f}")
            wandb.log({"eval_acc": acc, "epoch": epoch + 1})

            # Compute and log AUC
            try:
                if model.output_units == 1:
                    area_under_curve = roc_auc_score(all_labels, all_probs)
                else:
                    area_under_curve = roc_auc_score(np.array(all_labels), np.array(all_probs)[:,1], multi_class='ovr')
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

        print(f"Best model saved to {args.output} with acc {best_acc:.4f} at epoch {best_epoch}")
        # Load best model for evaluation/plotting
        model.load_state_dict(torch.load(args.output, map_location=device))
        model.eval()

    # --- Draw ROC curve using the best or loaded model ---
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for a_feats_batch, z_pair_feats_batch, yb in eval_loader:
            a_feats_batch = [a.to(device, dtype=torch.float32) for a in a_feats_batch]
            z_pair_feats_batch = [z.to(device, dtype=torch.float32) for z in z_pair_feats_batch]
            yb = yb.to(device)
            out = model(a_feats_batch, z_pair_feats_batch)
            if model.output_units == 1:
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
    # Binary ROC
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs[:,1])
    roc_auc = auc(fpr, tpr)

    # Save ROC curve data for later replotting
    np.savez('./output/roc_curve_classifier_only.npz', fpr=fpr, tpr=tpr, thresholds=thresholds, roc_auc=roc_auc)

    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig('./output/roc_curve_confidence_classifier.png')
    wandb.log({"roc_curve": wandb.Image('./output/roc_curve_confidence_classifier.png')})
    plt.close()
    
    wandb.finish()


if __name__ == "__main__":
    main()
