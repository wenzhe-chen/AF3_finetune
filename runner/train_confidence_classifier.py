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
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from protenix.model.modules.classifier import ConfidenceClassifier
import wandb
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Train Confidence Classifier")
    parser.add_argument('--feat_path', type=str, required=True, nargs='+', help='Path(s) to features .pt file(s)')
    parser.add_argument('--label_path', type=str, required=True, nargs='+', help='Path(s) to labels .pt file(s)')
    parser.add_argument('--ligand_length', type=int, required=True, help='Ligand length for classifier input dim')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--output', type=str, default='confidence_classifier.pt', help='Output model path')
    parser.add_argument('--number_of_chains', type=int, default=2, help='Number of chains')
    parser.add_argument('--use_intersted_atom_mask', action='store_true', help='Use interested atom mask')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience (epochs)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize wandb
    wandb.init(
        project="confidence_classifier",
        config=vars(args)
    )

    # Load features and labels from all provided paths and concatenate
    feat_list = []
    for path in args.feat_path:
        f = torch.load(path)
        if isinstance(f, np.ndarray):
            f = torch.from_numpy(f)
        feat_list.append(f)
    feats = torch.cat(feat_list, dim=0)

    label_list = []
    for path in args.label_path:
        l = torch.load(path)
        if isinstance(l, np.ndarray):
            l = torch.from_numpy(l)
        label_list.append(l)
    labels = torch.cat(label_list, dim=0)

    if labels.ndim > 1 and labels.shape[1] == 1:
        labels = labels.squeeze(1)

    print('feats',feats.shape)
    print('labels',labels.shape)

    # Train/eval split
    X_train, X_eval, y_train, y_eval = train_test_split(
        feats, labels, test_size=0.2, random_state=42, stratify=labels if labels.ndim == 1 else None
    )
    train_ds = TensorDataset(X_train, y_train)
    eval_ds = TensorDataset(X_eval, y_eval)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size)

    # Model
    model = ConfidenceClassifier(
        ligand_length=args.ligand_length,
        hidden_units=1024,
        output_units=len(torch.unique(labels)) if labels.ndim == 1 else labels.shape[1],
        number_of_chains=args.number_of_chains,
        use_intersted_atom_mask=args.use_intersted_atom_mask
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Loss and optimizer
    if model.output_units == 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, dtype=torch.float32), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            if model.output_units == 1:
                yb = yb.float()
                loss = criterion(out.squeeze(), yb)
            else:
                if yb.ndim > 1:
                    yb = yb.argmax(dim=1)
                loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
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
            for xb, yb in eval_loader:
                xb, yb = xb.to(device, dtype=torch.float32), yb.to(device)
                out = model(xb)
                if model.output_units == 1:
                    probs = torch.sigmoid(out.squeeze())
                    preds = (probs > 0.5).long()
                    all_probs.extend(probs.cpu().numpy().reshape(-1))
                    all_labels.extend(yb.cpu().numpy().reshape(-1))
                else:
                    probs = torch.softmax(out, dim=1)
                    preds = out.argmax(dim=1)
                    if yb.ndim > 1:
                        yb = yb.argmax(dim=1)
                    all_probs.extend(probs.cpu().numpy())
                    all_labels.extend(yb.cpu().numpy())
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
                #print('all_labels',np.array(all_labels).shape)
                #print('all_probs',np.array(all_probs).shape)
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
    
    # --- Draw ROC curve using the best model ---
    # Load best model
    model.load_state_dict(torch.load(args.output, map_location=device))
    model.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for xb, yb in eval_loader:
            xb, yb = xb.to(device, dtype=torch.float32), yb.to(device)
            out = model(xb)
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
    fpr, tpr, _ = roc_curve(all_labels, all_probs[:,1])
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig('./output/roc_curve.png')
    wandb.log({"roc_curve": wandb.Image('./output/roc_curve.png')})
    plt.close()
    
    wandb.finish()


if __name__ == "__main__":
    main()
