"""
train.py
========
Training and evaluation utilities for DSA-LoRA:
Domain-Shift-Aware Parameter-Efficient Fine-Tuning
for Malaria Detection in Low-Resource Clinical Settings.
"""

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


# ── TRAINING ──────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Run one full training epoch.

    Applies gradient clipping (max_norm=1.0) to stabilise ViT fine-tuning,
    which can be sensitive to large gradient updates especially with
    very small datasets.

    Args:
        model:     the model being fine-tuned
        loader:    DataLoader for training data
        criterion: loss function (CrossEntropyLoss)
        optimizer: optimizer (AdamW)
        device:    torch device

    Returns:
        avg_loss: mean loss over all batches
        accuracy: training accuracy over the epoch
    """
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping — important for stable ViT fine-tuning
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


# ── EVALUATION ────────────────────────────────────────────────────────────────

def evaluate(model, loader, device):
    """
    Evaluate model on a dataset.

    Reports three metrics standard in medical imaging literature:
        - AUC-ROC: primary metric, robust to class imbalance
        - F1 Score: balances precision and recall
        - Accuracy: for comparison with prior work

    Handles the edge case where only one class is present in the
    evaluation set — returns nan for AUC in that case rather than
    crashing, with a printed warning.

    Args:
        model:  model to evaluate
        loader: DataLoader for evaluation data
        device: torch device

    Returns:
        accuracy: classification accuracy
        f1:       binary F1 score
        auc:      AUC-ROC score (nan if only one class present)
    """
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds   = torch.argmax(outputs, dim=1).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds,
                        average='binary', zero_division=0)

    classes_present = set(all_labels)
    if len(classes_present) < 2:
        print(f"  WARNING: evaluation set contains only "
              f"class(es) {classes_present} — AUC undefined")
        auc = float('nan')
    else:
        auc = roc_auc_score(all_labels, all_probs)

    return accuracy, f1, auc


# ── TRAINING LOOP ─────────────────────────────────────────────────────────────

def train_with_early_stopping(model, train_loader, val_loader,
                               criterion, optimizer, scheduler,
                               device, epochs=15, metric='auc'):
    """
    Full training loop with best model tracking.

    Saves the model state that achieves the highest validation AUC
    (or F1 as fallback if AUC is nan). Returns the best model state
    for final evaluation.

    Args:
        model:        model to train
        train_loader: DataLoader for training data
        val_loader:   DataLoader for validation data
        criterion:    loss function
        optimizer:    optimizer
        scheduler:    learning rate scheduler
        device:       torch device
        epochs:       number of training epochs
        metric:       metric to track for best model ('auc' or 'f1')

    Returns:
        model:       model loaded with best checkpoint weights
        history:     dict of training metrics per epoch
        best_epoch:  epoch at which best validation metric was achieved
        best_metric: best validation metric value
    """
    history = {
        'train_loss': [], 'train_acc': [],
        'val_acc':    [], 'val_f1':    [], 'val_auc': []
    }

    best_metric_val = 0.0
    best_epoch      = 0
    best_state      = None

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_acc, val_f1, val_auc = evaluate(model, val_loader, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)

        # Track best model — use AUC, fall back to F1 if AUC is nan
        current = val_auc if not np.isnan(val_auc) else val_f1
        if current > best_metric_val:
            best_metric_val = current
            best_epoch      = epoch
            best_state      = {k: v.clone() if torch.is_tensor(v) else v
                               for k, v in model.state_dict().items()}

        if epoch % 5 == 0:
            print(f"  Epoch {epoch}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val AUC: {val_auc:.4f} | "
                  f"Val F1: {val_f1:.4f}")

    # Load best checkpoint for final evaluation
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, best_epoch, best_metric_val


# ── OPTIMIZER & SCHEDULER FACTORY ────────────────────────────────────────────

def build_optimizer_scheduler(model, lr=1e-4, weight_decay=1e-4, epochs=15):
    """
    Build AdamW optimizer and cosine annealing scheduler.

    Only parameters with requires_grad=True are passed to the optimizer,
    ensuring frozen layers (including frozen LoRA base weights) are not
    included in parameter updates.

    Args:
        model:        model with some parameters frozen
        lr:           learning rate (default: 1e-4)
        weight_decay: L2 regularisation (default: 1e-4)
        epochs:       total training epochs for scheduler

    Returns:
        optimizer: AdamW optimizer
        scheduler: CosineAnnealingLR scheduler
    """
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-7
    )
    return optimizer, scheduler
