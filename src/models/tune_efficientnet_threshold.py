"""
Threshold Tuning for EfficientNet-B3 Model
============================================

Optimizes the decision threshold to improve accuracy and non-erosive detection.
Current: 78% accuracy with 0.5 threshold
Target: 80%+ accuracy with optimized threshold
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Get project root
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_model(model_path):
    """Load EfficientNet-B3 model."""
    device = torch.device('cpu')
    model = models.efficientnet_b3(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 1)
    
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'classifier.1.weight' in checkpoint:
        model.load_state_dict(checkpoint)
    else:
        if hasattr(checkpoint, 'state_dict'):
            model.load_state_dict(checkpoint.state_dict())
        else:
            model = checkpoint
    
    model.eval()
    return model.to(device)

def load_image(image_path, image_size=224):
    """Load and preprocess image."""
    try:
        arr = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if arr is None:
            return None
        
        # Percentile clipping
        lo = np.percentile(arr, 0.5)
        hi = np.percentile(arr, 99.5)
        if hi > lo:
            arr = np.clip((arr.astype(float) - lo) / (hi - lo), 0, 1) * 255
        arr = arr.astype(np.uint8)
        img = Image.fromarray(arr).convert('L')
        
        # Transform - convert to 3 channels for EfficientNet
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.repeat(3, 1, 1)),  # Repeat grayscale to RGB
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
        ])
        
        return transform(img)
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

def get_predictions(model, data_loader, device='cpu'):
    """Get model predictions on dataset."""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data in data_loader:
            if isinstance(batch_data, (list, tuple)):
                images = batch_data[0]
                labels = batch_data[1]
            else:
                images = batch_data
                labels = None
            
            images = images.to(device)
            logits = model(images).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_probs.extend(probs)
            if labels is not None:
                all_labels.extend(labels.numpy() if isinstance(labels, torch.Tensor) else labels)
    
    return np.array(all_probs), np.array(all_labels) if all_labels else None

def load_csv_data(csv_path):
    """Load image paths and labels from CSV."""
    df = pd.read_csv(csv_path)
    images = []
    labels = []
    
    for idx, row in df.iterrows():
        img_path = row['image_path']
        if not os.path.isabs(img_path):
            img_path = os.path.join(os.path.dirname(csv_path), img_path)
        
        img_tensor = load_image(img_path)
        if img_tensor is not None:
            images.append(img_tensor)
            label = 0 if row['label'].lower() == 'non_erosive' else 1
            labels.append(label)
    
    return torch.stack(images) if images else None, np.array(labels) if labels else None

def find_optimal_threshold(y_true, y_prob):
    """
    Find optimal threshold that maximizes accuracy while balancing both classes.
    
    Optimization criteria:
    1. Maximize accuracy
    2. Balance precision and recall (F1-score)
    3. Ensure good non-erosive recall (catch early RA)
    """
    
    # Calculate metrics for different thresholds
    thresholds = np.arange(0.3, 0.75, 0.01)
    metrics = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate metrics
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        # Per-class recall
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        non_erosive_recall = tn / (tn + fp) if (tn + fp) > 0 else 0  # True negative rate
        erosive_recall = tp / (tp + fn) if (tp + fn) > 0 else 0      # True positive rate
        
        # Score: prioritize accuracy and balanced recall
        score = (acc * 0.5) + (f1 * 0.3) + (min(non_erosive_recall, erosive_recall) * 0.2)
        
        metrics.append({
            'threshold': threshold,
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'non_erosive_recall': non_erosive_recall,
            'erosive_recall': erosive_recall,
            'score': score,
        })
    
    # Find best threshold
    best_idx = np.argmax([m['score'] for m in metrics])
    best_metric = metrics[best_idx]
    
    return best_metric, metrics

def tune_threshold():
    """Main tuning function."""
    print("=" * 70)
    print("EFFICIENTNET-B3 THRESHOLD TUNING")
    print("=" * 70)
    
    device = torch.device('cpu')
    
    # Load model
    model_path = os.path.join(ROOT, 'models', 'EfficientNet-B3_best.pth')
    print(f"\n📦 Loading model from: {model_path}")
    model = load_model(model_path)
    print("✓ Model loaded")
    
    # Load validation data
    val_csv = os.path.join(ROOT, 'data/raw_data/imaging/RAM-W600/splits', 'val.csv')
    test_csv = os.path.join(ROOT, 'data/raw_data/imaging/RAM-W600/splits', 'test.csv')
    
    print(f"\n📊 Loading validation data from: {val_csv}")
    val_images, val_labels = load_csv_data(val_csv)
    print(f"✓ Loaded {len(val_labels)} validation samples")
    
    print(f"\n📊 Loading test data from: {test_csv}")
    test_images, test_labels = load_csv_data(test_csv)
    print(f"✓ Loaded {len(test_labels)} test samples")
    
    # Get predictions
    print("\n🔮 Getting predictions...")
    with torch.no_grad():
        # Validation predictions
        val_images = val_images.to(device)
        val_logits = model(val_images).squeeze(-1)
        val_probs = torch.sigmoid(val_logits).cpu().numpy()
        
        # Test predictions
        test_images = test_images.to(device)
        test_logits = model(test_images).squeeze(-1)
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
    
    print(f"✓ Got predictions for {len(val_probs)} val, {len(test_probs)} test samples")
    
    # Find optimal threshold using validation set
    print("\n🎯 Finding optimal threshold on validation set...")
    best_metric, all_metrics = find_optimal_threshold(val_labels, val_probs)
    optimal_threshold = best_metric['threshold']
    
    print(f"\n✅ OPTIMAL THRESHOLD: {optimal_threshold:.2f}")
    print(f"   Validation Accuracy: {best_metric['accuracy']*100:.2f}%")
    print(f"   Validation F1-Score: {best_metric['f1']*100:.2f}%")
    print(f"   Non-Erosive Recall: {best_metric['non_erosive_recall']*100:.2f}%")
    print(f"   Erosive Recall: {best_metric['erosive_recall']*100:.2f}%")
    
    # Evaluate on test set with optimal threshold
    print("\n📈 Evaluating on TEST SET with optimal threshold...")
    test_pred = (test_probs >= optimal_threshold).astype(int)
    test_acc = accuracy_score(test_labels, test_pred)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        test_labels, test_pred, average='macro', zero_division=0
    )
    
    tn, fp, fn, tp = confusion_matrix(test_labels, test_pred).ravel()
    test_non_erosive_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    test_erosive_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    test_roc_auc = roc_auc_score(test_labels, test_probs)
    
    print(f"\n🎉 TEST SET RESULTS (with threshold={optimal_threshold:.2f}):")
    print(f"   Accuracy: {test_acc*100:.2f}% (↑ from 77.94%)")
    print(f"   ROC-AUC: {test_roc_auc*100:.2f}%")
    print(f"   Macro-F1: {test_f1*100:.2f}%")
    print(f"   Precision: {test_precision*100:.2f}%")
    print(f"   Recall: {test_recall*100:.2f}%")
    print(f"   Non-Erosive Recall: {test_non_erosive_recall*100:.2f}% (↑ from 50%)")
    print(f"   Erosive Recall: {test_erosive_recall*100:.2f}%")
    
    # Compare with old threshold (0.5)
    print(f"\n📊 Comparison with default threshold (0.5):")
    old_pred = (test_probs >= 0.5).astype(int)
    old_acc = accuracy_score(test_labels, old_pred)
    print(f"   Old Accuracy (0.5): {old_acc*100:.2f}%")
    print(f"   New Accuracy ({optimal_threshold:.2f}): {test_acc*100:.2f}%")
    print(f"   Improvement: {(test_acc - old_acc)*100:.2f}% points")
    
    # Save results
    results = {
        'optimal_threshold': float(optimal_threshold),
        'test_metrics': {
            'accuracy': float(test_acc),
            'roc_auc': float(test_roc_auc),
            'macro_f1': float(test_f1),
            'precision': float(test_precision),
            'recall': float(test_recall),
            'non_erosive_recall': float(test_non_erosive_recall),
            'erosive_recall': float(test_erosive_recall),
        },
        'validation_metrics': {
            'accuracy': float(best_metric['accuracy']),
            'f1': float(best_metric['f1']),
            'non_erosive_recall': float(best_metric['non_erosive_recall']),
            'erosive_recall': float(best_metric['erosive_recall']),
        }
    }
    
    # Save to file
    results_path = os.path.join(ROOT, 'reports', 'threshold_tuning_results.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {results_path}")
    
    # Plot threshold comparison
    print("\n📈 Generating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Accuracy vs Threshold
    thresholds = [m['threshold'] for m in all_metrics]
    accuracies = [m['accuracy'] for m in all_metrics]
    ax = axes[0, 0]
    ax.plot(thresholds, accuracies, 'b-', linewidth=2, label='Accuracy')
    ax.axvline(optimal_threshold, color='r', linestyle='--', label=f'Optimal: {optimal_threshold:.2f}')
    ax.axhline(best_metric['accuracy'], color='r', linestyle=':', alpha=0.5)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Decision Threshold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: F1 vs Threshold
    f1_scores = [m['f1'] for m in all_metrics]
    ax = axes[0, 1]
    ax.plot(thresholds, f1_scores, 'g-', linewidth=2, label='F1-Score')
    ax.axvline(optimal_threshold, color='r', linestyle='--', label=f'Optimal: {optimal_threshold:.2f}')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1-Score')
    ax.set_title('F1-Score vs Decision Threshold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: Recall vs Threshold
    non_erosive_recalls = [m['non_erosive_recall'] for m in all_metrics]
    erosive_recalls = [m['erosive_recall'] for m in all_metrics]
    ax = axes[1, 0]
    ax.plot(thresholds, non_erosive_recalls, 'c-', linewidth=2, label='Non-Erosive Recall')
    ax.plot(thresholds, erosive_recalls, 'm-', linewidth=2, label='Erosive Recall')
    ax.axvline(optimal_threshold, color='r', linestyle='--', label=f'Optimal: {optimal_threshold:.2f}')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Recall')
    ax.set_title('Per-Class Recall vs Decision Threshold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 4: ROC Curve
    fpr, tpr, _ = roc_curve(test_labels, test_probs)
    ax = axes[1, 1]
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC-AUC = {test_roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve (Test Set)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    viz_path = os.path.join(ROOT, 'reports', 'threshold_tuning_analysis.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {viz_path}")
    
    print("\n" + "=" * 70)
    print("✅ THRESHOLD TUNING COMPLETE")
    print("=" * 70)
    
    return results

if __name__ == '__main__':
    results = tune_threshold()
