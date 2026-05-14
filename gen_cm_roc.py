import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                              roc_curve, auc, classification_report)

# Load
probs  = np.load('outputs/results/gmu_val_probs.npy')
labels = np.load('outputs/results/gmu_val_labels.npy')

preds = (probs >= 0.5).astype(int)

print(f"Total samples: {len(labels)}")
print(f"Recurrence=1: {labels.sum()}, Non-recurrence=0: {(labels==0).sum()}")
print(f"\nClassification Report:")
print(classification_report(labels, preds,
      target_names=['Non-Recurrence', 'Recurrence']))

# ── Confusion Matrix ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm = confusion_matrix(labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Non-Recurrence', 'Recurrence'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title('GMU Confusion Matrix\n(5-fold CV, n=169, threshold=0.5)',
                   fontsize=12, fontweight='bold', pad=12)
axes[0].set_xlabel('Predicted Label', fontsize=11)
axes[0].set_ylabel('True Label', fontsize=11)

# Add counts and percentages
total = cm.sum()
for i in range(2):
    for j in range(2):
        pct = cm[i,j]/total*100
        axes[0].text(j, i, f'\n({pct:.1f}%)',
                    ha='center', va='center', fontsize=9, color='gray')

# ── ROC Curve ─────────────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(labels, probs)
roc_auc = auc(fpr, tpr)

axes[1].plot(fpr, tpr, color='#1565C0', lw=2.5,
             label=f'GMU (AUC = {roc_auc:.4f})')
axes[1].plot([0,1],[0,1], 'k--', lw=1, alpha=0.5, label='Random (AUC = 0.50)')
axes[1].fill_between(fpr, tpr, alpha=0.08, color='#1565C0')
axes[1].set_xlim([0,1]); axes[1].set_ylim([0,1.02])
axes[1].set_xlabel('False Positive Rate', fontsize=11)
axes[1].set_ylabel('True Positive Rate', fontsize=11)
axes[1].set_title('GMU ROC Curve\n(5-fold CV, n=169)',
                   fontsize=12, fontweight='bold', pad=12)
axes[1].legend(loc='lower right', fontsize=10)
axes[1].grid(alpha=0.3)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('outputs/figures/gmu_confusion_matrix_roc.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("\nSaved: outputs/figures/gmu_confusion_matrix_roc.png")