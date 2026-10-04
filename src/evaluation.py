"""
Module evaluation: Visualization and reporting for classification results.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)


def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    """Plot and optionally save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Repeat (0)', 'Repeat (1)'],
                yticklabels=['No Repeat (0)', 'Repeat (1)'], ax=ax,
                annot_kws={'fontsize': 14})
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_roc_curves(trained_pipelines, X_test, y_test, save_path=None):
    """Plot ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(10, 8))
    for name, pipeline in trained_pipelines.items():
        try:
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={roc_auc_val:.3f})')
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (AUC=0.500)')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_precision_recall_curves(trained_pipelines, X_test, y_test, save_path=None):
    """Plot Precision-Recall curves for all models."""
    fig, ax = plt.subplots(figsize=(10, 8))
    for name, pipeline in trained_pipelines.items():
        try:
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
            ap = average_precision_score(y_test, y_proba)
            ax.plot(recall_vals, precision_vals, linewidth=2, label=f'{name} (AP={ap:.3f})')
        except Exception:
            pass
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves — Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_model_comparison(comparison_df, save_path=None):
    """Plot bar chart comparing key metrics across models."""
    metrics_to_plot = ['test_f1', 'test_precision', 'test_recall', 'test_roc_auc', 'test_balanced_accuracy']
    available = [m for m in metrics_to_plot if m in comparison_df.columns]
    
    plot_df = comparison_df.set_index('model')[available]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    plot_df.plot(kind='bar', ax=ax, edgecolor='white', linewidth=1.5)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')
    ax.legend(fontsize=9, loc='lower right')
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_feature_importance(fi_df, model_name, top_n=15, save_path=None):
    """Plot feature importance bar chart."""
    top = fi_df.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top)))
    ax.barh(range(len(top)), top['importance'].values, color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top['feature'].values, fontsize=10)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Feature Importance — {model_name}', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def save_classification_report_csv(y_true, y_pred, save_path):
    """Save sklearn classification_report as CSV."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    df = pd.DataFrame(report).transpose().round(4)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path)
    return df
