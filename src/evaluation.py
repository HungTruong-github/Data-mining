import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path

def evaluate_classification_model(model, X_test, y_test, model_name="Best Model", save_dir=None):
    """
    Vẽ Confusion Matrix và in Báo cáo phân loại.
    """
    y_pred = model.predict(X_test)
    
    # 1. In Classification Report
    print(f"--- ĐÁNH GIÁ MÔ HÌNH: {model_name} ---")
    print(classification_report(y_test, y_pred))
    
    # 2. Vẽ Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Không mua lại', 'Mua lại'], 
                yticklabels=['Không mua lại', 'Mua lại'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Thực tế (Actual)')
    plt.xlabel('Dự đoán (Predicted)')
    
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        
    plt.show()