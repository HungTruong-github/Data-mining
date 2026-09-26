import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from pathlib import Path

def train_and_compare_models(df, target_col='Is_Repeat_Buyer'):
    """
    Huấn luyện và so sánh các mô hình phân loại.
    """
    print("Đang chuẩn bị dữ liệu...")
    # Loại bỏ các cột định danh không có giá trị dự đoán
    X = df.drop(columns=['CustomerID', target_col], errors='ignore')
    
    # MỚI THÊM: Mã hóa One-hot cho các cột dạng chuỗi (vd: 'Country')
    X = pd.get_dummies(X, drop_first=True)
    
    # Xử lý missing values tạm thời (nếu có) bằng 0
    X = X.fillna(0) 
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    results = []
    trained_models = {}
    best_f1 = 0
    best_model_name = ""
    
    print("Đang huấn luyện các mô hình...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        })
        trained_models[name] = model
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

    # Lưu mô hình tốt nhất
    model_dir = Path(__file__).resolve().parents[1] / 'models' / 'classification'
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_dir / 'best_classifier.pkl'
    joblib.dump(trained_models[best_model_name], best_model_path)
    print(f"[OK] Đã lưu mô hình tốt nhất ({best_model_name}) tại {best_model_path}")
    
    results_df = pd.DataFrame(results).set_index('Model').round(4)
    return results_df, trained_models, X_test, y_test