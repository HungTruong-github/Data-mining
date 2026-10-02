import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import time
import subprocess
from pathlib import Path

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

PROJECT_ROOT = Path(__file__).resolve().parent

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}")
    print(f"{text.center(60)}")
    print(f"{'=' * 60}{Colors.ENDC}\n")

def run_script(script_path, step_name):
    print(f"{Colors.OKBLUE}>>> Bắt đầu bước: {step_name}{Colors.ENDC}")
    print(f"Executing: python {script_path.relative_to(PROJECT_ROOT)}")
    start_time = time.time()
    
    try:
        process = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            check=True,
            text=True
        )
        elapsed = time.time() - start_time
        print(f"{Colors.OKGREEN}[THÀNH CÔNG] {step_name} hoàn tất trong {elapsed:.2f} giây.{Colors.ENDC}\n")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"{Colors.FAIL}[LỖI] {step_name} thất bại sau {elapsed:.2f} giây (Exit code: {e.returncode}).{Colors.ENDC}")
        print(f"{Colors.WARNING}Pipeline dừng lại do lỗi.{Colors.ENDC}")
        return False

def check_outputs(output_files):
    print(f"{Colors.BOLD}--- Kiểm tra kết quả Output ---{Colors.ENDC}")
    all_exist = True
    for file_path in output_files:
        path = PROJECT_ROOT / file_path
        if path.exists() and path.stat().st_size > 0:
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  [OK] {file_path} ({size_mb:.2f} MB)")
        else:
            print(f"  {Colors.FAIL}[MISSING or EMPTY] {file_path}{Colors.ENDC}")
            all_exist = False
    return all_exist

def main():
    print_header("ONLINE RETAIL DATA MINING PIPELINE")
    
    pipeline_steps = [
        {
            "name": "01. Data Understanding",
            "script": "notebooks/run_01_data_understanding.py",
            "outputs": [
                "outputs/tables/data_understanding/descriptive_statistics.csv",
                "outputs/tables/data_understanding/raw_summary.csv"
            ]
        },
        {
            "name": "02. EDA and Cleaning",
            "script": "notebooks/run_02_eda_and_cleaning.py",
            "outputs": [
                "data/interim/cleaned_transactions.csv",
                "data/interim/product_transactions.csv"
            ]
        },
        {
            "name": "03. Feature Engineering RFM",
            "script": "notebooks/run_03_feature_engineering.py",
            "outputs": [
                "data/processed/rfm_customer_features.csv",
                "data/processed/repeat_purchase_features.csv"
            ]
        },
        {
            "name": "04. Customer Clustering",
            "script": "notebooks/run_04_customer_clustering.py",
            "outputs": [
                "data/processed/customer_clusters.csv",
                "outputs/tables/clustering/clustering_algorithm_comparison.csv",
                "outputs/tables/clustering/cluster_profiles.csv",
                "models/clustering/clustering_model.pkl",
                "models/clustering/clustering_config.pkl"
            ]
        },
        {
            "name": "05. Repeat Purchase Classification",
            "script": "notebooks/run_05_repeat_purchase_classification.py",
            "outputs": [
                "outputs/tables/classification/model_comparison.csv",
                "outputs/tables/classification/cv_results.csv",
                "outputs/tables/classification/test_predictions.csv",
                "outputs/tables/classification/feature_importance.csv",
                "models/classification/best_classifier_pipeline.joblib",
                "models/classification/classification_metadata.json"
            ]
        },
        {
            "name": "06. Association Rules",
            "script": "notebooks/run_06_association_rules.py",
            "outputs": [
                "outputs/tables/association_rules/association_algorithm_comparison.csv",
                "outputs/tables/association_rules/selected_association_rules.csv",
                "outputs/tables/association_rules/association_business_insights.csv",
                "outputs/tables/association_rules/association_rules_equivalence_audit.csv"
            ]
        },
        {
            "name": "07. Model Comparison and Insights",
            "script": "notebooks/run_07_model_comparison_and_insights.py",
            "outputs": [
                "outputs/tables/model_comparison/clustering_model_comparison.csv",
                "outputs/tables/model_comparison/classification_model_comparison.csv",
                "outputs/tables/model_comparison/association_rules_comparison.csv",
                "outputs/tables/insights/customer_segment_insights.csv",
                "outputs/tables/insights/customer_segment_action_plan.csv",
                "outputs/tables/insights/product_association_insights.csv",
                "outputs/tables/insights/classification_feature_insights.csv",
                "outputs/reports/07_model_comparison_and_insights.md",
                "outputs/reports/07_model_comparison_and_insights_summary.json",
                "outputs/evidence/pipeline_manifest.json"
            ]
        }
    ]
    
    total_start = time.time()
    
    for step in pipeline_steps:
        script_path = PROJECT_ROOT / step["script"]
        if not script_path.exists():
            print(f"{Colors.FAIL}[LỖI] Không tìm thấy file script: {step['script']}{Colors.ENDC}")
            sys.exit(1)
            
        success = run_script(script_path, step["name"])
        if not success:
            sys.exit(1)
            
        outputs_ok = check_outputs(step["outputs"])
        if not outputs_ok:
            print(f"{Colors.FAIL}[LỖI] Dừng pipeline vì thiếu hoặc rỗng file output của bước {step['name']}{Colors.ENDC}")
            sys.exit(1)

        print("-" * 40 + "\n")
        
    total_end = time.time()
    total_elapsed = total_end - total_start
    print_header(f"PIPELINE HOÀN TẤT THÀNH CÔNG TỔNG THỜI GIAN: {total_elapsed:.2f}s")
    
if __name__ == "__main__":
    main()
