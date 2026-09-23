"""
Cấu hình đường dẫn và hằng số cho toàn bộ project.
Tất cả đường dẫn đều tính tương đối từ thư mục gốc project,
không hard-code đường dẫn cá nhân.
"""

from pathlib import Path

# ── Thư mục gốc project ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Dữ liệu ──────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# ── File dữ liệu gốc ────────────────────────────────────────────────
# Dataset gốc ở dạng CSV
RAW_DATA_CSV = RAW_DIR / "Online Retail.csv"

# ── Models ────────────────────────────────────────────────────────────
MODELS_DIR = PROJECT_ROOT / "models"

# ── Outputs ───────────────────────────────────────────────────────────
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
RULES_DIR = OUTPUTS_DIR / "rules"

def get_stage_dirs(stage_name: str):
    """
    Tạo và trả về bộ đôi đường dẫn (figures_dir, tables_dir) cho từng stage.
    Ví dụ: stage_name="data_understanding" ->
           outputs/figures/data_understanding/
           outputs/tables/data_understanding/
    """
    fig_dir = FIGURES_DIR / stage_name
    tbl_dir = TABLES_DIR / stage_name
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, tbl_dir

# Đường dẫn riêng cho stage 01 - Data Understanding
FIGURES_DATA_UNDERSTANDING, TABLES_DATA_UNDERSTANDING = get_stage_dirs("data_understanding")

# Đường dẫn riêng cho stage 02 - Data Preparation
FIGURES_DATA_PREPARATION, TABLES_DATA_PREPARATION = get_stage_dirs("data_preparation")

# Đường dẫn riêng cho stage 03 - Feature Engineering
FIGURES_FEATURE_ENGINEERING, TABLES_FEATURE_ENGINEERING = get_stage_dirs("feature_engineering")

# ── Docs ──────────────────────────────────────────────────────────────
DOCS_DIR = PROJECT_ROOT / "docs"

# ── App ───────────────────────────────────────────────────────────────
APP_DIR = PROJECT_ROOT / "app"

# ── Hằng số phân tích ────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
REPEAT_PURCHASE_WINDOW_DAYS = 90

# ── StockCode phi sản phẩm ───────────────────────────────────────────
# Các mã dịch vụ, phí, điều chỉnh — không phải sản phẩm vật lý
NON_PRODUCT_STOCK_CODES = {
    'POST', 'DOT', 'M', 'm', 'C2', 'D', 'S',
    'BANK CHARGES', 'AMAZONFEE', 'CRUK', 'B',
}

# ── Đảm bảo thư mục output tồn tại ──────────────────────────────────
for d in [FIGURES_DIR, TABLES_DIR, RULES_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# Duong dan rieng cho stage 04 - Customer Clustering
FIGURES_CLUSTERING, TABLES_CLUSTERING = get_stage_dirs("clustering")
MODELS_CLUSTERING_DIR = MODELS_DIR / "clustering"
MODELS_CLUSTERING_DIR.mkdir(parents=True, exist_ok=True)

# Duong dan rieng cho stage 06 - Association Rules
FIGURES_ASSOCIATION, TABLES_ASSOCIATION = get_stage_dirs("association_rules")
TABLES_ASSOCIATION_RULES_DIR = TABLES_ASSOCIATION
