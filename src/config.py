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

# ── Docs ──────────────────────────────────────────────────────────────
DOCS_DIR = PROJECT_ROOT / "docs"

# ── App ───────────────────────────────────────────────────────────────
APP_DIR = PROJECT_ROOT / "app"

# ── Hằng số phân tích ────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
REPEAT_PURCHASE_WINDOW_DAYS = 90

# ── Đảm bảo thư mục output tồn tại ──────────────────────────────────
for d in [FIGURES_DIR, TABLES_DIR, RULES_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
