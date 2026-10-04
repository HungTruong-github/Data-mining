import json

with open("notebooks/07_model_comparison_and_insights.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update Cell 1 imports
nb['cells'][1]['source'] = [
    "import sys, os\n",
    "sys.path.insert(0, os.path.abspath('..'))\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import json\n",
    "import joblib\n",
    "import matplotlib\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "from src.config import PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR\n",
    "from src.model_comparison import (\n",
    "    validate_inputs, build_clustering_comparison,\n",
    "    build_classification_comparison, build_association_comparison,\n",
    "    verify_model_artifacts, generate_report, generate_manifest,\n",
    "    export_all_step_07_outputs, generate_summary_json\n",
    ")\n",
    "from src.insights import generate_all_insights\n",
    "\n",
    "TABLES_MC = OUTPUTS_DIR / 'tables' / 'model_comparison'\n",
    "TABLES_MC.mkdir(parents=True, exist_ok=True)\n",
    "FIGURES_MC = OUTPUTS_DIR / 'figures' / 'model_comparison'\n",
    "FIGURES_MC.mkdir(parents=True, exist_ok=True)\n",
    "REPORTS_DIR = OUTPUTS_DIR / 'reports'\n",
    "REPORTS_DIR.mkdir(parents=True, exist_ok=True)\n",
    "EVIDENCE_DIR = OUTPUTS_DIR / 'evidence'\n",
    "EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "pd.set_option('display.max_columns', None)\n",
    "pd.set_option('display.float_format', lambda x: f'{x:.4f}')\n",
    "\n",
    "%matplotlib inline\n",
    "print('Setup complete.')"
]

# Update Cell 20 (export step)
nb['cells'][20]['source'] = [
    "# Centralized Export of all Step 07 synchronized outputs (tables, 10 figures, report, summary, manifest)\n",
    "export_res = export_all_step_07_outputs(\n",
    "    clust_comp=clustering_comp,\n",
    "    class_comp=classification_comp,\n",
    "    assoc_comp=assoc_comp,\n",
    "    seg_insights=insights.get('customer_segment_insights'),\n",
    "    action_plan=insights.get('customer_segment_action_plan'),\n",
    "    prod_insights=insights.get('product_association_insights'),\n",
    "    feat_insights=insights.get('classification_feature_insights'),\n",
    "    save_figures=True\n",
    ")\n",
    "print('[OK] Step 07 synchronized export complete.')\n",
    "\n",
    "# Check manifest validation status\n",
    "manifest = export_res['manifest']\n",
    "if manifest.get('validation_status') != 'PASS':\n",
    "    missing = manifest.get('missing_files', [])\n",
    "    raise RuntimeError(f\"Step 07 FAILED: pipeline_manifest.json validation_status is 'FAIL'. Missing files: {missing}\")\n",
    "print(f\"[OK] Manifest validated with status: PASS\")"
]

with open("notebooks/07_model_comparison_and_insights.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Updated notebooks/07_model_comparison_and_insights.ipynb successfully!")
