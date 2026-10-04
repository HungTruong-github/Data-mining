import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent

notebooks_to_run = [
    PROJECT_ROOT / "notebooks" / "05_repeat_purchase_classification.ipynb",
    PROJECT_ROOT / "notebooks" / "07_model_comparison_and_insights.ipynb",
]

for nb_path in notebooks_to_run:
    print(f"\n>>> Executing notebook: {nb_path.name}...")
    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(nb_path.parent)}})
    client.execute()
    nbformat.write(nb, nb_path)
    print(f"[OK] {nb_path.name} executed cleanly and saved.")

print("\nAll target notebooks executed successfully!")
