import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
with open('notebooks/05_repeat_purchase_classification.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    print(f"\n{'='*20} CELL {i} [{cell['cell_type']}] {'='*20}")
    print(''.join(cell['source']))
