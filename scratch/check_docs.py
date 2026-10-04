import sys
sys.stdout.reconfigure(encoding='utf-8')
import re
from pathlib import Path

patterns = [
    r'34\s*tests',
    r'34/34',
    r'class_weight=[\'"]balanced[\'"]',
    r'13[,.]?369',
    r'267',
    r'extreme outliers',
    r'grid search',
    r'40\s*lượt fit',
]

for doc_path in list(Path('docs').glob('*.md')) + [Path('README.md'), Path('README_REVIEW.md')]:
    if not doc_path.exists(): continue
    text = doc_path.read_text(encoding='utf-8')
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            start = max(0, m.start() - 40)
            end = min(len(text), m.end() + 40)
            snippet = text[start:end].replace('\n', ' ')
            print(f'{doc_path} [{pat}]: ...{snippet}...')
