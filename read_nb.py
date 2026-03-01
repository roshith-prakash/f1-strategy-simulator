import json, sys
sys.stdout.reconfigure(encoding='utf-8')
with open('f1_laptime_model.ipynb','r',encoding='utf-8') as f:
    nb = json.load(f)
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        print(f'\n===== CODE CELL {i} =====')
        print(''.join(c.get('source', [])))
