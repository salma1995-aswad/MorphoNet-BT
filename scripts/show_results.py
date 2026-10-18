"""Display archived aggregate results only; no experiment or filesystem writes."""
from pathlib import Path
import csv
from decimal import Decimal

def read_tables(root):
    tables = {}
    for name in ('T03_slice_comparison', 'T04_patient_comparison', 'T02_complexity'):
        with (root / 'results' / 'tables' / (name + '.csv')).open(encoding='utf-8-sig', newline='') as stream:
            tables[name] = list(csv.DictReader(stream))
    return tables

def main():
    tables = read_tables(Path(__file__).resolve().parents[1])
    print('Archived historical results; exploratory pooled out-of-fold estimates. No inference performed.')
    for name, rows in tables.items():
        print('\n' + name)
        fields = list(rows[0])
        print(' | '.join(fields))
        for row in rows:
            print(' | '.join(row[field] for field in fields))
        if name != 'T02_complexity':
            a4 = next(row for row in rows if row['Model'] == 'MorphoNet-BT A4')
            print('A4 percentages: ' + ', '.join(key + '=' + format(Decimal(a4[key]) * 100, '.4f') + '%' for key in ('Accuracy', 'Macro F1', 'Macro ROC-AUC')))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
