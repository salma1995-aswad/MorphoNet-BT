"""Windows-safe entry point for the bounded one-batch diagnostic only."""
from pathlib import Path
import sys
import json
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.training.trainer import preflight
from src.training.setup_validation import single_batch_smoke, pure_logic_checks


def main():
    config=yaml.safe_load((ROOT/'configs/default.yaml').read_text(encoding='utf-8'))
    info=preflight(ROOT,config,1)
    report=single_batch_smoke(ROOT,config,info)
    report['logic_checks']=pure_logic_checks()
    (ROOT/'results/training_setup/training_smoke_test.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('PASS: single disposable batch, AMP/AdamW/cosine, checkpoint/RNG resume. No full training.')


if __name__=='__main__': main()
