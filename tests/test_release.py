"""Release-only structural tests: no source imports, notebook cells or models."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import re
import subprocess
import sys
import unittest
import yaml

ROOT = Path(__file__).resolve().parents[1]

class ReleaseTests(unittest.TestCase):
    def test_a4_manifest_and_actual_structural_constants(self):
        a = yaml.safe_load((ROOT/'configs/ablation.yaml').read_text())
        p = yaml.safe_load((ROOT/'configs/morphonet_bt_a4.yaml').read_text())
        expected = dict(a['model']); expected.update(a['variants']['A4'])
        self.assertEqual(p['model'], expected)
        for key in ('seed', 'preprocessing', 'class_mapping', 'training'):
            self.assertEqual(p[key], a[key])
        tree = ast.parse((ROOT/'src/models/morphonet_bt.py').read_text())
        call = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'MODEL_CONFIG' for t in n.targets))
        actual = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
        self.assertEqual(actual, a['model'])
        ab = ast.parse((ROOT/'src/models/ablation.py').read_text())
        flags = next(ast.literal_eval(n.value) for n in ab.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'FLAGS' for t in n.targets))
        self.assertEqual(a['variants']['A4'], dict(zip(flags, [True]*4+[False])))
        self.assertIn('range(6)', ast.unparse(ab))
        self.assertIn('j < i', ast.unparse(ab))
        self.assertTrue(yaml.safe_load((ROOT/'configs/default.yaml').read_text())['model']['use_amgf'])

    def test_preserved_scientific_bytes(self):
        records = json.loads((ROOT/'tests/data/scientific_sha256.json').read_text())
        for path, digest in records.items():
            self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(), digest, path)
        self.assertEqual(len(list((ROOT/'results').rglob('*.csv'))), 21)

    def test_safe_convenience_entrypoints(self):
        for name, allowed in [('show_results.py', {'pathlib','csv','decimal'}), ('validate_release.py', {'pathlib','sys','unittest'})]:
            tree = ast.parse((ROOT/'scripts'/name).read_text())
            for node in tree.body:
                self.assertIsInstance(node, (ast.Expr, ast.Import, ast.ImportFrom, ast.FunctionDef, ast.If))
                if isinstance(node, ast.Expr): self.assertIsInstance(node.value, ast.Constant)
                if isinstance(node, ast.Import): self.assertTrue(all(n.name in allowed for n in node.names))
                if isinstance(node, ast.ImportFrom): self.assertIn(node.module, allowed)
                if isinstance(node, ast.If): self.assertEqual(ast.unparse(node.test), "__name__ == '__main__'")
            self.assertNotIn('torch', ast.unparse(tree))
        before = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*') if p.is_file()}
        result = subprocess.run([sys.executable, '-B', str(ROOT/'scripts/show_results.py')], capture_output=True, text=True, check=True)
        self.assertIn('Archived historical', result.stdout)
        for value in ('92.5914%', '91.5452%', '98.3899%', '95.2790%', '95.3466%', '98.7162%'):
            self.assertIn(value, result.stdout)
        after = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_yaml_and_all_markdown_links(self):
        for p in [*ROOT.rglob('*.yaml'), *ROOT.rglob('*.yml'), ROOT/'CITATION.cff']:
            self.assertIsInstance(yaml.safe_load(p.read_text()), dict)
        for p in ROOT.rglob('*.md'):
            for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', p.read_text(encoding='utf-8')):
                if '://' not in link and not link.startswith(('#','mailto:')):
                    self.assertTrue((p.parent/link.split('#')[0]).exists(), (p,link))
        citation = yaml.safe_load((ROOT/'CITATION.cff').read_text())
        self.assertEqual(citation['authors'][0]['orcid'], 'https://orcid.org/0009-0007-2906-2001')
        self.assertFalse({'doi','date-released','version','url'} & set(citation))

    def test_portable_paths_and_only_diagram_images(self):
        paths = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()]
        self.assertEqual(len(paths), len(set(p.casefold() for p in paths)))
        for path in paths:
            self.assertNotRegex(path, r'[<>:"|?*]')
            self.assertFalse(any(part.endswith((' ', '.')) for part in Path(path).parts))
            if Path(path).suffix.lower() in {'.png','.pdf','.jpg','.jpeg','.gif','.tiff','.bmp','.svg'}:
                self.assertIn(path, {'results/figures/A4_architecture.png','results/figures/A4_architecture.pdf'})
            self.assertNotIn(Path(path).suffix.lower(), {'.zip','.pyc','.npy','.npz','.pkl','.pickle','.parquet'})

    def test_scientific_manifest_coverage(self):
        recorded = set(json.loads((ROOT/'tests/data/scientific_sha256.json').read_text()))
        actual = {p.relative_to(ROOT).as_posix() for folder in ('src','configs','notebooks','results') for p in (ROOT/folder).rglob('*') if p.is_file() and (folder in ('src','configs') or p.suffix in ('.ipynb','.csv','.png','.pdf'))}
        actual.add('scripts/phase6_smoke.py')
        self.assertEqual(recorded, actual)
        self.assertFalse((ROOT/'notebooks/historical').exists())
        self.assertFalse((ROOT/'results/historical_diagnostics').exists())

if __name__ == '__main__':
    unittest.main()
