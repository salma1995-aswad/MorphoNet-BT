"""Static-only repository checks. Never import project modules or execute cells."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import re
import unittest
import yaml
import nbformat

ROOT = Path(__file__).resolve().parents[1]

class StaticRepositoryTests(unittest.TestCase):
    def test_notebook_schema_and_gates(self):
        files = list((ROOT / 'notebooks').rglob('*.ipynb'))
        self.assertEqual(len(files), 17)
        for p in files:
            n = nbformat.read(p, as_version=4)
            nbformat.validate(n)
            codes = [c for c in n.cells if c.cell_type == 'code']
            self.assertIn('CONFIRM_PUBLIC_WORKFLOW = False', codes[0].source)
            for c in codes:
                self.assertIsNone(c.execution_count)
                self.assertEqual(c.outputs, [])
                tree = ast.parse(c.source)
                start = 1 if isinstance(tree.body[0], ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CONFIRM_PUBLIC_WORKFLOW' for t in tree.body[0].targets) else 0
                gate = tree.body[start]
                self.assertIsInstance(gate, ast.If)
                self.assertIn('CONFIRM_PUBLIC_WORKFLOW', ast.unparse(gate.test))
                self.assertIsInstance(gate.body[0], ast.Raise)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id.startswith('CONFIRM_') for t in node.targets):
                        self.assertFalse(isinstance(node.value, ast.Constant) and node.value.value is True)
                self.assertNotRegex(c.source, r'[\u0600-\u06ff]')

    def test_python_syntax_and_local_import_structure(self):
        for p in ROOT.rglob('*.py'):
            tree = ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p.relative_to(ROOT)))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module == 'src' or node.module.startswith('src.'):
                        dest = ROOT.joinpath(*node.module.split('.'))
                        self.assertTrue(dest.is_dir() or dest.with_suffix('.py').is_file(), (p, node.module))
                    elif node.level and p.is_relative_to(ROOT / 'src'):
                        base = p.parent
                        for _ in range(node.level - 1):
                            base = base.parent
                        dest = base.joinpath(*node.module.split('.'))
                        self.assertTrue(dest.is_dir() or dest.with_suffix('.py').is_file(), (p, node.module))

    def test_configuration_and_schema(self):
        d = yaml.safe_load((ROOT / 'configs/default.yaml').read_text())
        a = yaml.safe_load((ROOT / 'configs/ablation.yaml').read_text())
        b = yaml.safe_load((ROOT / 'configs/baselines.yaml').read_text())
        self.assertEqual(d['seed'], 42)
        self.assertEqual(d['image_size'], 256)
        self.assertFalse(d['normalization']['clipping'])
        self.assertEqual(d['normalization']['method'], 'zscore')
        self.assertEqual([d['label_mapping'][i]['class_index'] for i in (1, 2, 3)], [0, 1, 2])
        self.assertEqual(a['variants']['A4'], dict(use_mss=True, use_morphology_supervision=True, use_baa=True, use_dual_path=True, use_amgf=False))
        self.assertTrue(d['model']['use_amgf'])  # Original A5 remains A5.
        self.assertEqual(a['folds'], [1, 2, 3, 4, 5])
        self.assertEqual(d['training']['loss'], dict(lambda_region=1., lambda_boundary=1., lambda_distance=.5, lambda_morphology=.3))
        self.assertFalse(b['pretrained'])
        self.assertIsNone(b['weights'])
        self.assertEqual(len(b['models']), 5)
        schema = json.loads((ROOT / 'docs/split_manifest_schema.json').read_text())
        source = (ROOT / 'src/data/kfold.py').read_text()
        tree = ast.parse(source)
        identity = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'IDENTITY' for t in n.targets))
        self.assertTrue(set(identity) <= set(schema['required']))
        self.assertEqual(schema['fold'], [1, 2, 3, 4, 5])

    def test_architecture_and_aggregation_source_contracts(self):
        a = (ROOT / 'src/models/ablation.py').read_text()
        self.assertIn('fused = .5 * (global_features + local)', a)
        self.assertIn('def forward(self, image, *, return_aux=False)', a)
        self.assertIn('needed = self.use_baa or (return_aux and self.use_morphology_supervision)', a)
        self.assertIn("morphology.get('boundary_logits_low')", a)
        agg = (ROOT / 'src/evaluation/analysis.py').read_text()
        self.assertIn("'float64'", agg)
        self.assertIn('groups[PROBS].mean()', agg)
        stats = (ROOT / 'src/phase14_statistics.py').read_text()
        self.assertIn('binomtest', stats)
        self.assertIn('BOOTSTRAP_ITERATIONS=10000', stats)
        self.assertIn('RANDOM_SEED=42', stats)

    def test_frozen_aggregate_bytes_and_transformed_source_hashes(self):
        records = json.loads((ROOT / 'tests/data/scientific_sha256.json').read_text())
        for path, digest in records.items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest, path)
        self.assertEqual(len(list((ROOT / 'results/tables').glob('*.csv'))), 21)

    def test_readme_links_and_required_documentation(self):
        for p in [ROOT / 'README.md', *list((ROOT / 'docs').glob('*.md'))]:
            for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', p.read_text(encoding='utf-8')):
                if '://' not in link and not link.startswith('#'):
                    self.assertTrue((p.parent / link.split('#')[0]).exists(), (p.name, link))
        for name in ['README.md','LICENSE','CITATION.cff','requirements.txt','environment.yml','.gitignore','.gitattributes','pyproject.toml','docs/installation.md','docs/dataset_setup.md','docs/reproducibility.md','docs/licensing_decisions.md']:
            self.assertTrue((ROOT / name).is_file(), name)

    def test_citation_truthfulness_pending_author_completion(self):
        c = yaml.safe_load((ROOT / 'CITATION.cff').read_text())
        self.assertEqual(c['cff-version'], '1.2.0')
        self.assertEqual(c['type'], 'software')
        self.assertIn('unpublished', c['abstract'].lower())
        self.assertNotIn('doi', c)
        self.assertEqual([(a['given-names'], a['family-names'], a['email']) for a in c['authors']], [('Salma', 'Aswad', 'salma.abdullah@uod.ac'), ('Serkan', 'Ozturk', 'Serkan@ergiyes.edu.tr')])

    def test_sensitive_files_paths_and_credentials(self):
        forbidden = {'.pt','.pth','.ckpt','.safetensors','.mat','.h5','.hdf5','.nii','.dcm','.pem','.key'}
        secret_patterns = [r'AKIA[0-9A-Z]{16}', r'gh[pousr]_[A-Za-z0-9]{30,}', r'-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----', r'(?i)(?:api_key|password|client_secret)\s*[:=]\s*[\"\x27][A-Za-z0-9_\-/+]{16,}[\"\x27]']
        for p in ROOT.rglob('*'):
            if not p.is_file():
                continue
            self.assertNotIn(p.suffix.lower(), forbidden, p.name)
            self.assertNotIn('__pycache__', p.parts)
            if p.suffix.lower() in {'.py','.ipynb','.md','.json','.yaml','.yml','.txt','.csv','.cff','.toml'}:
                text = p.read_text(encoding='utf-8-sig')
                self.assertNotRegex(text, r'[A-Za-z]:[\\/]+Users[\\/]|/home/[A-Za-z0-9_. -]+/|/Users/[A-Za-z0-9_. -]+/')
                self.assertNotRegex(text, r'[\u0600-\u06ff]')
                for pattern in secret_patterns:
                    self.assertNotRegex(text, pattern)
            if p.suffix == '.csv':
                with p.open(encoding='utf-8-sig', newline='') as f:
                    header = next(csv.reader(f))
                self.assertFalse({'PID','pid','sample_id','Sample_ID','patient_id'} & set(header), p.name)

if __name__ == '__main__':
    unittest.main()
