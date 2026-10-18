"""Public static release checks; no internal approval evidence or experiments."""
from pathlib import Path
import hashlib
import json
import unittest
import yaml

ROOT = Path(__file__).resolve().parents[1]

class FinalReleaseTests(unittest.TestCase):
    def test_final_license_terms(self):
        text = (ROOT/'LICENSE').read_text()
        self.assertTrue(text.startswith('MIT License\n\nCopyright (c) 2026 Salma Aswad and Serkan Ozturk\n'))
        body = text[text.index('Permission is hereby granted'):].strip()
        record = json.loads((ROOT/'tests/data/license_integrity.json').read_text())
        self.assertEqual(hashlib.sha256(body.encode()).hexdigest(), record['terms_sha256'])
        self.assertNotIn('PENDING', text)
        self.assertNotIn('non-commercial', body.lower())

    def test_author_metadata(self):
        c = yaml.safe_load((ROOT/'CITATION.cff').read_text())
        one = 'Computer Engineering, Erciyes University, Kayseri, Turkey'
        two = 'Department of Computer Science, College of Science, University of Duhok, Duhok 42001, Iraq'
        self.assertEqual(c['authors'][0]['affiliation'], one+'; '+two)
        self.assertEqual(c['authors'][1]['affiliation'], one)
        self.assertEqual(c['authors'][1]['email'], 'Serkan@ergiyes.edu.tr')
        self.assertEqual(c['authors'][0]['email'], 'salma.abdullah@uod.ac')
        self.assertEqual(c['authors'][0]['orcid'], 'https://orcid.org/0009-0007-2906-2001')
        self.assertNotIn('orcid', c['authors'][1])
        self.assertEqual(c['license'], 'MIT')
        self.assertIn('unpublished', c['abstract'].lower())

    def test_release_hygiene(self):
        for p in ROOT.rglob('*'):
            if not p.is_file():
                continue
            name = p.name.lower()
            self.assertFalse(name.startswith('.env'), p)
            self.assertNotIn(name, {'license_status.md','desktop.ini','thumbs.db','.ds_store'})
            self.assertFalse(any(part in {'__pycache__','.git','.ipynb_checkpoints','.venv','internal','private'} for part in p.parts))
            self.assertFalse(any(marker in name for marker in ('author_approval','legal_release_gate','copyright_ownership','metadata_changes')))
        self.assertFalse((ROOT/'tests/test_licensing_candidate.py').exists())

    def test_readme_scientific_summary(self):
        text = (ROOT/'README.md').read_text()
        for value in ('92.5914%','95.2790%','91.5452%','95.3466%','98.3899%','98.7162%','575,359','275,711,872','2.301436 decimal MB','233 patients','3,064'):
            self.assertIn(value,text)
        self.assertIn('No AMGF in A4',text)
        self.assertIn('may be optimistic',text)
        self.assertIn('results/figures/A4_architecture.png',text)

if __name__ == '__main__':
    unittest.main()
