"""Public checks work without internal source or checksum records."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agu_james import runtime, verification


class PublicVerification(unittest.TestCase):
    def test_check_ignores_private_audit_files(self):
        scratch = Path(__file__).resolve().parents[1] / 'artifacts/verification'
        scratch.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(dir=scratch) as directory:
                root = Path(directory)
                package = root / 'agu_james'
                package.mkdir()
                (package / '__init__.py').write_text('', encoding='utf-8')
                (package / 'example.py').write_text('value = 1\n', encoding='utf-8')
                private = root / 'provenance'
                private.mkdir()
                for name in ['sources.json', 'files.json', 'reference_sources.json']:
                    (private / name).write_text('obsolete internal record', encoding='utf-8')
                runtime.configure({'artifact_root': str(root / 'outputs')})
                with patch.object(verification, '__file__', str(package / 'verification.py')):
                    result = verification.check(import_modules=False)
                self.assertEqual(result['syntax_checked_modules'], 2)
                self.assertEqual(result['issues'], [])
                self.assertEqual(result['unavailable_dependencies'], [])
                saved = json.loads((root / 'outputs/verification/check.json').read_text(encoding='utf-8'))
                self.assertEqual(saved, result)
        finally:
            runtime.configure({})


if __name__ == '__main__':
    unittest.main()
