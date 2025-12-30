import os
import subprocess
import sys
import unittest


class TestMiaImportSideEffects(unittest.TestCase):
    def test_import_core_mia_has_no_seed_or_dataset_logs(self):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        env = os.environ.copy()
        env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [sys.executable, "-c", "import core.mia"],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        combined = (proc.stdout or "") + (proc.stderr or "")
        self.assertNotIn("Random seeds set", combined)
        self.assertNotIn("Using default dataset path", combined)


if __name__ == "__main__":
    unittest.main()

