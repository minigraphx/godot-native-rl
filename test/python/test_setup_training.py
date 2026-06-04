import os
import subprocess
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT = os.path.join(REPO_ROOT, "scripts", "setup_training.sh")


class TestSetupTraining(unittest.TestCase):
    def test_script_exists_and_is_executable(self):
        self.assertTrue(os.path.isfile(SCRIPT), "scripts/setup_training.sh missing")
        self.assertTrue(os.access(SCRIPT, os.X_OK), "setup_training.sh not executable")

    def test_check_mode_runs_and_names_next_step(self):
        # --check must not create venvs or pip-install; it validates and prints guidance.
        result = subprocess.run(
            [SCRIPT, "--check"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        out = result.stdout + result.stderr
        self.assertIn("requirements-train.txt", out)
        self.assertIn("requirements-convert.txt", out)
        self.assertIn("train_chase.sh", out)  # points at the next command
        # --check is non-destructive: it must not have created the venvs.
        self.assertFalse(os.path.isdir(os.path.join(REPO_ROOT, ".venv-train")))
        self.assertFalse(os.path.isdir(os.path.join(REPO_ROOT, ".venv")))


if __name__ == "__main__":
    unittest.main()
