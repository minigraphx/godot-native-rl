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
        venv_train = os.path.join(REPO_ROOT, ".venv-train")
        venv_convert = os.path.join(REPO_ROOT, ".venv")
        # Snapshot beforehand so the assertion holds whether or not the dev already has venvs
        # (the test must verify --check changes nothing, not that the dirs are absent).
        before = (os.path.isdir(venv_train), os.path.isdir(venv_convert))
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
        # --check is non-destructive: it must neither create nor remove the venvs.
        after = (os.path.isdir(venv_train), os.path.isdir(venv_convert))
        self.assertEqual(before, after, "--check must not create or remove venvs")

    def test_check_mode_names_sf_requirements(self):
        # The SF backend lives in a third venv (.venv-sf); --check must name its requirements file.
        result = subprocess.run(
            [SCRIPT, "--check"], cwd=REPO_ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        out = result.stdout + result.stderr
        self.assertIn("requirements-sf.txt", out)


if __name__ == "__main__":
    unittest.main()
