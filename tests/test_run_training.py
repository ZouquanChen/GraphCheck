import sys
import tempfile
import unittest
from pathlib import Path


class RunTrainingTests(unittest.TestCase):
    def test_build_train_command_uses_local_qwen_defaults_when_unspecified(self):
        from run_training import build_train_command

        root = Path(r"E:/demo/GraphCheck")
        command = build_train_command(
            root,
            "MiniCheck_Train",
            ["--project", "GraphCheck_Debug"],
        )

        self.assertEqual(
            command,
            [
                sys.executable,
                str(root / "train.py"),
                "--train_dataset",
                "MiniCheck_Train",
                "--llm_model_name",
                "qwen_7b",
                "--llm_model_path",
                r"E:\HFModels\Qwen2.5-7B-Instruct",
                "--project",
                "GraphCheck_Debug",
            ],
        )

    def test_build_train_command_forwards_extra_args(self):
        from run_training import build_train_command

        root = Path(r"E:/demo/GraphCheck")
        command = build_train_command(
            root,
            "MiniCheck_Train",
            ["--project", "GraphCheck_Debug", "--llm_model_name", "llama_8b"],
        )

        self.assertEqual(
            command,
            [
                sys.executable,
                str(root / "train.py"),
                "--train_dataset",
                "MiniCheck_Train",
                "--project",
                "GraphCheck_Debug",
                "--llm_model_name",
                "llama_8b",
            ],
        )

    def test_ensure_extracted_dataset_copies_raw_pickle_when_missing(self):
        from run_training import ensure_extracted_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "dataset" / "MiniCheck_Train"
            raw_dir.mkdir(parents=True)
            raw_pkl = raw_dir / "MiniCheck_Train.pkl"
            raw_pkl.write_bytes(b"raw-pickle-bytes")

            extracted_pkl = ensure_extracted_dataset(root, "MiniCheck_Train")

            self.assertEqual(extracted_pkl.read_bytes(), b"raw-pickle-bytes")
            self.assertEqual(
                extracted_pkl,
                root
                / "dataset"
                / "extracted_KG"
                / "MiniCheck_Train"
                / "MiniCheck_Train.pkl",
            )

    def test_preprocessing_required_when_graphs_or_split_missing(self):
        from run_training import preprocessing_required

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_dir = root / "dataset" / "extracted_KG" / "MiniCheck_Train"
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "MiniCheck_Train.pkl").write_bytes(b"data")

            self.assertTrue(preprocessing_required(root, "MiniCheck_Train"))

    def test_preprocessing_not_required_when_expected_artifacts_exist(self):
        from run_training import preprocessing_required

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_dir = root / "dataset" / "extracted_KG" / "MiniCheck_Train"
            (dataset_dir / "graphs" / "claim").mkdir(parents=True)
            (dataset_dir / "graphs" / "doc").mkdir(parents=True)
            (dataset_dir / "split").mkdir(parents=True)
            (dataset_dir / "MiniCheck_Train.pkl").write_bytes(b"data")
            (dataset_dir / "graphs" / "claim" / "0.pt").write_bytes(b"claim")
            (dataset_dir / "graphs" / "doc" / "0.pt").write_bytes(b"doc")
            for name in ("train_indices.txt", "val_indices.txt", "test_indices.txt"):
                (dataset_dir / "split" / name).write_text("0\n", encoding="utf-8")

            self.assertFalse(preprocessing_required(root, "MiniCheck_Train"))


if __name__ == "__main__":
    unittest.main()
