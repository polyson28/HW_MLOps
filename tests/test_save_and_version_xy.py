from pathlib import Path
import pandas as pd


def test_save_and_version_xy_creates_files_and_returns_ref(monkeypatch, tmp_path):
    """
    Проверям сохранение и версионированеи без настоящего DVC и без настоящего Minio
    - создаётся папка датасета и файл train.parquet
    - создаётся .dvc файл 
    - md5 достаётся из этого .dvc файла через _extract_md5
    - возвращается DatasetRef с нормальными полями
    """

    # подменчем переменные и пути 
    monkeypatch.setenv("DVC_REPO_DIR", str(tmp_path))
    monkeypatch.setenv("S3_BUCKET", "mlops-test")
    monkeypatch.setenv("S3_ENDPOINT_URL", "http://minio:9000")

    from app.dvc_datasets import DVCDatasetManager

    monkeypatch.setattr(DVCDatasetManager, "_ensure_dvc_ready", lambda self: None)

    # создаем какой-то файл
    def fake_to_parquet(self, path, index=False):
        Path(path).write_bytes(b"PARQUET_STUB")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet, raising=True)

    calls = []

    # мокаем внутренний  _run, чтобы не запускать DVC команды реально
    def fake_run(self, args: list[str]) -> str:
        calls.append(args)

        if args[:2] == ["dvc", "add"]:
            rel = args[2]
            # сздаём .dvc файл там, где DVCDatasetManager его ожидает после dvc add
            dvc_path = self.repo_dir / (rel + ".dvc")
            dvc_path.parent.mkdir(parents=True, exist_ok=True)
            dvc_path.write_text(
                "md5: 11112222333344445555666677778888\n" f"path: {rel}\n",
                encoding="utf-8",
            )

            return ""

        if args[:2] == ["dvc", "push"]:
            return ""

        return ""

    monkeypatch.setattr(DVCDatasetManager, "_run", fake_run, raising=True)

    mgr = DVCDatasetManager()

    X = [[1, 2], [3, 4]]
    y = [0, 1]
    ref = mgr.save_and_version_xy(X, y)

    assert ref.dataset_id
    assert ref.dvc_md5 == "11112222333344445555666677778888"
    assert ref.data_path.endswith("train.parquet")
    assert ref.dvc_file.endswith("train.parquet.dvc")
    assert Path(ref.data_path).exists()
    assert Path(ref.dvc_file).exists()
    flat = [" ".join(c) for c in calls]
    assert any(s.startswith("dvc add ") for s in flat), flat
    assert any(s.startswith("dvc push") for s in flat), flat
