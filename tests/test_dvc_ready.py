import subprocess
from pathlib import Path


def test_ensure_dvc_ready_adds_minio_remote(monkeypatch, tmp_path):
    """
    Проверяем, что при первом запуске менеджер сам "поднимает" DVC-репо и настраивает remote minio.

    Важно: в тесте не запускаем настоящий DVC и не ходим в Minio/S3.
    Вместо этого мокем subprocess.run и смотрим, какие команды менеджер попытался вызвать.
    """

    # уводим "DVC-репозиторий" в временную папку pytest
    monkeypatch.setenv("DVC_REPO_DIR", str(tmp_path))

    # эти переменные поденяются при запуске dvc remote add/modify
    monkeypatch.setenv("S3_BUCKET", "mlops-test")
    monkeypatch.setenv("S3_ENDPOINT_URL", "http://minio:9000")

    calls = []

    def fake_run(args, cwd=None, **kwargs):
        calls.append({"args": args, "cwd": cwd})

        # делаем вид, что minio ещё не настроен
        stdout = ""
        if args[:3] == ["dvc", "remote", "list"]:
            stdout = "origin s3://somewhere/other\n"

        class R:
            returncode = 0
            stderr = ""

            def __init__(self, out):
                self.stdout = out

        return R(stdout)

    # подмена настоящего subprocess на фейковый
    monkeypatch.setattr(subprocess, "run", fake_run)
    from app.dvc_datasets import DVCDatasetManager

    mgr = DVCDatasetManager()
    cmds = [" ".join(c["args"]) for c in calls]
    assert any(c.startswith("dvc init --no-scm") for c in cmds), cmds
    assert any(c.startswith("dvc remote list") for c in cmds), cmds
    assert any("dvc remote add -d minio s3://mlops-test/dvc" in c for c in cmds), cmds
    assert any(
        "dvc remote modify minio endpointurl http://minio:9000" in c for c in cmds
    ), cmds
    assert any("dvc remote modify minio use_ssl false" in c for c in cmds), cmds
    assert any(c.startswith("dvc remote default") for c in cmds), cmds
    assert all(Path(c["cwd"]) == mgr.repo_dir for c in calls), calls
