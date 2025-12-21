from __future__ import annotations

import logging
import os
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

logger = logging.getLogger("ml_service")


@dataclass(frozen=True)
class DatasetRef:
    dataset_id: str
    data_path: str
    dvc_file: str
    dvc_md5: str
    created_at: str


class DVCDatasetManager:
    def __init__(self) -> None:
        self.repo_dir = Path(os.getenv("DVC_REPO_DIR", "/app/storage/dvc_repo"))
        self.repo_dir.mkdir(parents=True, exist_ok=True)

        self.datasets_dir = self.repo_dir / "datasets"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

        self.bucket = os.getenv("S3_BUCKET", "mlops")
        self.endpoint = os.getenv("S3_ENDPOINT_URL", "http://minio:9000")

        logger.info(
            "DVC: init manager repo_dir=%s datasets_dir=%s",
            self.repo_dir,
            self.datasets_dir,
        )
        logger.info(
            "DVC: remote target s3://%s/dvc endpoint=%s", self.bucket, self.endpoint
        )

        self._ensure_dvc_ready()

    def _run(self, args: list[str]) -> str:
        """
        Запускает команду DVC с подробным логированием.
        Возвращает stdout (удобно для dvc remote list).
        """
        cmd = " ".join(args)
        logger.info("DVC: run: %s (cwd=%s)", cmd, self.repo_dir)

        p = subprocess.run(
            args,
            cwd=self.repo_dir,
            text=True,
            capture_output=True,
        )

        if p.stdout:
            logger.info("DVC: stdout:\n%s", p.stdout.strip())
        if p.stderr:
            logger.info("DVC: stderr:\n%s", p.stderr.strip())

        if p.returncode != 0:
            logger.error("DVC: command failed rc=%d: %s", p.returncode, cmd)
            raise subprocess.CalledProcessError(
                p.returncode, args, output=p.stdout, stderr=p.stderr
            )

        return p.stdout or ""

    def _ensure_dvc_ready(self) -> None:
        if not (self.repo_dir / ".dvc").exists():
            logger.info("DVC: repo not initialized -> dvc init --no-scm")
            self._run(["dvc", "init", "--no-scm"])
        else:
            logger.info("DVC: repo already initialized")

        out = self._run(["dvc", "remote", "list"])
        if "minio" not in out:
            logger.info("DVC: remote 'minio' not found -> adding and setting default")
            self._run(
                ["dvc", "remote", "add", "-d", "minio", f"s3://{self.bucket}/dvc"]
            )
            self._run(
                ["dvc", "remote", "modify", "minio", "endpointurl", self.endpoint]
            )
            self._run(["dvc", "remote", "modify", "minio", "use_ssl", "false"])
        else:
            logger.info("DVC: remote 'minio' exists")

        self._run(["dvc", "remote", "default"])

    def save_and_version_xy(
        self, X: Sequence[Sequence[Any]], y: Sequence[Any]
    ) -> DatasetRef:
        dataset_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
        out_dir = self.datasets_dir / dataset_id
        out_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(list(X))
        df["__target__"] = list(y)

        data_path = out_dir / "train.parquet"
        logger.info(
            "DVC: saving dataset file -> %s (rows=%d, cols=%d)",
            data_path,
            len(df),
            df.shape[1],
        )
        df.to_parquet(data_path, index=False)

        rel = data_path.relative_to(self.repo_dir)
        logger.info("DVC: dvc add %s", rel)
        self._run(["dvc", "add", str(rel)])

        dvc_path = self.repo_dir / (str(rel) + ".dvc")
        logger.info(
            "DVC: generated dvc file -> %s (exists=%s)", dvc_path, dvc_path.exists()
        )

        logger.info("DVC: dvc push (to minio)")
        self._run(["dvc", "push", "-v"])

        md5 = self._extract_md5(dvc_path)
        logger.info("DVC: dataset version md5=%s dataset_id=%s", md5, dataset_id)

        return DatasetRef(
            dataset_id=dataset_id,
            data_path=str(data_path),
            dvc_file=str(dvc_path),
            dvc_md5=md5,
            created_at=datetime.utcnow().isoformat() + "Z",
        )

    @staticmethod
    def _extract_md5(dvc_path: Path) -> str:
        if not dvc_path.exists():
            return ""
        text = dvc_path.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("md5:"):
                return line.split(":", 1)[1].strip()
        return ""
