"""Algorithm configuration models loaded from environment variables."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _detect_project_root(start: Path | None = None) -> Path:
    """Walk upward from ``start`` to find the workspace root.

    Two layouts are recognised, because ``alg`` ships in two shapes: the
    research monorepo (marker: ``backend/pyproject.toml``, or ``compose.yaml``
    beside ``backend/``) and the public replication package, where ``alg/``,
    ``scripts/`` and ``data/`` sit directly under the repository root. The
    second marker matters: without it the detector silently fell back to the
    current working directory, so every path in the replication package was
    correct only when commands were run from its root.
    """
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        backend_pyproject = candidate / "backend" / "pyproject.toml"
        if backend_pyproject.exists():
            return candidate
        if (candidate / "compose.yaml").exists() and (candidate / "backend").is_dir():
            return candidate
        if ((candidate / "pyproject.toml").exists()
                and (candidate / "alg").is_dir()
                and (candidate / "scripts").is_dir()
                and (candidate / "data").is_dir()):
            return candidate
    return here


class AlgConfig(BaseSettings):
    """Algorithm configuration loaded from environment variables."""

    # Data Path
    input_data: str = Field(default="data/input", description="Input data directory path")
    output_data: str = Field(default="data/output", description="Output data directory path")
    model_cache: str = Field(default="data/cache", description="Model cache directory path")

    # Embedding
    embedding_dimensions: int = Field(default=768, description="Embedding dimensions")
    embedding_batch_size: int = Field(default=32, description="Batch size for embedding")

    # Processing
    chunk_size: int = Field(default=1000, description="Text chunk size")
    chunk_overlap: int = Field(default=200, description="Text chunk overlap")
    max_concurrency: int = Field(default=10, description="Maximum concurrent tasks")


WandbMode = Literal["online", "offline", "disabled"]


class EpfConfig(BaseSettings):
    """Configuration for the EPF (Electricity Price Forecasting) research harness.

    All fields can be overridden via environment variables prefixed with ``EPF_``,
    e.g. ``EPF_DEFAULT_AREA=tokyo``. Path fields resolve relative to ``project_root``
    when not absolute.
    """

    model_config = SettingsConfigDict(env_prefix="EPF_", extra="ignore", frozen=False)

    project_root: Path = Field(default_factory=_detect_project_root)
    raw_data_dir: Path = Field(default=Path("data/raw"))
    processed_data_dir: Path = Field(default=Path("data/processed"))
    reference_data_dir: Path = Field(default=Path("data/reference"))
    run_dir: Path = Field(default=Path("data/runs"))
    mlflow_tracking_uri: str = Field(default="file:./data/mlruns")

    wandb_project: str | None = Field(default=None)
    wandb_entity: str | None = Field(default=None)
    wandb_mode: WandbMode = Field(default="disabled")

    default_area: str = Field(default="kyushu")
    tz: str = Field(default="Asia/Tokyo")

    def resolved(self, attr: str) -> Path:
        """Return path attribute resolved against ``project_root`` if relative."""
        raw = getattr(self, attr)
        if not isinstance(raw, Path):
            msg = f"{attr} is not a Path field"
            raise TypeError(msg)
        return raw if raw.is_absolute() else (self.project_root / raw).resolve()

    def resolved_mlflow_uri(self) -> str:
        """Return ``mlflow_tracking_uri`` with ``file:./...`` resolved to absolute.

        Anything not starting with ``file:./`` is returned unchanged.
        """
        prefix = "file:./"
        if self.mlflow_tracking_uri.startswith(prefix):
            rel = self.mlflow_tracking_uri[len(prefix) :]
            absolute = (self.project_root / rel).resolve()
            return f"file:{absolute}"
        return self.mlflow_tracking_uri
