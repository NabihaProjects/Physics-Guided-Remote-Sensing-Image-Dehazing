"""Configuration management module with schema validation and typed dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class ProjectConfig:
    name: str = "rs-dehazing-benchmark"
    experiment_name: str = "default_run"
    output_dir: str = "experiments"
    seed: int = 42
    deterministic: bool = True


@dataclass
class LoggingConfig:
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "rs-dehazing-benchmark"


@dataclass
class HardwareConfig:
    device: str = "auto"
    num_workers: int = 0
    mixed_precision: bool = True


@dataclass
class HazeSimulationConfig:
    beta_range: List[float] = field(default_factory=lambda: [0.04, 0.12])
    atmospheric_light: List[float] = field(default_factory=lambda: [0.7, 1.0])
    non_uniform: bool = True


@dataclass
class DataConfig:
    dataset_name: str = "synthetic_rs"
    data_dir: str = "data"
    image_size: int = 256
    batch_size: int = 8
    val_split: float = 0.2
    haze_simulation: HazeSimulationConfig = field(default_factory=HazeSimulationConfig)


@dataclass
class ModelConfig:
    name: str = "rs_pgnet"
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 32


@dataclass
class TrainingConfig:
    epochs: int = 30
    learning_rate: float = 0.0002
    weight_decay: float = 0.0001
    scheduler: str = "cosine"
    min_lr: float = 1e-6
    save_interval: int = 5
    eval_interval: int = 1


@dataclass
class LossWeightsConfig:
    l1: float = 1.0
    ssim: float = 0.2
    spectral: float = 0.1
    tv_transmission: float = 0.05


@dataclass
class ExperimentConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss_weights: LossWeightsConfig = field(default_factory=LossWeightsConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def save_yaml(self, path: Path | str) -> None:
        """Save configuration to a YAML file for experiment archiving."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


def _dict_to_dataclass(raw: Dict[str, Any]) -> ExperimentConfig:
    """Helper to convert nested dictionary into typed ExperimentConfig."""
    proj = ProjectConfig(**raw.get("project", {}))
    log_cfg = LoggingConfig(**raw.get("logging", {}))
    hw = HardwareConfig(**raw.get("hardware", {}))

    data_raw = raw.get("data", {})
    haze_raw = data_raw.pop("haze_simulation", {})
    haze_cfg = HazeSimulationConfig(**haze_raw) if isinstance(haze_raw, dict) else HazeSimulationConfig()
    data = DataConfig(haze_simulation=haze_cfg, **data_raw)

    model = ModelConfig(**raw.get("model", {}))
    training = TrainingConfig(**raw.get("training", {}))
    loss_weights = LossWeightsConfig(**raw.get("loss_weights", {}))

    return ExperimentConfig(
        project=proj,
        logging=log_cfg,
        hardware=hw,
        data=data,
        model=model,
        training=training,
        loss_weights=loss_weights,
    )


def load_config(config_path: Path | str, overrides: Optional[Dict[str, Any]] = None) -> ExperimentConfig:
    """Load and validate configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file.
        overrides: Optional key-value overrides for configuration parameters.

    Returns:
        Validated ExperimentConfig object.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f) or {}

    # Handle optional simple overrides if provided
    if overrides:
        for key, val in overrides.items():
            keys = key.split(".")
            d = raw_dict
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = val

    return _dict_to_dataclass(raw_dict)
