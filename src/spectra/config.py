"""Typed configuration for retained SPECTRA data, ABMIL, and training paths."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

QC_THRESHOLD = 0.46509784932900855


@dataclass(frozen=True)
class DataConfig:
    manifest_path: str | None = None
    validation_manifest_path: str | None = None
    source_roots: tuple[str, ...] = ("raw/source_a", "raw/source_b")
    slide_id_column: str = "slide_id"
    slide_path_column: str | None = "slide_path"
    group_id_column: str = "group_id"
    label_column: str = "label"
    entity_id_column: str | None = "patient_id"
    canonicalize_entity_ids: bool = False
    sampling_stratum_column: str | None = None
    sample_weight_column: str | None = None
    allow_reused_slides: bool = False
    max_slides_per_group: int | None = None
    max_rows_per_slide: int | None = 10_000
    top_rows_per_slide: int | None = None
    slide_cache_size: int = 256
    raw_embedding_cache_dir: str | None = None
    require_raw_embedding_cache: bool = False
    transformed_embedding_cache_dir: str | None = None
    qc_score_cache_dir: str | None = None
    qc_hard_filter_threshold: float | None = None
    immune_score_hard_filter_threshold: float | None = None
    immune_score_column: str | None = None
    marker_mean_hard_filter_threshold: float | None = None
    marker_mean_hard_filter_columns: tuple[str, ...] = (
        "TRITC_mean",
        "CY5_mean",
        "FITC_mean",
    )
    input_dim: int = 512
    embedding_indices: tuple[int, ...] | None = None
    embedding_normalization_path: str | None = None
    auxiliary_columns: tuple[str, ...] = ()
    feature_dtype: str = "float16"
    preserve_filtered_slide_exposure: bool = False
    cell_covariate_cache_dir: str | None = None


DATA_CONTRACT_FIELDS = (
    "source_roots",
    "slide_id_column",
    "slide_path_column",
    "group_id_column",
    "label_column",
    "entity_id_column",
    "canonicalize_entity_ids",
    "sampling_stratum_column",
    "sample_weight_column",
    "allow_reused_slides",
    "max_slides_per_group",
    "max_rows_per_slide",
    "top_rows_per_slide",
    "raw_embedding_cache_dir",
    "require_raw_embedding_cache",
    "transformed_embedding_cache_dir",
    "qc_score_cache_dir",
    "qc_hard_filter_threshold",
    "immune_score_hard_filter_threshold",
    "immune_score_column",
    "marker_mean_hard_filter_threshold",
    "marker_mean_hard_filter_columns",
    "input_dim",
    "embedding_indices",
    "embedding_normalization_path",
    "auxiliary_columns",
    "feature_dtype",
)


def _canonical_contract_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_canonical_contract_value(item) for item in value)
    if isinstance(value, dict):
        return tuple((key, _canonical_contract_value(item)) for key, item in sorted(value.items()))
    return value


def data_contract(data: DataConfig | dict[str, Any]) -> dict[str, Any]:
    """Return the feature, grouping, exposure, and QC contract for a dataset."""
    values = asdict(data) if isinstance(data, DataConfig) else data
    return {
        name: _canonical_contract_value(values.get(name, False if name == "canonicalize_entity_ids" else None))
        for name in DATA_CONTRACT_FIELDS
    }


def validate_checkpoint_data_config(
    checkpoint_payload: dict[str, Any],
    data: DataConfig,
    *,
    context: str,
) -> None:
    """Reject a checkpoint trained with different data processing."""
    metadata = checkpoint_payload.get("metadata")
    experiment = metadata.get("experiment_config") if isinstance(metadata, dict) else None
    saved_data = experiment.get("data") if isinstance(experiment, dict) else None
    if not isinstance(saved_data, dict):
        raise TypeError("SPECTRA checkpoint lacks its data-processing contract")
    saved_contract = data_contract(saved_data)
    current_contract = data_contract(data)
    mismatches = [
        name for name in DATA_CONTRACT_FIELDS if saved_contract[name] != current_contract[name]
    ]
    if mismatches:
        raise ValueError(f"SPECTRA {context} data contract mismatch: " + ", ".join(mismatches))


def _artifact_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve(strict=True).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "abmil_ungated"
    input_dim: int = 512
    model_dim: int = 128
    abmil_input_activation: str = "gelu"
    dropout: float = 0.1
    max_slides: int = 128
    additive_head_count: int = 1
    additive_cell_chunk_size: int = 4096
    burden_mode: str = "intercept"
    burden_readout_learning_rate: float = 0.01
    population_count: int = 8
    population_frequency_scale: float = 100.0
    population_interactions: bool = True
    cell_covariate_dim: int = 0
    cell_covariate_normalization_path: str | None = None
    abmil_attention_dim: int = 128
    abmil_attention_normalization: str = "softmax"
    abmil_attention_temperature: float = 1.0
    abmil_entmax_alpha_start: float = 1.0
    abmil_entmax_alpha_end: float = 2.0
    abmil_entmax_anneal_epochs: int = 40
    abmil_independent_threshold: float = 0.0
    abmil_independent_scale: float = 1.0
    abmil_stochastic_topk: int = 0
    abmil_stochastic_topk_drop_probability: float = 0.0
    abmil_additive_pooling: bool = False
    abmil_additive_normalization: str = "post_attention"
    clam_instance_k: int = 8
    clam_instance_loss_weight: float = 0.0
    clam_bag_loss_weight: float | None = None
    abmil_slide_center_input: bool = False
    abmil_slide_scale_strength: float = 0.0
    abmil_slide_scale_minimum_variance: float = 1e-4
    qc_threshold: float = QC_THRESHOLD
    qc_penalty_exponent: float = 2.0
    qc_max_suppression: float = 1.0
    batch_residualization_path: str | None = None
    batch_residualization_components: int = 0
    batch_residualization_strength: float = 1.0
    batch_residualization_standardize_input: bool = False


@dataclass(frozen=True)
class StageConfig:
    name: str
    epochs: int
    trainable: str = "all"
    objective: str = "joint"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: str = "outputs/train"
    seed: int = 17
    batch_size: int = 1
    num_workers: int = 4
    validation_num_workers: int | None = None
    validation_slide_cache_size: int | None = None
    precision: str = "bf16"
    gradient_accumulation_steps: int = 1
    gradient_strategy: str = "standard"
    attention_alpha_schedule_path: str | None = None
    max_grad_norm: float = 1.0
    pos_weight: float | None = None
    label_smoothing: float = 0.0
    entity_structured_batch_sampling: bool = False
    positive_entities_per_batch: int = 24
    negative_entities_per_batch: int = 8
    entity_balanced_sampling: bool = False
    entity_class_balanced_sampling: bool = False
    negative_entity_pair_sampling: bool = False
    negative_entity_dynamic_slide_pairing: bool = False
    negative_cross_entity_pair_probability: float = 0.0
    negative_entity_balance_exponent: float = 1.0
    positive_stratum_balance_exponent: float = 0.0
    entity_balance_exponent: float = 1.0
    learning_rate_schedule: str = "none"
    warmup_fraction: float = 0.0
    # Fixed horizon preserves the inner-CV learning-rate prefix during refit.
    scheduler_horizon_epochs: int | None = None
    early_stopping_patience: int | None = None
    early_stopping_min_epochs: int = 0
    early_stopping_patience_start_epoch: int | None = None
    log_every_n_steps: int = 25
    checkpoint_monitor: str = "validation_loss"
    checkpoint_selection_start_epoch: int = 1
    strict_reproducibility: bool = False
    batch_schedule_path: str | None = None
    strict_resume: bool = False
    stages: tuple[StageConfig, ...] = field(
        default_factory=lambda: (StageConfig(name="train", epochs=20),)
    )


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool = False
    project: str = "spectra"
    entity: str | None = None
    name: str | None = None
    group: str | None = None
    tags: tuple[str, ...] = ()
    mode: str = "online"


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tuple_or_none(value: Any) -> tuple[Any, ...] | None:
    return None if value is None else tuple(value)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve(strict=True)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise TypeError("experiment configuration must be a mapping")
    unknown = set(raw) - {"data", "model", "training", "wandb"}
    if unknown:
        raise ValueError(f"unknown top-level config keys: {sorted(unknown)}")

    data_raw = dict(raw.get("data", {}))
    if "expected_rows_per_slide" in data_raw:
        if "max_rows_per_slide" in data_raw:
            raise ValueError(
                "data.expected_rows_per_slide and data.max_rows_per_slide cannot both be set"
            )
        data_raw["max_rows_per_slide"] = data_raw.pop("expected_rows_per_slide")
    if "source_roots" in data_raw:
        data_raw["source_roots"] = tuple(data_raw["source_roots"])
    if "embedding_indices" in data_raw:
        data_raw["embedding_indices"] = _tuple_or_none(data_raw["embedding_indices"])
    if "auxiliary_columns" in data_raw:
        data_raw["auxiliary_columns"] = tuple(data_raw["auxiliary_columns"])
    if "marker_mean_hard_filter_columns" in data_raw:
        data_raw["marker_mean_hard_filter_columns"] = tuple(
            data_raw["marker_mean_hard_filter_columns"]
        )
    data = DataConfig(**data_raw)
    model_raw = dict(raw.get("model", {}))
    model = ModelConfig(**model_raw)

    training_raw = dict(raw.get("training", {}))
    if "stages" in training_raw:
        training_raw["stages"] = tuple(StageConfig(**stage) for stage in training_raw["stages"])
    training = TrainingConfig(**training_raw)
    wandb_raw = dict(raw.get("wandb", {}))
    if "tags" in wandb_raw:
        wandb_raw["tags"] = tuple(wandb_raw["tags"])
    wandb = WandbConfig(**wandb_raw)

    architectures = {
        "population_abundance",
        "additive_experts",
        "additive_burden",
        "abmil_vanilla",
        "abmil_ungated",
        "abmil_gated",
        "clam_gated",
        "clam_gated_multihead",
        "gated_frequency",
    }
    if model.architecture not in architectures:
        raise ValueError(f"model.architecture must be one of {sorted(architectures)}")
    if model.architecture == "population_abundance":
        if (model.population_count < 1 or model.additive_cell_chunk_size < 1
                or not math.isfinite(model.population_frequency_scale)
                or model.population_frequency_scale <= 0
                or not isinstance(model.population_interactions, bool)
                or model.abmil_slide_center_input or model.abmil_slide_scale_strength
                or model.qc_max_suppression != 1.0
                or model.batch_residualization_path is not None):
            raise ValueError("invalid population abundance configuration")
    if model.architecture in {"additive_experts", "additive_burden"}:
        if (model.abmil_attention_normalization not in {
                "sparsemax", "softmax", "entmax15", "entmax_annealed"}
                or not model.abmil_additive_pooling
                or model.abmil_additive_normalization != "pre_attention"
                or model.additive_head_count < 1 or model.additive_cell_chunk_size < 1
                or model.cell_covariate_dim not in (0, 6)):
            raise ValueError("invalid corrected additive expert configuration")
        if bool(model.cell_covariate_dim) != bool(data.cell_covariate_cache_dir):
            raise ValueError("expert covariate dimensions and cache must agree")
        if bool(model.cell_covariate_dim) != bool(model.cell_covariate_normalization_path):
            raise ValueError("expert measurements require a training normalization artifact")
    elif data.cell_covariate_cache_dir is not None or model.cell_covariate_dim:
        raise ValueError("cell covariates require additive_experts")
    if model.architecture == "additive_burden":
        from .additive_burden import BURDEN_MODES
        if (model.burden_mode not in BURDEN_MODES or model.cell_covariate_dim
                or model.batch_residualization_components or model.abmil_slide_scale_strength
                or not data.preserve_filtered_slide_exposure
                or not math.isfinite(model.burden_readout_learning_rate)
                or model.burden_readout_learning_rate <= 0):
            raise ValueError("invalid additive burden configuration/exposure")
    if model.architecture == "abmil_vanilla":
        vanilla_extras = {
            "abmil_input_activation": model.abmil_input_activation != "gelu",
            "dropout": model.dropout != 0.0,
            "abmil_attention_normalization": model.abmil_attention_normalization != "softmax",
            "abmil_stochastic_topk": model.abmil_stochastic_topk != 0,
            "abmil_stochastic_topk_drop_probability": (
                model.abmil_stochastic_topk_drop_probability != 0.0
            ),
            "abmil_additive_pooling": model.abmil_additive_pooling,
            "clam_instance_loss_weight": model.clam_instance_loss_weight != 0.0,
            "clam_bag_loss_weight": model.clam_bag_loss_weight is not None,
            "abmil_slide_center_input": model.abmil_slide_center_input,
            "abmil_slide_scale_strength": model.abmil_slide_scale_strength != 0.0,
            "qc_max_suppression": model.qc_max_suppression != 1.0,
            "batch_residualization_path": model.batch_residualization_path is not None,
            "batch_residualization_components": model.batch_residualization_components != 0,
            "batch_residualization_standardize_input": (
                model.batch_residualization_standardize_input
            ),
        }
        enabled = sorted(name for name, present in vanilla_extras.items() if present)
        if enabled:
            raise ValueError("vanilla ABMIL forbids extra mechanisms: " + ", ".join(enabled))
    normalizations = {
        "fixed_uniform",
        "fixed_rank",
        "ecod_monotone",
        "fixed_outlier",
        "softmax",
        "entmax15",
        "entmax_annealed",
        "sparsemax",
        "sigmoid_mass",
        "sparse_sigmoid_mass",
        "dual_sparse_sigmoid_mass",
        "shared_sparse_sigmoid_mass",
        "independent_relu",
    }
    if model.abmil_attention_normalization not in normalizations:
        raise ValueError(f"ABMIL normalization must be one of {sorted(normalizations)}")
    if training.attention_alpha_schedule_path is not None and (
        model.architecture != "additive_experts"
        or model.abmil_attention_normalization != "entmax_annealed"
    ):
        raise ValueError("an attention alpha schedule requires annealed additive_experts")
    if model.abmil_attention_normalization == "entmax_annealed":
        if (
            model.architecture != "additive_experts"
            or not math.isfinite(model.abmil_entmax_alpha_start)
            or not math.isfinite(model.abmil_entmax_alpha_end)
            or model.abmil_entmax_alpha_start != 1.0
            or model.abmil_entmax_alpha_end != 2.0
            or model.abmil_entmax_anneal_epochs < 2
        ):
            raise ValueError(
                "annealed entmax requires additive_experts, alpha 1 to 2, and at least two epochs"
            )
    if (not math.isfinite(model.abmil_attention_temperature)
            or model.abmil_attention_temperature <= 0):
        raise ValueError("ABMIL attention temperature must be finite and positive")
    if model.abmil_attention_temperature != 1.0 and (
        model.architecture not in {"abmil_gated", "abmil_ungated", "clam_gated"}
        or model.abmil_attention_normalization not in {"softmax", "sparsemax", "entmax15"}
    ):
        raise ValueError("attention temperature requires standard single-head ABMIL pooling")
    if model.abmil_attention_normalization == "independent_relu":
        if (model.architecture not in {"abmil_gated", "abmil_ungated"}
                or not model.abmil_additive_pooling
                or model.abmil_additive_normalization != "pre_attention"):
            raise ValueError("independent ReLU requires pre-attention Additive ABMIL")
        if (not math.isfinite(model.abmil_independent_threshold)
                or not math.isfinite(model.abmil_independent_scale)
                or model.abmil_independent_scale <= 0):
            raise ValueError("independent ReLU threshold and positive scale must be finite")
        if model.abmil_stochastic_topk != 0:
            raise ValueError("independent ReLU forbids competitive top-k masking")
        if not data.preserve_filtered_slide_exposure:
            raise ValueError("independent ReLU requires preserved acquired slide exposure")
    frequency_normalizations = {
        "sigmoid_mass",
        "sparse_sigmoid_mass",
        "dual_sparse_sigmoid_mass",
        "shared_sparse_sigmoid_mass",
    }
    if (model.architecture == "gated_frequency") != (
        model.abmil_attention_normalization in frequency_normalizations
    ):
        raise ValueError(
            "gated_frequency requires a sigmoid-mass normalization, and "
            "sigmoid-mass normalization requires gated_frequency"
        )
    if data.input_dim != model.input_dim or data.input_dim < 1:
        raise ValueError("data and model input dimensions must match and be positive")
    fixed_rank_normalizations = {"fixed_uniform", "fixed_rank", "ecod_monotone"}
    if data.auxiliary_columns:
        valid_outlier = (
            model.abmil_attention_normalization == "fixed_outlier"
            and data.auxiliary_columns == ("outlier_score",)
        )
        valid_rank = (
            model.abmil_attention_normalization in fixed_rank_normalizations
            and data.auxiliary_columns == ("rank_fraction",)
        )
        if not ((valid_outlier or valid_rank) and data.input_dim == 513):
            raise ValueError("consolidated ABMIL accepts biological Z features only")
    elif model.abmil_attention_normalization == "fixed_outlier":
        raise ValueError("fixed_outlier pooling requires the outlier_score auxiliary column")
    elif model.abmil_attention_normalization in fixed_rank_normalizations:
        raise ValueError("fixed ECOD-rank pooling requires the rank_fraction auxiliary column")
    if model.model_dim < 2 or model.abmil_attention_dim < 1:
        raise ValueError("model dimensions must be positive")
    if model.abmil_input_activation not in {"gelu", "identity"}:
        raise ValueError("model.abmil_input_activation must be 'gelu' or 'identity'")
    if model.clam_instance_k < 1:
        raise ValueError("model.clam_instance_k must be positive")
    if not 0.0 <= model.clam_instance_loss_weight < 1.0:
        raise ValueError("model.clam_instance_loss_weight must be in [0, 1)")
    if model.clam_bag_loss_weight is not None and model.clam_bag_loss_weight <= 0.0:
        raise ValueError("model.clam_bag_loss_weight must be positive when specified")
    clam_architectures = {"clam_gated", "clam_gated_multihead"}
    if model.architecture not in clam_architectures and model.clam_bag_loss_weight is not None:
        raise ValueError("an explicit CLAM bag loss weight requires a CLAM architecture")

    if model.architecture not in clam_architectures and model.clam_instance_loss_weight != 0.0:
        raise ValueError("CLAM instance loss requires a CLAM architecture")
    if model.architecture == "clam_gated_multihead":
        if model.abmil_attention_normalization != "sparsemax":
            raise ValueError("multihead CLAM currently requires sparsemax normalization")
        if not model.abmil_additive_pooling:
            raise ValueError("multihead CLAM currently requires Additive MIL pooling")
    if not 0.0 <= model.dropout < 1.0:
        raise ValueError("model.dropout must be in [0, 1)")
    if model.abmil_stochastic_topk < 0:
        raise ValueError("model.abmil_stochastic_topk must be non-negative")
    if not 0.0 <= model.abmil_stochastic_topk_drop_probability <= 1.0:
        raise ValueError("model.abmil_stochastic_topk_drop_probability must be in [0, 1]")
    if (model.abmil_stochastic_topk == 0) != (model.abmil_stochastic_topk_drop_probability == 0.0):
        raise ValueError("STKIM top-k and drop probability must be enabled together")
    if model.max_slides < 1:
        raise ValueError("model.max_slides must be positive")
    if data.max_slides_per_group is not None and data.max_slides_per_group > model.max_slides:
        raise ValueError("data.max_slides_per_group exceeds model.max_slides")
    if data.max_rows_per_slide is not None and data.max_rows_per_slide < 1:
        raise ValueError("data.max_rows_per_slide must be positive")
    if data.top_rows_per_slide is not None:
        raise ValueError("top_rows_per_slide is not retained; every cell must remain available")
    if data.slide_cache_size < 0:
        raise ValueError("data.slide_cache_size must be non-negative")
    if not isinstance(data.canonicalize_entity_ids, bool):
        raise TypeError("canonicalize_entity_ids must be boolean")
    if data.canonicalize_entity_ids and data.entity_id_column is None:
        raise ValueError("canonical identities require a donor/patient entity_id_column")
    if data.feature_dtype not in {"float16", "float32"}:
        raise ValueError("data.feature_dtype must be float16 or float32")
    if data.require_raw_embedding_cache and data.raw_embedding_cache_dir is None:
        raise ValueError("require_raw_embedding_cache requires raw_embedding_cache_dir")
    if model.abmil_slide_scale_strength and not model.abmil_slide_center_input:
        raise ValueError("ABMIL slide scaling requires slide centering")
    if not 0.0 <= model.abmil_slide_scale_strength <= 1.0:
        raise ValueError("ABMIL slide scale strength must be in [0, 1]")
    if model.abmil_additive_pooling and model.architecture == "gated_frequency":
        raise ValueError("additive pooling is not defined for gated_frequency auxiliary features")
    if model.abmil_additive_normalization not in {"post_attention", "pre_attention", "none"}:
        raise ValueError(
            "abmil_additive_normalization must be post_attention, pre_attention, or none"
        )
    if not model.abmil_additive_pooling and model.abmil_additive_normalization != "post_attention":
        raise ValueError("additive normalization modes require abmil_additive_pooling")
    if model.abmil_slide_scale_minimum_variance <= 0.0:
        raise ValueError("ABMIL slide scale minimum variance must be positive")
    if not math.isclose(model.qc_threshold, QC_THRESHOLD, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"qc_threshold is frozen at {QC_THRESHOLD}")
    if not math.isclose(model.qc_penalty_exponent, 2.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("qc_penalty_exponent is frozen at 2.0")
    if model.qc_max_suppression < 1.0 or not math.isfinite(model.qc_max_suppression):
        raise ValueError("qc_max_suppression must be finite and at least one")
    if model.qc_max_suppression > 1.0 and data.qc_score_cache_dir is None:
        raise ValueError("QC-aware ABMIL requires data.qc_score_cache_dir")
    if data.qc_hard_filter_threshold is not None:
        if not math.isclose(
            data.qc_hard_filter_threshold,
            QC_THRESHOLD,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"QC hard-filter threshold is frozen at {QC_THRESHOLD}")
        if data.qc_score_cache_dir is None:
            raise ValueError("QC hard filtering requires data.qc_score_cache_dir")
        if model.qc_max_suppression != 1.0:
            raise ValueError("hard filtering and soft QC suppression cannot be combined")
    if data.immune_score_hard_filter_threshold is not None:
        if not math.isfinite(data.immune_score_hard_filter_threshold) or not 0 < data.immune_score_hard_filter_threshold <= 1:
            raise ValueError("immune score threshold must be in (0, 1]")
        if not data.immune_score_column:
            raise ValueError("immune score filtering requires an explicit column")
        if data.top_rows_per_slide is not None:
            raise ValueError("immune score filtering is incompatible with top rows")
    if data.marker_mean_hard_filter_threshold is not None:
        if not math.isfinite(data.marker_mean_hard_filter_threshold):
            raise ValueError("marker mean hard-filter threshold must be finite")
        if data.marker_mean_hard_filter_threshold < 0.0:
            raise ValueError("marker mean hard-filter threshold must be non-negative")
        if not data.marker_mean_hard_filter_columns:
            raise ValueError("marker mean hard filtering requires at least one marker column")
        if len(set(data.marker_mean_hard_filter_columns)) != len(
            data.marker_mean_hard_filter_columns
        ):
            raise ValueError("marker mean hard-filter columns must be unique")
    if model.batch_residualization_components < 0:
        raise ValueError("batch residualization components must be non-negative")
    if model.batch_residualization_components and not model.batch_residualization_path:
        raise ValueError("batch residualization components require an artifact")
    if model.batch_residualization_standardize_input and not model.batch_residualization_path:
        raise ValueError("model-side standardization requires a residualization artifact")
    if not 0.0 <= model.batch_residualization_strength <= 1.0:
        raise ValueError("batch residualization strength must be in [0, 1]")

    if training.batch_size < 1 or training.num_workers < 0:
        raise ValueError("training batch size and worker count are invalid")
    if training.validation_num_workers is not None and training.validation_num_workers < 0:
        raise ValueError("validation_num_workers must be non-negative")
    if training.gradient_accumulation_steps < 1 or training.log_every_n_steps < 1:
        raise ValueError("training accumulation and logging intervals must be positive")
    if training.scheduler_horizon_epochs is not None:
        if training.scheduler_horizon_epochs < 1 or any(
            stage.epochs > training.scheduler_horizon_epochs for stage in training.stages
        ):
            raise ValueError("scheduler horizon must cover every training stage")
    if training.checkpoint_selection_start_epoch < 1:
        raise ValueError("checkpoint_selection_start_epoch must be positive")
    if training.max_grad_norm <= 0.0:
        raise ValueError("training.max_grad_norm must be positive")
    if training.pos_weight is not None and training.pos_weight <= 0.0:
        raise ValueError("training.pos_weight must be positive")
    if training.gradient_strategy not in {"standard", "pcgrad"}:
        raise ValueError("training.gradient_strategy must be standard or pcgrad")
    if training.gradient_strategy == "pcgrad" and (
        model.architecture != "additive_experts"
        or data.sampling_stratum_column != "cancer_types"
        or data.sample_weight_column is not None
        or training.pos_weight not in (None, 1.0)
        or training.label_smoothing != 0.0
        or any(stage.objective != "joint" for stage in training.stages)
    ):
        raise ValueError(
            "PCGrad requires additive_experts, cancer_types metadata, and unweighted hard-label BCE"
        )
    if not 0.0 <= training.label_smoothing < 1.0:
        raise ValueError("training.label_smoothing must be in [0, 1)")
    if training.precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError("training.precision must be fp32, bf16, or fp16")
    if training.strict_resume and not training.strict_reproducibility:
        raise ValueError("strict_resume requires strict_reproducibility")
    if training.strict_reproducibility:
        if training.precision != "fp32":
            raise ValueError("strict reproducibility requires FP32")
        if training.batch_schedule_path is None:
            raise ValueError("strict reproducibility requires a materialized batch schedule")
        if training.num_workers != 0 or training.validation_num_workers not in {None, 0}:
            raise ValueError("strict reproducibility requires single-process data loading")
    if training.learning_rate_schedule not in {"none", "warmup_cosine"}:
        raise ValueError("unsupported learning-rate schedule")
    if not 0.0 <= training.warmup_fraction < 1.0:
        raise ValueError("training.warmup_fraction must be in [0, 1)")
    sampling_modes = (
        training.entity_balanced_sampling,
        training.entity_class_balanced_sampling,
        training.negative_entity_pair_sampling,
        training.entity_structured_batch_sampling,
    )
    if sum(sampling_modes) > 1:
        raise ValueError("training entity sampling modes are mutually exclusive")
    if not 0.0 <= training.negative_cross_entity_pair_probability <= 1.0:
        raise ValueError("negative_cross_entity_pair_probability must be in [0, 1]")
    if (
        training.negative_cross_entity_pair_probability
        and not training.negative_entity_dynamic_slide_pairing
    ):
        raise ValueError("negative cross-entity pairing requires dynamic negative slide pairing")
    if training.entity_structured_batch_sampling and training.batch_size != (
        training.positive_entities_per_batch + training.negative_entities_per_batch
    ):
        raise ValueError("structured class counts must sum to training.batch_size")
    if training.checkpoint_monitor not in {
        "validation_loss",
        "validation_auroc",
        "validation_entity_weighted_auroc",
        "train_loss",
        "train_auroc",
    }:
        raise ValueError("unsupported checkpoint monitor")
    if not training.stages:
        raise ValueError("at least one training stage is required")
    for stage in training.stages:
        if stage.epochs < 1 or stage.learning_rate < 0.0 or stage.weight_decay < 0.0:
            raise ValueError("training stage epochs/rates are invalid")
        if stage.trainable not in {"all", "latents_and_head", "head"}:
            raise ValueError("unsupported training stage policy")
        if stage.objective != "joint":
            raise ValueError("unsupported training stage objective")
    return ExperimentConfig(data=data, model=model, training=training, wandb=wandb)


def validate_additive_experts_checkpoint_data_config(checkpoint_payload, data, *, context):
    """Keep expert measurement columns, QC, and provenance fixed at inference."""
    saved = checkpoint_payload.get("metadata", {}).get("experiment_config", {}).get("data")
    if not isinstance(saved, dict):
        raise ValueError("expert checkpoint lacks its data contract")
    current = asdict(data)
    fields = (*DATA_CONTRACT_FIELDS, "preserve_filtered_slide_exposure", "cell_covariate_cache_dir")
    differences = [key for key in fields if _canonical_contract_value(saved.get(key, False if key == "canonicalize_entity_ids" else None)) != _canonical_contract_value(current.get(key))]
    if differences:
        raise ValueError(f"expert {context} data contract mismatch: {', '.join(differences)}")
    metadata = checkpoint_payload["metadata"]
    model = metadata["experiment_config"].get("model", {})
    artifacts = {
        "embedding_normalization": data.embedding_normalization_path,
        "batch_residualization": model.get("batch_residualization_path"),
        "qc_score_cache": None if data.qc_score_cache_dir is None else str(Path(data.qc_score_cache_dir) / "metadata.json"),
        "cell_covariate_cache": None if data.cell_covariate_cache_dir is None else str(Path(data.cell_covariate_cache_dir) / "metadata.json"),
        "cell_covariate_normalization": model.get("cell_covariate_normalization_path"),
    }
    for name, path in artifacts.items():
        if path is not None:
            expected = metadata.get("provenance", {}).get(name, {}).get("sha256")
            if expected != _artifact_sha256(path):
                raise ValueError(f"expert {context} {name} SHA-256 mismatch")
