"""Full-coverage quantisation-aware training for low-bit conversion.

The subpackage holds the pieces required to run BitNet-style QAT on a whole
pretrained model:

* :mod:`wal_tat.qat.quant` -- ternary/binary group quantizers and the
  :class:`~wal_tat.qat.quant.QuantLinear` layer that keeps a latent fp32
  weight and executes the hard quantized weight with a straight-through
  gradient.
* :mod:`wal_tat.qat.convert` -- in-place model surgery, coverage accounting,
  checkpointing, and export to the deployable packing format.
* :mod:`wal_tat.qat.evaluate` -- the shared evaluation harness.  Everything a
  QAT experiment reports about quality must go through it so that numbers from
  different runs remain directly comparable.

Typical use::

    model = WhisperForConditionalGeneration.from_pretrained(...).to(device)
    moments = collect_input_second_moments(model, run_calibration)
    manifest = convert_model_to_qat(model, precision="t3", input_second_moments=moments)
    print(manifest.summary())
"""

from .convert import (
    DEFAULT_TARGETS,
    QATEntry,
    QATManifest,
    collect_input_second_moments,
    convert_model_to_qat,
    export_packed_artifact,
    iter_target_linears,
    load_qat_checkpoint,
    qat_modules,
    save_qat_checkpoint,
)
from .data import (
    CACHE_FORMAT_VERSION,
    LIBRISPEECH_TRAIN_SPLITS,
    CacheSummary,
    FeatureSample,
    ShardRecord,
    Utterance,
    WhisperCollator,
    WhisperFeatureShards,
    build_dataloader,
    build_feature_cache,
    librispeech_parquet_shards,
    load_manifest,
    summarize_manifest,
)
from .distill import (
    AUTOCAST_DTYPES,
    CodeFlipTracker,
    DistillConfig,
    LossTerms,
    QATDistiller,
    StepRecord,
    TrainingReport,
    cosine_lr_multiplier,
    distillation_loss,
    parameter_groups,
    shift_labels_right,
    transformer_block_modules,
)
from .evaluate import (
    LIBRISPEECH_SPLITS,
    SpeechDatasetSplit,
    UtteranceResult,
    build_english_normalizer,
    evaluate_wer,
    load_librispeech,
    paired_bootstrap,
    resolve_dataset_split,
)
from .quant import (
    PRECISIONS,
    SCALE_EPS,
    BinaryQuantizer,
    GroupQuantizer,
    QuantLinear,
    TernaryQuantizer,
    get_quantizer,
    grouped_view,
)

__all__ = [
    "AUTOCAST_DTYPES",
    "CACHE_FORMAT_VERSION",
    "BinaryQuantizer",
    "CacheSummary",
    "CodeFlipTracker",
    "DEFAULT_TARGETS",
    "DistillConfig",
    "FeatureSample",
    "GroupQuantizer",
    "LIBRISPEECH_SPLITS",
    "LIBRISPEECH_TRAIN_SPLITS",
    "LossTerms",
    "PRECISIONS",
    "QATDistiller",
    "QATEntry",
    "QATManifest",
    "QuantLinear",
    "SCALE_EPS",
    "ShardRecord",
    "SpeechDatasetSplit",
    "StepRecord",
    "TernaryQuantizer",
    "TrainingReport",
    "Utterance",
    "UtteranceResult",
    "WhisperCollator",
    "WhisperFeatureShards",
    "build_dataloader",
    "build_english_normalizer",
    "build_feature_cache",
    "collect_input_second_moments",
    "convert_model_to_qat",
    "cosine_lr_multiplier",
    "distillation_loss",
    "evaluate_wer",
    "export_packed_artifact",
    "get_quantizer",
    "grouped_view",
    "iter_target_linears",
    "librispeech_parquet_shards",
    "load_librispeech",
    "load_manifest",
    "load_qat_checkpoint",
    "paired_bootstrap",
    "parameter_groups",
    "qat_modules",
    "resolve_dataset_split",
    "save_qat_checkpoint",
    "shift_labels_right",
    "summarize_manifest",
    "transformer_block_modules",
]
