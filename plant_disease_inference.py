"""Reusable preprocessing, model-building and inference utilities.

This module is the single source of truth for:

* the image preprocessing pipeline used by the notebook and the future API;
* the dynamically sized CNN used by the Optuna search and the final model;
* loading the two saved artifacts (pipeline + model) independently.

The preprocessing pipeline contains **no model code**, and the model contains
**no preprocessing code**.  A typical inference flow is::

    pipeline = load_preprocessing_pipeline("artifacts/preprocessing_pipeline.joblib")
    model, checkpoint = load_trained_model("artifacts/plant_disease_model.pt")
    processed = pipeline.transform("leaf.jpg")          # torch.float32 (3, 256, 256)
    result = predict(model, processed, checkpoint["class_names"])

Only inference-time dependencies are imported here: ``torch``,
``torchvision``, ``numpy``, ``pillow`` and ``joblib``.  Optuna and scikit-learn
are training-time dependencies and are intentionally not required to load the
artifacts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Configuration shared by the notebook, the artifacts and the backend.
# ---------------------------------------------------------------------------

IMAGE_SIZE: tuple[int, int] = (256, 256)
MEAN: tuple[float, float, float] = (0.5, 0.5, 0.5)
STD: tuple[float, float, float] = (0.5, 0.5, 0.5)
ROTATION_DEGREES: float = 10.0
HORIZONTAL_FLIP_PROB: float = 0.5

# Channel widths used by the existing 5-block CNN.  Deeper Optuna models keep
# the 512-channel width instead of doubling it further (memory safety).
CONV_CHANNEL_SCHEDULE: tuple[int, ...] = (32, 64, 128, 256, 512, 512, 512)
MIN_CONV_LAYERS: int = 3
MAX_CONV_LAYERS: int = 7

MODEL_FILE_NAME: str = "plant_disease_model.pt"
PIPELINE_FILE_NAME: str = "preprocessing_pipeline.joblib"
CONFIG_FILE_NAME: str = "inference_config.json"

__all__ = [
    "IMAGE_SIZE",
    "MEAN",
    "STD",
    "CONV_CHANNEL_SCHEDULE",
    "MIN_CONV_LAYERS",
    "MAX_CONV_LAYERS",
    "MODEL_FILE_NAME",
    "PIPELINE_FILE_NAME",
    "CONFIG_FILE_NAME",
    "ImagePreprocessingPipeline",
    "DynamicCNN",
    "build_cnn",
    "conv_channels_for_depth",
    "model_config_from_hyperparameters",
    "save_model_checkpoint",
    "load_preprocessing_pipeline",
    "load_trained_model",
    "predict",
    "predict_batch",
]


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------


def _validate_pair(value: Any, name: str) -> tuple[int, int]:
    try:
        height, width = value
    except Exception as exc:  # noqa: BLE001 - re-raised with a clear message
        raise TypeError(f"{name} must be a (height, width) pair.") from exc
    height, width = int(height), int(width)
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} values must be positive, got {(height, width)}.")
    return height, width


def _validate_triplet(value: Any, name: str) -> tuple[float, float, float]:
    try:
        values = tuple(float(v) for v in value)
    except Exception as exc:  # noqa: BLE001
        raise TypeError(f"{name} must contain three numeric values.") from exc
    if len(values) != 3:
        raise ValueError(f"{name} must contain three values (one per RGB channel).")
    return values


class ImagePreprocessingPipeline:
    """Deterministic image -> model-input pipeline.

    The pipeline reproduces the notebook preprocessing exactly:

    ``Resize((256, 256))`` -> ``ToTensor()`` -> ``Normalize(0.5, 0.5)``

    Training images additionally receive ``RandomHorizontalFlip`` and
    ``RandomRotation(10)``.  Those augmentations are only exposed through
    :meth:`as_torch_transform` with ``augment=True`` and are never used for
    validation, testing or inference.

    There are **no data-dependent fitted parameters** (no dataset mean/std, no
    PCA, no learned scaler), so there is no risk of validation leakage.  The
    pipeline's complete state is its configuration, which is what gets saved by
    :mod:`joblib`.

    Parameters
    ----------
    image_size:
        Output spatial size ``(height, width)``.  Defaults to ``(256, 256)``.
    mean, std:
        Per-channel normalization constants.  Defaults to ``0.5`` for all
        channels, matching the notebook, which maps ``[0, 1]`` pixels to
        ``[-1, 1]``.
    rotation_degrees:
        Maximum rotation used **only** during training augmentation.
    horizontal_flip_prob:
        Probability of the horizontal flip used **only** during training.
    """

    def __init__(
        self,
        image_size: tuple[int, int] = IMAGE_SIZE,
        mean: tuple[float, float, float] = MEAN,
        std: tuple[float, float, float] = STD,
        rotation_degrees: float = ROTATION_DEGREES,
        horizontal_flip_prob: float = HORIZONTAL_FLIP_PROB,
    ) -> None:
        self.image_size = _validate_pair(image_size, "image_size")
        self.mean = _validate_triplet(mean, "mean")
        self.std = _validate_triplet(std, "std")
        if any(s <= 0 for s in self.std):
            raise ValueError("std values must be strictly positive.")

        rotation_degrees = float(rotation_degrees)
        if rotation_degrees < 0:
            raise ValueError("rotation_degrees must be non-negative.")
        horizontal_flip_prob = float(horizontal_flip_prob)
        if not 0.0 <= horizontal_flip_prob <= 1.0:
            raise ValueError("horizontal_flip_prob must be between 0 and 1.")

        self.rotation_degrees = rotation_degrees
        self.horizontal_flip_prob = horizontal_flip_prob

        # IMPORTANT: keep these in the same order as the original notebook.
        self.eval_transform = T.Compose(
            [
                T.Resize(self.image_size),
                T.ToTensor(),
                T.Normalize(mean=self.mean, std=self.std),
            ]
        )
        self.train_transform = T.Compose(
            [
                T.Resize(self.image_size),
                T.RandomHorizontalFlip(p=self.horizontal_flip_prob),
                T.RandomRotation(self.rotation_degrees),
                T.ToTensor(),
                T.Normalize(mean=self.mean, std=self.std),
            ]
        )

    # -- serialization -----------------------------------------------------
    def __getstate__(self) -> dict[str, Any]:
        """Save configuration only, not live torchvision objects.

        This keeps the joblib file small and robust across torchvision
        versions; the transforms are rebuilt in :meth:`__setstate__`.
        """
        return {
            "image_size": self.image_size,
            "mean": self.mean,
            "std": self.std,
            "rotation_degrees": self.rotation_degrees,
            "horizontal_flip_prob": self.horizontal_flip_prob,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(**state)

    def get_config(self) -> dict[str, Any]:
        """Return a JSON-serializable description of the preprocessing."""
        return {
            "type": "ImagePreprocessingPipeline",
            "image_size": list(self.image_size),
            "channels": 3,
            "color_mode": "RGB",
            "output_layout": "CHW",
            "output_dtype": "float32",
            "normalization": "pixel / 255, then (x - mean) / std",
            "mean": list(self.mean),
            "std": list(self.std),
            "value_range": [-1.0, 1.0],
            "resize": "stretch to image_size (no center crop)",
            "augmentation": {
                "random_horizontal_flip_prob": self.horizontal_flip_prob,
                "random_rotation_degrees": self.rotation_degrees,
                "applied_only_during_training": True,
            },
        }

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Serialize the pipeline with joblib and return the written path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "ImagePreprocessingPipeline":
        """Load a pipeline saved with :meth:`save`."""
        pipeline = load_preprocessing_pipeline(path)
        if not isinstance(pipeline, cls):
            raise TypeError(
                f"{path!r} does not contain an {cls.__name__} "
                f"(got {type(pipeline)!r})."
            )
        return pipeline

    # -- public transforms -------------------------------------------------
    def as_torch_transform(self, augment: bool = False) -> T.Compose:
        """Return a torchvision transform for ``ImageFolder``/``DataLoader``.

        ``augment=True`` must only be used for the training split.
        """
        return self.train_transform if augment else self.eval_transform

    def transform(self, image: Any, augment: bool = False) -> torch.Tensor:
        """Convert one raw image into a normalized ``(3, H, W)`` tensor.

        Accepted inputs
        ---------------
        * path to an image (``str`` or ``os.PathLike``);
        * ``PIL.Image.Image``;
        * ``numpy.ndarray`` in HWC or CHW order, grayscale/RGB/RGBA,
          ``uint8`` or float in ``[0, 1]``/``[0, 255]``;
        * ``torch.Tensor`` in CHW order, float in ``[0, 1]``/``[0, 255]``.

        Already-normalized tensors (containing negative values) are rejected
        with an error instead of being normalized a second time.

        Raises
        ------
        FileNotFoundError, ValueError, TypeError
            With a message explaining exactly what was wrong with the input.
        """
        pil_image = self._to_pil(image)
        transform = self.train_transform if augment else self.eval_transform
        tensor = transform(pil_image)
        expected = (3, self.image_size[0], self.image_size[1])
        if tuple(tensor.shape) != expected:
            raise RuntimeError(
                f"Preprocessing produced shape {tuple(tensor.shape)}, "
                f"expected {expected}."
            )
        return tensor.contiguous()

    def transform_batch(
        self, images: Any, augment: bool = False
    ) -> torch.Tensor:
        """Convert several raw images into a normalized ``(N, 3, H, W)`` tensor.

        Accepts a list/tuple of single-image inputs, an ``(N, H, W, C)`` or
        ``(N, C, H, W)`` numpy array, or an ``(N, C, H, W)``/``(N, H, W, C)``
        tensor.  Every item is processed with the same pipeline as
        :meth:`transform`.
        """
        if isinstance(images, (str, bytes, os.PathLike)):
            raise TypeError(
                "transform_batch() expects a batch, not a single path. "
                "Use transform() for one image."
            )

        items: list[Any]
        if isinstance(images, torch.Tensor):
            if images.ndim == 3:
                items = [images]
            elif images.ndim == 4:
                if images.shape[1] in (1, 3, 4):
                    items = [images[i] for i in range(images.shape[0])]
                elif images.shape[-1] in (1, 3, 4):
                    items = [images[i] for i in range(images.shape[0])]
                else:
                    raise ValueError(
                        "4-D tensor must be NCHW or NHWC with 1, 3 or 4 channels; "
                        f"got shape {tuple(images.shape)}."
                    )
            else:
                raise ValueError(
                    "Tensor input must have 3 or 4 dimensions "
                    f"(CHW or NCHW), got {images.ndim}."
                )
        elif isinstance(images, np.ndarray):
            if images.ndim == 3:
                items = [images]
            elif images.ndim == 4:
                if images.shape[-1] in (1, 3, 4):
                    items = [images[i] for i in range(images.shape[0])]
                elif images.shape[1] in (1, 3, 4):
                    items = [images[i] for i in range(images.shape[0])]
                else:
                    raise ValueError(
                        "4-D array must be NHWC or NCHW with 1, 3 or 4 channels; "
                        f"got shape {tuple(images.shape)}."
                    )
            else:
                raise ValueError(
                    "Array input must have 3 or 4 dimensions "
                    f"(HWC/CHW or NHWC/NCHW), got {images.ndim}."
                )
        elif isinstance(images, Iterable):
            items = list(images)
        else:
            raise TypeError(
                "transform_batch() accepts a list/tuple of images, a numpy "
                "array, or a torch.Tensor; got "
                f"{type(images).__name__}."
            )

        if len(items) == 0:
            raise ValueError("transform_batch() received an empty batch.")

        tensors = [self.transform(item, augment=augment) for item in items]
        return torch.stack(tensors, dim=0)

    # -- input normalization ----------------------------------------------
    @staticmethod
    def _is_path_like(image: Any) -> bool:
        return isinstance(image, (str, bytes, os.PathLike))

    @classmethod
    def _to_pil(cls, image: Any) -> Image.Image:
        if cls._is_path_like(image):
            path = Path(os.fsdecode(image)).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")
            if not path.is_file():
                raise ValueError(f"Image path is not a file: {path}")
            try:
                with Image.open(path) as handle:
                    # convert() forces a full decode and yields a standalone RGB image
                    pil_image = handle.convert("RGB")
            except UnidentifiedImageError as exc:
                raise ValueError(
                    f"File is not a supported image: {path}"
                ) from exc
            except OSError as exc:
                raise ValueError(f"Could not read image {path}: {exc}") from exc
            return pil_image

        if isinstance(image, Image.Image):
            if image.width == 0 or image.height == 0:
                raise ValueError("PIL image has zero width or height.")
            return image if image.mode == "RGB" else image.convert("RGB")

        if isinstance(image, np.ndarray):
            return cls._numpy_to_pil(image)

        if isinstance(image, torch.Tensor):
            return cls._tensor_to_pil(image)

        raise TypeError(
            "Unsupported image type "
            f"{type(image).__name__}. Pass a file path, PIL image, numpy "
            "array, or torch.Tensor."
        )

    @staticmethod
    def _array_to_uint8(array: np.ndarray, name: str) -> np.ndarray:
        if array.dtype == np.uint8:
            return array

        if np.issubdtype(array.dtype, np.floating):
            if not np.isfinite(array).all():
                raise ValueError(f"{name} contains NaN or infinite values.")
            minimum, maximum = float(array.min()), float(array.max())
            if minimum >= 0.0 and maximum <= 1.0:
                array = array * 255.0
            elif minimum >= 0.0 and maximum <= 255.0:
                pass
            else:
                raise ValueError(
                    f"{name} float values must be in [0, 1] or [0, 255]; "
                    f"got range [{minimum:.4f}, {maximum:.4f}]."
                )
        elif np.issubdtype(array.dtype, np.integer):
            minimum, maximum = int(array.min()), int(array.max())
            if minimum < 0 or maximum > 255:
                raise ValueError(
                    f"{name} integer values must be in [0, 255]; "
                    f"got range [{minimum}, {maximum}]."
                )
        elif array.dtype == np.bool_:
            array = array.astype(np.uint8) * 255
        else:
            raise TypeError(f"{name} has unsupported dtype {array.dtype}.")

        return np.clip(np.rint(array), 0, 255).astype(np.uint8)

    @classmethod
    def _numpy_to_pil(cls, array: np.ndarray) -> Image.Image:
        if array.ndim == 2:
            gray = cls._array_to_uint8(array, "Grayscale array")
            return Image.fromarray(gray, mode="L").convert("RGB")

        if array.ndim != 3:
            raise ValueError(
                "Array images must have 2 dimensions (H, W) or 3 dimensions "
                f"(H, W, C)/(C, H, W); got shape {array.shape}."
            )

        if array.shape[-1] in (1, 3, 4):
            hwc = array
        elif array.shape[0] in (1, 3, 4):
            hwc = np.transpose(array, (1, 2, 0))
        else:
            raise ValueError(
                "Array must have 1, 3 or 4 channels in the last axis (HWC) or "
                f"first axis (CHW); got shape {array.shape}."
            )

        if hwc.shape[-1] == 1:
            hwc = hwc[..., 0]
            return cls._numpy_to_pil(hwc)

        hwc = cls._array_to_uint8(hwc, "Image array")
        mode = "RGBA" if hwc.shape[-1] == 4 else "RGB"
        return Image.fromarray(hwc, mode=mode).convert("RGB")

    @classmethod
    def _tensor_to_pil(cls, tensor: torch.Tensor) -> Image.Image:
        if tensor.ndim == 3:
            if tensor.shape[0] in (1, 3, 4):
                chw = tensor
            elif tensor.shape[-1] in (1, 3, 4):
                chw = tensor.permute(2, 0, 1)
            else:
                raise ValueError(
                    "Tensor images must have 1, 3 or 4 channels; "
                    f"got shape {tuple(tensor.shape)}."
                )
        elif tensor.ndim == 2:
            chw = tensor.unsqueeze(0)
        else:
            raise ValueError(
                "transform() expects a single CHW tensor; use "
                f"transform_batch() for batches (got {tensor.ndim} dims)."
            )

        array = chw.detach().cpu().numpy()
        if np.issubdtype(array.dtype, np.floating):
            if not np.isfinite(array).all():
                raise ValueError("Tensor contains NaN or infinite values.")
            minimum, maximum = float(array.min()), float(array.max())
            if minimum < -1e-6:
                raise ValueError(
                    "Tensor already contains negative values and looks "
                    "normalized. Pass a raw image in [0, 1] or [0, 255] to "
                    "transform(), not an already-normalized model input."
                )
            if maximum <= 1.0 + 1e-6:
                array = array * 255.0
            elif maximum > 255.0 + 1e-6:
                raise ValueError(
                    "Tensor values must be in [0, 1] or [0, 255]; "
                    f"got range [{minimum:.4f}, {maximum:.4f}]."
                )
        array = cls._array_to_uint8(array, "Tensor")
        if chw.shape[0] in (1, 3, 4):
            hwc = np.transpose(array, (1, 2, 0))
        else:
            hwc = array
        if hwc.shape[-1] == 1:
            hwc = hwc[..., 0]
            return Image.fromarray(hwc, mode="L").convert("RGB")
        mode = "RGBA" if hwc.shape[-1] == 4 else "RGB"
        return Image.fromarray(hwc, mode=mode).convert("RGB")


# ---------------------------------------------------------------------------
# Dynamic CNN (same design as the notebook, with tunable depth/width)
# ---------------------------------------------------------------------------


def conv_channels_for_depth(num_conv_layers: int) -> list[int]:
    """Return channel widths for 3-7 convolutional layers.

    The first five widths are exactly the notebook's ``32, 64, 128, 256, 512``
    schedule.  Layers six and seven reuse 512 channels, which keeps GPU memory
    bounded while preserving the existing design.
    """
    if isinstance(num_conv_layers, bool) or not isinstance(num_conv_layers, int):
        raise TypeError("num_conv_layers must be an integer.")
    if not MIN_CONV_LAYERS <= num_conv_layers <= MAX_CONV_LAYERS:
        raise ValueError(
            f"num_conv_layers must be between {MIN_CONV_LAYERS} and "
            f"{MAX_CONV_LAYERS}; got {num_conv_layers}."
        )
    return list(CONV_CHANNEL_SCHEDULE[:num_conv_layers])


class DynamicCNN(nn.Module):
    """Configurable CNN that preserves the original block design.

    Each convolutional block is ``Conv2d(3x3, padding=1) -> BatchNorm2d ->
    ReLU -> MaxPool2d(2, 2)``; the final convolutional block omits pooling and
    is followed by global average pooling, exactly like the original model.
    """

    def __init__(
        self,
        conv_channels: Sequence[int],
        hidden_units: Sequence[int],
        num_classes: int,
        in_channels: int = 3,
        dropout: float = 0.4,
        dropout_hidden: float = 0.3,
    ) -> None:
        super().__init__()

        conv_channels = [int(c) for c in conv_channels]
        hidden_units = [int(u) for u in hidden_units]

        if not conv_channels:
            raise ValueError("conv_channels must contain at least one layer.")
        if not hidden_units:
            raise ValueError("hidden_units must contain at least one FC layer.")
        if any(c <= 0 for c in conv_channels):
            raise ValueError("conv_channels values must be positive.")
        if any(u <= 0 for u in hidden_units):
            raise ValueError("hidden_units values must be positive.")
        if int(num_classes) < 2:
            raise ValueError("num_classes must be at least 2.")
        if int(in_channels) <= 0:
            raise ValueError("in_channels must be positive.")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if not 0.0 <= float(dropout_hidden) < 1.0:
            raise ValueError("dropout_hidden must be in [0, 1).")

        self.conv_channels = conv_channels
        self.hidden_units = hidden_units
        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.dropout = float(dropout)
        self.dropout_hidden = float(dropout_hidden)

        conv_layers: list[nn.Module] = []
        previous_channels = self.in_channels
        last_index = len(conv_channels) - 1
        for index, channels in enumerate(conv_channels):
            conv_layers.extend(
                [
                    nn.Conv2d(previous_channels, channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                ]
            )
            if index < last_index:
                conv_layers.append(nn.MaxPool2d(2, 2))
            previous_channels = channels

        conv_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features = nn.Sequential(*conv_layers)

        classifier: list[nn.Module] = [nn.Flatten(), nn.Dropout(p=self.dropout)]
        previous_units = conv_channels[-1]
        for index, units in enumerate(hidden_units):
            classifier.extend(
                [
                    nn.Linear(previous_units, units),
                    nn.ReLU(inplace=True),
                ]
            )
            # The original model places dropout(0.3) between the hidden FC
            # layer and the output layer; keep that behaviour for every depth.
            classifier.append(nn.Dropout(p=self.dropout_hidden))
            previous_units = units
        classifier.append(nn.Linear(previous_units, self.num_classes))
        self.classifier = nn.Sequential(*classifier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def get_config(self) -> dict[str, Any]:
        """Configuration needed to rebuild this exact architecture."""
        return {
            "conv_channels": list(self.conv_channels),
            "hidden_units": list(self.hidden_units),
            "num_classes": self.num_classes,
            "in_channels": self.in_channels,
            "dropout": self.dropout,
            "dropout_hidden": self.dropout_hidden,
        }


def build_cnn(
    conv_channels: Sequence[int],
    hidden_units: Sequence[int],
    num_classes: int,
    in_channels: int = 3,
    dropout: float = 0.4,
    dropout_hidden: float = 0.3,
) -> DynamicCNN:
    """Build a :class:`DynamicCNN` from an explicit architecture config."""
    return DynamicCNN(
        conv_channels=conv_channels,
        hidden_units=hidden_units,
        num_classes=num_classes,
        in_channels=in_channels,
        dropout=dropout,
        dropout_hidden=dropout_hidden,
    )


def model_config_from_hyperparameters(
    hyperparameters: dict[str, Any],
    num_classes: int,
    in_channels: int = 3,
    dropout: float = 0.4,
    dropout_hidden: float = 0.3,
) -> dict[str, Any]:
    """Convert Optuna trial parameters into a ``build_cnn`` configuration.

    Expected Optuna keys
    --------------------
    ``n_conv_layers`` (3-7), ``n_fc_layers`` (2-5) and ``fc_units_1`` ...
    ``fc_units_{n_fc_layers}`` (64-256).  The output layer is added
    separately and is never part of the search space.
    """
    try:
        n_conv = int(hyperparameters["n_conv_layers"])
        n_fc = int(hyperparameters["n_fc_layers"])
        hidden_units = [
            int(hyperparameters[f"fc_units_{index}"]) for index in range(1, n_fc + 1)
        ]
    except KeyError as exc:
        raise KeyError(
            f"Missing Optuna hyperparameter {exc.args[0]!r}."
        ) from exc

    if not 2 <= n_fc <= 5:
        raise ValueError(f"n_fc_layers must be between 2 and 5; got {n_fc}.")
    if any(not 64 <= units <= 256 for units in hidden_units):
        raise ValueError(
            f"Every hidden FC layer must have 64-256 neurons; got {hidden_units}."
        )

    return {
        "conv_channels": conv_channels_for_depth(n_conv),
        "hidden_units": hidden_units,
        "num_classes": int(num_classes),
        "in_channels": int(in_channels),
        "dropout": float(dropout),
        "dropout_hidden": float(dropout_hidden),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def load_preprocessing_pipeline(
    path: str | os.PathLike[str],
) -> ImagePreprocessingPipeline:
    """Load the serialized preprocessing pipeline from ``path``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Preprocessing pipeline not found: {path}")
    pipeline = joblib.load(path)
    if not isinstance(pipeline, ImagePreprocessingPipeline):
        raise TypeError(
            f"{path} does not contain an ImagePreprocessingPipeline "
            f"(got {type(pipeline).__name__})."
        )
    return pipeline


def save_model_checkpoint(
    model: nn.Module,
    path: str | os.PathLike[str],
    model_config: dict[str, Any],
    class_names: Sequence[str] | None = None,
    class_to_idx: dict[str, int] | None = None,
    preprocessing_config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save ``state_dict`` plus the configuration required to rebuild it.

    The checkpoint is deliberately **model-only**: the preprocessing pipeline
    is stored in a separate joblib file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: dict[str, Any] = {
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "model_config": dict(model_config),
    }
    if class_names is not None:
        checkpoint["class_names"] = list(class_names)
    if class_to_idx is not None:
        checkpoint["class_to_idx"] = dict(class_to_idx)
    if preprocessing_config is not None:
        checkpoint["preprocessing_config"] = dict(preprocessing_config)
    if extra:
        checkpoint.update(extra)

    torch.save(checkpoint, path)
    return path


def load_trained_model(
    path: str | os.PathLike[str], device: str | torch.device = "cpu"
) -> tuple[DynamicCNN, dict[str, Any]]:
    """Load the trained model and its checkpoint metadata.

    Returns
    -------
    (model, checkpoint)
        ``model`` is in evaluation mode on ``device``; ``checkpoint`` contains
        ``class_names``, ``model_config`` and any saved metrics.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Trained model not found: {path}")

    device = torch.device(device)
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # very old torch versions
        checkpoint = torch.load(path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise ValueError("Model checkpoint must be a dictionary.")
    if "state_dict" not in checkpoint or "model_config" not in checkpoint:
        raise ValueError(
            "Checkpoint is missing 'state_dict' or 'model_config'. "
            "Re-save it with save_model_checkpoint()."
        )

    model = build_cnn(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def _validate_model_input(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError(
            "Expected a preprocessed torch.Tensor. Run the image through "
            "ImagePreprocessingPipeline.transform() first."
        )
    if image.ndim == 3:
        image = image.unsqueeze(0)
    elif image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError(
                "predict() handles one image; use predict_batch() for a batch."
            )
    else:
        raise ValueError(
            "Expected a (3, H, W) or (1, 3, H, W) tensor; "
            f"got shape {tuple(image.shape)}."
        )
    if image.shape[1] != 3:
        raise ValueError(
            f"Expected 3 input channels (RGB), got {image.shape[1]}."
        )
    if not torch.is_floating_point(image):
        raise TypeError("Model input must be a floating-point tensor.")
    return image


def _top_k_result(
    probabilities: torch.Tensor,
    class_names: Sequence[str] | None,
    top_k: int,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    k = min(int(top_k), probabilities.numel())
    top_probabilities, top_indices = torch.topk(probabilities, k)

    entries = []
    for probability, index in zip(top_probabilities.tolist(), top_indices.tolist()):
        entry: dict[str, Any] = {
            "index": int(index),
            "probability": float(probability),
        }
        if class_names is not None:
            if not 0 <= index < len(class_names):
                raise ValueError(
                    f"Predicted index {index} is outside class_names "
                    f"(length {len(class_names)})."
                )
            entry["class_name"] = class_names[index]
        entries.append(entry)

    result: dict[str, Any] = {
        "predicted_index": int(top_indices[0]),
        "confidence": float(top_probabilities[0]),
        "top_k": entries,
    }
    if class_names is not None:
        result["predicted_class"] = class_names[int(top_indices[0])]
    return result


@torch.no_grad()
def predict(
    model: nn.Module,
    processed_image: torch.Tensor,
    class_names: Sequence[str] | None = None,
    top_k: int = 3,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Run inference on one **already preprocessed** image.

    Parameters
    ----------
    model:
        Model returned by :func:`load_trained_model`.
    processed_image:
        Tensor from ``pipeline.transform(image)`` with shape ``(3, 256, 256)``
        (or a batch of one).  This function intentionally does not preprocess
        raw images, keeping model inference separate from preprocessing.
    class_names:
        Ordered class names (index -> name), usually
        ``checkpoint["class_names"]``.
    top_k:
        Number of ranked predictions to return.
    device:
        Device to run on; defaults to the model's current device.

    Returns
    -------
    dict
        ``predicted_index``, ``predicted_class`` (if names provided),
        ``confidence`` and a ranked ``top_k`` list.
    """
    image = _validate_model_input(processed_image)
    if device is not None:
        device = torch.device(device)
    else:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    model.eval()
    logits = model(image.to(device))
    probabilities = torch.softmax(logits.float(), dim=1)[0]
    return _top_k_result(probabilities, class_names, top_k)


@torch.no_grad()
def predict_batch(
    model: nn.Module,
    processed_batch: torch.Tensor,
    class_names: Sequence[str] | None = None,
    top_k: int = 3,
    device: str | torch.device | None = None,
) -> list[dict[str, Any]]:
    """Run inference on a batch of preprocessed images."""
    if not isinstance(processed_batch, torch.Tensor):
        raise TypeError("processed_batch must be a torch.Tensor.")
    if processed_batch.ndim != 4 or processed_batch.shape[1] != 3:
        raise ValueError(
            "Expected a (N, 3, H, W) tensor; "
            f"got shape {tuple(processed_batch.shape)}."
        )
    if processed_batch.shape[0] == 0:
        raise ValueError("processed_batch is empty.")
    if not torch.is_floating_point(processed_batch):
        raise TypeError("Model input must be a floating-point tensor.")

    if device is not None:
        device = torch.device(device)
    else:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    model.eval()
    logits = model(processed_batch.to(device))
    probabilities = torch.softmax(logits.float(), dim=1)
    return [
        _top_k_result(sample_probabilities, class_names, top_k)
        for sample_probabilities in probabilities
    ]
