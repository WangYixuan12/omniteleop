"""Environment preparation for SAPIEN's Vulkan renderer."""

from __future__ import annotations

import os
from pathlib import Path
from typing import MutableMapping

DEFAULT_NVIDIA_VULKAN_ICD = Path("/usr/share/vulkan/icd.d/nvidia_icd.json")


def prepare_sapien_render_env(
    *,
    env: MutableMapping[str, str] | None = None,
    nvidia_icd_path: str | os.PathLike[str] = DEFAULT_NVIDIA_VULKAN_ICD,
) -> None:
    """Select the NVIDIA Vulkan ICD by default, without overriding user settings.

    SAPIEN 3.0.0b1 can fail during renderer initialization when the Vulkan loader
    enumerates every system ICD on mixed-driver machines. If the NVIDIA ICD exists,
    make it the default for this process before importing ``sapien``.
    """
    target = os.environ if env is None else env
    icd = Path(nvidia_icd_path)
    if not icd.is_file():
        return
    target.setdefault("VK_ICD_FILENAMES", str(icd))
    target.setdefault("CUDA_VISIBLE_DEVICES", "0")
