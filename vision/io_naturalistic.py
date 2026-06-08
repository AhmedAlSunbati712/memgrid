from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import re

import numpy as np


@dataclass(frozen=True)
class NaturalisticCategory:
    category: str
    image_dir: Path
    image_paths: tuple[Path, ...]
    filenames: tuple[str, ...]
    concepts: tuple[str, ...]
    similarity_matrix: np.ndarray
    image_size: tuple[int, int]
    integrity: dict[str, object]


def _concept_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    match = re.match(r"^(.*?)(\d+)$", stem)
    return match.group(1) if match else stem


def load_peterson_dataset(pkl_path: str | Path) -> dict[str, dict[str, object]]:
    with Path(pkl_path).open("rb") as handle:
        return pickle.load(handle)


def load_naturalistic_category(
    category: str,
    *,
    dataset_pkl: str | Path = "datasets_peterson.pkl",
    image_root: str | Path = ".",
    required_exemplars: int | None = None,
) -> NaturalisticCategory:
    payload = load_peterson_dataset(dataset_pkl)
    if category not in payload:
        raise KeyError(f"Unknown category: {category}")

    category_data = payload[category]
    filenames = tuple(str(name) for name in category_data["fnames"])
    similarity = np.asarray(category_data["similarity"], dtype=np.float64)
    image_dir = Path(image_root) / category
    disk_files = {path.name for path in image_dir.iterdir() if path.is_file() and not path.name.startswith(".")}
    rating_files = set(filenames)

    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError(f"{category} similarity matrix must be square")
    if similarity.shape[0] != len(filenames):
        raise ValueError(f"{category} similarity matrix size does not match filename count")

    missing_on_disk = sorted(rating_files - disk_files)
    extra_on_disk = sorted(disk_files - rating_files)
    if missing_on_disk:
        raise ValueError(f"{category} is missing rated files on disk: {missing_on_disk[:5]}")

    concepts = tuple(_concept_from_filename(filename) for filename in filenames)
    concept_histogram = __import__("collections").Counter(concepts)
    dropped_concepts: list[str] = []
    if required_exemplars is not None:
        kept_indices = [idx for idx, concept in enumerate(concepts) if concept_histogram[concept] == required_exemplars]
        dropped_concepts = sorted({concept for concept, count in concept_histogram.items() if count != required_exemplars})
        filenames = tuple(filenames[idx] for idx in kept_indices)
        concepts = tuple(concepts[idx] for idx in kept_indices)
        similarity = similarity[np.ix_(kept_indices, kept_indices)]
        rating_files = set(filenames)
        missing_on_disk = sorted(rating_files - disk_files)
        if missing_on_disk:
            raise ValueError(f"{category} filtered subset is missing rated files on disk: {missing_on_disk[:5]}")
        concept_histogram = __import__("collections").Counter(concepts)

    image_paths = tuple(image_dir / filename for filename in filenames)
    from PIL import Image

    with Image.open(image_paths[0]) as img:
        image_size = img.size

    integrity = {
        "category": category,
        "n_images": len(filenames),
        "matrix_shape": list(similarity.shape),
        "is_symmetric": bool(np.allclose(similarity, similarity.T)),
        "diag_mean": float(np.mean(np.diag(similarity))),
        "missing_on_disk": missing_on_disk,
        "extra_on_disk": extra_on_disk,
        "n_concepts": len(set(concepts)),
        "concept_histogram": {
            key: int(value)
            for key, value in sorted(concept_histogram.items(), key=lambda item: item[0])
        },
        "image_size": list(image_size),
        "required_exemplars": required_exemplars,
        "dropped_concepts": dropped_concepts,
    }
    return NaturalisticCategory(
        category=category,
        image_dir=image_dir,
        image_paths=image_paths,
        filenames=filenames,
        concepts=concepts,
        similarity_matrix=similarity,
        image_size=image_size,
        integrity=integrity,
    )


def build_leave_one_exemplar_out_folds(
    concepts: tuple[str, ...] | list[str],
) -> list[tuple[np.ndarray, np.ndarray]]:
    concept_to_indices: dict[str, list[int]] = {}
    for idx, concept in enumerate(concepts):
        concept_to_indices.setdefault(str(concept), []).append(idx)

    exemplar_counts = {concept: len(indices) for concept, indices in concept_to_indices.items()}
    if len(set(exemplar_counts.values())) != 1:
        raise ValueError("All concepts must have the same exemplar count for leave-one-exemplar-out folds")

    n_exemplars = next(iter(exemplar_counts.values()))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for heldout_pos in range(n_exemplars):
        stored: list[int] = []
        probes: list[int] = []
        for concept in sorted(concept_to_indices):
            indices = sorted(concept_to_indices[concept])
            probes.append(indices[heldout_pos])
            stored.extend(idx for pos, idx in enumerate(indices) if pos != heldout_pos)
        folds.append((np.asarray(stored, dtype=np.int64), np.asarray(probes, dtype=np.int64)))
    return folds
