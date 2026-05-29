"""End-to-end curation pipeline: dedup -> mislabel -> suboptimal -> diversity budget.

Stage weights/thresholds are explicit in ``CurateConfig`` (no hidden automatic
aggregation). The pipeline returns a ``SelectionResult`` and a ``CurationReport``;
materializing a derived dataset is a separate step (``io.materialize``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._version import __version__
from .core.dedup import (
    SuboptimalThresholds,
    find_near_duplicates,
    is_suboptimal,
    suboptimal_scores,
)
from .core.diversity import diversity_of, select_diverse
from .core.mislabel import MislabelItem, detect_mislabels
from .core.signature import signature_features
from .embed.base import Embedder
from .io.lerobot_v3 import LeRobotV3Reader
from .ir import (
    CurationReport,
    DropReason,
    EpisodeRef,
    FrameEmbedding,
    MislabelFlag,
    MislabelStatus,
    SelectionResult,
)


@dataclass
class CurateConfig:
    budget: int | None = None
    cosine_threshold: float = 0.99
    sig_depth: int = 3
    rff_dim: int = 512
    mislabel_q: float = 0.05
    mislabel_floor: float = 0.10
    drop_suboptimal: bool = True
    suboptimal: SuboptimalThresholds = field(default_factory=SuboptimalThresholds)
    max_frames: int = 8
    seed: int = 0
    n_clusters: int | None = None


def _embed_episodes(
    reader: LeRobotV3Reader, embedder: Embedder, refs: list[EpisodeRef], config: CurateConfig
) -> list[FrameEmbedding]:
    fes: list[FrameEmbedding] = []
    for ref in refs:
        frames = reader.episode_frames(ref.episode_index, max_frames=config.max_frames)
        imgs = [frames[i] for i in range(frames.shape[0])]
        vecs = embedder.embed_images(imgs)
        text = None
        if embedder.has_text_head and ref.task:
            tv = embedder.embed_text([ref.task])
            if tv is not None and tv.shape[0] > 0:
                text = tv[0]
        fes.append(
            FrameEmbedding(ref.episode_index, vecs, embedder.source, embedder.model_id, text)
        )
    return fes


def _feature_matrix(
    reader: LeRobotV3Reader, refs: list[EpisodeRef], fes: list[FrameEmbedding], config: CurateConfig
) -> tuple[np.ndarray, str]:
    """Per-episode diversity features: path signature if available for all, else image means."""
    feats: list[np.ndarray] = []
    ok = True
    for ref in refs:
        try:
            feats.append(
                signature_features(reader.episode_path(ref.episode_index), depth=config.sig_depth)
            )
        except (ValueError, NotImplementedError):
            ok = False
            break
    if ok and feats and len({f.shape[0] for f in feats}) == 1:
        return np.stack(feats), "signature"
    return np.stack([fe.mean_vector for fe in fes]), "embedding"


def curate(
    reader: LeRobotV3Reader, embedder: Embedder, config: CurateConfig | None = None
) -> tuple[SelectionResult, CurationReport]:
    config = config or CurateConfig()
    refs = reader.episode_refs()
    ep_index = [r.episode_index for r in refs]
    n_in = len(refs)

    drop_reasons: dict[int, DropReason] = {}
    if n_in == 0:
        result = SelectionResult(kept_episode_ids=[], drop_reasons={}, dedup_pairs=[])
        report = _build_report(reader, embedder, config, result, [], {}, "none", n_in)
        return result, report

    fes = _embed_episodes(reader, embedder, refs, config)
    image_means = np.stack([fe.mean_vector for fe in fes])
    features, feat_mode = _feature_matrix(reader, refs, fes, config)

    # 1) dedup
    drop_idx, dup_pairs_idx = find_near_duplicates(
        image_means, config.cosine_threshold, config.n_clusters, config.seed
    )
    for i in drop_idx:
        drop_reasons[ep_index[i]] = DropReason.NEAR_DUPLICATE
    dup_pairs = [(ep_index[a], ep_index[b]) for a, b in dup_pairs_idx]
    survivors = [i for i in range(n_in) if ep_index[i] not in drop_reasons]

    # 2) mislabel (degrade-first-class)
    items = [
        MislabelItem(ep_index[i], fes[i].mean_vector, fes[i].text_vector, refs[i].task)
        for i in survivors
    ]
    flags = detect_mislabels(items, config.mislabel_q, config.mislabel_floor)
    suspected = {f.episode_index for f in flags if f.status is MislabelStatus.SUSPECTED}
    for ep in suspected:
        drop_reasons[ep] = DropReason.MISLABEL_SUSPECTED
    survivors = [i for i in survivors if ep_index[i] not in suspected]

    # 3) suboptimal proxies (conservative defaults; heuristic)
    if config.drop_suboptimal:
        for i in list(survivors):
            acts = reader.episode_actions(ep_index[i])
            sc = suboptimal_scores(fes[i].vectors, acts)
            if is_suboptimal(sc, config.suboptimal):
                drop_reasons[ep_index[i]] = DropReason.SUBOPTIMAL
        survivors = [i for i in survivors if ep_index[i] not in drop_reasons]

    # 4) diversity budget selection
    diversity: dict[str, object] = {"feature_mode": feat_mode}
    if survivors:
        surv_feats = features[survivors]
        if config.budget is not None and config.budget < len(survivors):
            kept_local, info = select_diverse(
                surv_feats, config.budget, n_rff=config.rff_dim, seed=config.seed
            )
            kept_set = set(kept_local)
            final = [survivors[j] for j in kept_local]
            for j in range(len(survivors)):
                if j not in kept_set:
                    drop_reasons[ep_index[survivors[j]]] = DropReason.LOW_DIVERSITY
            diversity.update({k: v for k, v in info.items()})
        else:
            final = survivors
            diversity["vendi_full"] = diversity_of(
                surv_feats, n_rff=config.rff_dim, seed=config.seed
            )
            diversity["n_out"] = len(final)
    else:
        final = []

    kept_ids = sorted(ep_index[i] for i in final)
    result = SelectionResult(
        kept_episode_ids=kept_ids, drop_reasons=dict(drop_reasons), dedup_pairs=dup_pairs
    )
    report = _build_report(reader, embedder, config, result, flags, diversity, feat_mode, n_in)
    return result, report


def _build_report(
    reader: LeRobotV3Reader,
    embedder: Embedder,
    config: CurateConfig,
    result: SelectionResult,
    flags: list[MislabelFlag],
    diversity: dict[str, object],
    feat_mode: str,
    n_in: int,
) -> CurationReport:
    info = reader.info() if n_in else {}
    return CurationReport(
        source_repo=reader.root,
        tool_version=__version__,
        source_revision=str(info.get("codebase_version")) if info else None,
        embedder={
            "id": embedder.model_id,
            "source": str(embedder.source),
            "has_text_head": bool(embedder.has_text_head),
        },
        n_episodes_in=n_in,
        n_episodes_out=result.n_kept,
        kept_episode_ids=result.kept_episode_ids,
        drop_reasons={ep: str(r) for ep, r in result.drop_reasons.items()},
        dedup_pairs=result.dedup_pairs,
        diversity=diversity,
        mislabel_flags=flags,
        provenance={
            "bytes_downloaded": 0 if reader.is_local() else None,
            "streamed": True,
            "seed": config.seed,
            "sig_depth": config.sig_depth,
            "feature_mode": feat_mode,
            "thresholds": {
                "cosine": config.cosine_threshold,
                "mislabel_q": config.mislabel_q,
                "mislabel_floor": config.mislabel_floor,
            },
        },
    )
