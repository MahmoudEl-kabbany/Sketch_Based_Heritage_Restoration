"""Context-aware candidate scorer: a relational graph network over the drawing (WP-7).

The graph comes from restoration/graph_features.py. A few rounds of relational
message passing (R-GCN: a separate weight matrix per relation, mean over each
relation's neighbours, a residual connection) give every node a summary of its
surroundings. Each candidate bridge is then classified from

    [h_a + h_b, h_a * h_b, standardized candidate measurements, shape one-hot]

where h_a, h_b are the states of its two stroke ends (sum and product, so the
order of the ends does not matter) and the measurements are the scorecard's
(scoring.candidate_features, NaN -> 0 after standardizing). The output is the
probability that the bridge redraws a missing stroke; the rest of the pipeline
(threshold, ASP utility) is unchanged.

Written in plain PyTorch (index_add for the message passing); no PyTorch
Geometric needed. Training is seeded and runs with deterministic algorithms,
so a fit is repeatable on the same machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from restoration.graph_features import NODE_FEATURES, RELATIONS

SHAPES = ("smooth", "corner")


@dataclass
class GnnConfig:
    relations: Tuple[str, ...] = RELATIONS
    hidden: int = 32
    layers: int = 3
    epochs: int = 30
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    seed: int = 0
    feature_mean: List[float] = field(default_factory=list)   # candidate-measurement standardization
    feature_scale: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GnnConfig":
        d = dict(d)
        d["relations"] = tuple(d.get("relations", RELATIONS))
        return cls(**d)


class RelationalLayer(nn.Module):
    def __init__(self, hidden: int, relations: Sequence[str]):
        super().__init__()
        self.relations = list(relations)
        self.self_loop = nn.Linear(hidden, hidden)
        self.per_relation = nn.ModuleDict({r: nn.Linear(hidden, hidden, bias=False) for r in self.relations})

    def forward(self, h: torch.Tensor, edges: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.self_loop(h)
        for r in self.relations:
            e = edges.get(r)
            if e is None or e.shape[1] == 0:
                continue
            src, dst = e[0], e[1]
            messages = self.per_relation[r](h)[src]
            total = torch.zeros_like(h).index_add_(0, dst, messages)
            count = torch.zeros(h.shape[0], 1, dtype=h.dtype).index_add_(0, dst, torch.ones(len(dst), 1, dtype=h.dtype))
            out = out + total / count.clamp(min=1.0)
        return torch.relu(out) + h


class CandidateGnn(nn.Module):
    def __init__(self, n_features: int, config: GnnConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Sequential(nn.Linear(len(NODE_FEATURES), config.hidden), nn.ReLU())
        self.layers = nn.ModuleList([RelationalLayer(config.hidden, config.relations) for _ in range(config.layers)])
        self.head = nn.Sequential(
            nn.Linear(2 * config.hidden + n_features + len(SHAPES), 64), nn.ReLU(), nn.Linear(64, 1),
        )

    def forward(self, x, edges, pairs, features, shapes) -> torch.Tensor:
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h, edges)
        a, b = h[pairs[:, 0]], h[pairs[:, 1]]
        return self.head(torch.cat([a + b, a * b, features, shapes], dim=1)).squeeze(1)


# ═══════════════════════════════════════════════════════════════════════════
# Tensors for one image
# ═══════════════════════════════════════════════════════════════════════════

def _standardize(features: np.ndarray, config: GnnConfig) -> np.ndarray:
    z = (features - np.asarray(config.feature_mean)) / np.asarray(config.feature_scale)
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _tensors(graph: Dict[str, Any], features: np.ndarray, shapes: Sequence[str], config: GnnConfig):
    x = torch.from_numpy(np.asarray(graph["x"], dtype=np.float32))
    edges = {r: torch.from_numpy(np.asarray(graph["edges"][r], dtype=np.int64)) for r in config.relations}
    pairs = torch.from_numpy(np.asarray(graph["candidate_nodes"], dtype=np.int64).reshape(-1, 2))
    f = torch.from_numpy(_standardize(np.asarray(features, dtype=np.float64), config))
    s = torch.from_numpy(np.array([[shape == k for k in SHAPES] for shape in shapes], dtype=np.float32).reshape(-1, len(SHAPES)))
    return x, edges, pairs, f, s


class GnnScorer:
    """A trained CandidateGnn with its configuration; gives probabilities for one drawing's candidates."""

    def __init__(self, net: CandidateGnn, config: GnnConfig):
        self.net, self.config = net, config
        self.net.eval()

    def probability(self, graph: Dict[str, Any], features: np.ndarray, shapes: Sequence[str]) -> np.ndarray:
        if len(shapes) == 0:
            return np.zeros(0)
        with torch.no_grad():
            logits = self.net(*_tensors(graph, features, shapes, self.config))
        return torch.sigmoid(logits).numpy().astype(np.float64)

    def state(self) -> Dict[str, Any]:
        return {"config": self.config.to_dict(), "n_features": self.net.head[0].in_features
                - 2 * self.config.hidden - len(SHAPES), "weights": self.net.state_dict()}

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "GnnScorer":
        config = GnnConfig.from_dict(state["config"])
        net = CandidateGnn(int(state["n_features"]), config)
        net.load_state_dict(state["weights"])
        return cls(net, config)


def train(images: Sequence[Dict[str, Any]], config: Optional[GnnConfig] = None, log=None) -> GnnScorer:
    """Fit on labelled images (train_scoring's, each with a "graph"): binary cross-entropy, one image per step."""
    config = config or GnnConfig()
    images = [i for i in images if "graph" in i and len(i["correct"])]
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    all_features = np.vstack([i["features"] for i in images])
    mean = np.nanmean(all_features, axis=0)
    scale = np.nanstd(all_features, axis=0)
    config.feature_mean = np.nan_to_num(mean).tolist()
    config.feature_scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, 1.0).tolist()

    data = [(_tensors(i["graph"], i["features"], list(i["shape"]), config),
             torch.from_numpy(np.asarray(i["correct"], dtype=np.float32))) for i in images]
    net = CandidateGnn(all_features.shape[1], config)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    order_rng = np.random.default_rng(config.seed)
    for epoch in range(config.epochs):
        net.train()
        total = 0.0
        for k in order_rng.permutation(len(data)):
            tensors, y = data[k]
            optimizer.zero_grad()
            loss = loss_fn(net(*tensors), y)
            loss.backward()
            optimizer.step()
            total += float(loss)
        if log:
            log(f"    epoch {epoch + 1}/{config.epochs}: mean loss {total / max(len(data), 1):.4f}")
    return GnnScorer(net, config)
