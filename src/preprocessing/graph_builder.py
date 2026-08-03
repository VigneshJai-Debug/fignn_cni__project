"""
graph_builder.py
----------------
Converts PLV matrices + band power into PyTorch Geometric Data objects.

One Data object = one epoch = one graph fed to the GAT.

Graph structure:
  Nodes  : electrodes (62 of them)
  Edges  : pairs of electrodes where PLV > threshold in ANY band
  x      : node features  — (n_ch, 4)  band power per electrode
  edge_index : (2, n_edges) — standard PyG sparse format
  edge_attr  : (n_edges, 4) — PLV value in each band per edge
  y      : label (0/1/2)   — cognitive load level

Why band power as node features:
  Each electrode's local spectral content tells the GAT what that
  region is doing. PLV on the edges tells it how regions communicate.
  Together: local state + connectivity = full picture.

Why sparse (threshold) instead of full 62x62 graph:
  Full graph = 62*61/2 = 1891 edges. Most near-zero PLV edges are
  noise. Thresholding keeps ~top 30% of connections, speeds up
  attention computation, and reduces overfitting.
"""

import numpy as np
import torch
from torch_geometric.data import Data

BANDS = ["delta", "theta", "alpha", "beta"]
PLV_THRESHOLD = 0.5


def plv_to_graph(connectivity: dict,
                 node_features: np.ndarray,
                 label: int,
                 threshold: float = PLV_THRESHOLD) -> Data:
    """
    Build PyG graph using PLV for edge selection + 
    PLV + correlation as 8-dimensional edge features.
    
    edge_attr shape: (n_edges, 8)
        columns 0-3: PLV per band
        columns 4-7: correlation per band
    """
    n_ch = node_features.shape[0]

    plv_matrices  = connectivity['plv']
    corr_matrices = connectivity['corr']

    # Stack PLV across bands → (n_ch, n_ch, 4)
    plv_stack  = np.stack([plv_matrices[b]  for b in BANDS], axis=-1)
    corr_stack = np.stack([corr_matrices[b] for b in BANDS], axis=-1)

    # Use mean PLV to decide which edges to keep
    plv_mean = plv_stack.mean(axis=-1)

    rows, cols = np.where(
        (plv_mean > threshold) &
        (np.triu(np.ones((n_ch, n_ch)), k=1) > 0)
    )

    if len(rows) == 0:
        cutoff = np.percentile(
            plv_mean[np.triu_indices(n_ch, k=1)], 90
        )
        rows, cols = np.where(
            (plv_mean >= cutoff) &
            (np.triu(np.ones((n_ch, n_ch)), k=1) > 0)
        )

    # Edge features: PLV (4) + correlation (4) = 8 per edge
    plv_feat  = plv_stack[rows, cols]   # (n_edges, 4)
    corr_feat = corr_stack[rows, cols]  # (n_edges, 4)
    edge_feat = np.concatenate([plv_feat, corr_feat], axis=1)  # (n_edges, 8)

    # Undirected: add reverse edges
    edge_index = torch.tensor(
        np.stack([np.concatenate([rows, cols]),
                  np.concatenate([cols, rows])], axis=0),
        dtype=torch.long
    )
    edge_attr = torch.tensor(
        np.concatenate([edge_feat, edge_feat], axis=0),
        dtype=torch.float32
    )

    x = torch.tensor(node_features, dtype=torch.float32)
    y = torch.tensor([label], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def build_graph_dataset(processed_file: dict,
                        connectivity_list: list) -> list:
    """
    Build all graphs for one .set file.
    connectivity_list: output of batch_compute_plv (now returns plv+corr)
    """
    band_epoch_dict = processed_file["band_epochs"]
    label           = processed_file["label"]
    n_epochs        = processed_file["n_epochs"]

    graphs = []
    for i in range(n_epochs):
        node_feats   = compute_band_power(band_epoch_dict, i)
        connectivity = connectivity_list[i]
        graph        = plv_to_graph(connectivity, node_feats, label)
        graphs.append(graph)

    print(f"    Built {len(graphs)} graphs  |  "
          f"sample edges: {graphs[0].edge_index.shape[1]}  |  "
          f"label: {label}")
    return graphs


def compute_band_power(band_epoch_dict: dict, epoch_idx: int) -> np.ndarray:
    """
    Compute relative band power per channel per band for one epoch.
    Relative = each band power / total power across all bands.
    Values sum to 1.0 per electrode — removes absolute amplitude differences
    between subjects and sessions, preserving spectral shape information.

    Returns:
        node_features : (n_ch, 4)  — relative band power per electrode
    """
    powers = []
    for band in BANDS:
        epoch_data = band_epoch_dict[band][epoch_idx]   # (n_ch, n_times)
        power = np.mean(epoch_data ** 2, axis=1)        # (n_ch,)
        powers.append(power)

    powers   = np.stack(powers, axis=1)          # (n_ch, 4)
    total    = powers.sum(axis=1, keepdims=True)  # (n_ch, 1)
    relative = powers / (total + 1e-30)           # (n_ch, 4) sums to 1
    return relative.astype(np.float32)