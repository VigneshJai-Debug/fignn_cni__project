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
PLV_THRESHOLD = 0.3   # edges below this are dropped


def compute_band_power(band_epoch_dict: dict, epoch_idx: int) -> np.ndarray:
    """
    Compute mean band power per channel per band for one epoch.

    Power = mean of squared signal amplitude.

    Returns:
        node_features : (n_ch, 4)  — one row per electrode, one col per band
    """
    features = []
    for band in BANDS:
        epoch_data = band_epoch_dict[band][epoch_idx]   # (n_ch, n_times)
        power = np.mean(epoch_data ** 2, axis=1)        # (n_ch,)
        features.append(power)

    return np.stack(features, axis=1).astype(np.float32)   # (n_ch, 4)


def plv_to_graph(plv_matrices: dict,
                 node_features: np.ndarray,
                 label: int,
                 threshold: float = PLV_THRESHOLD) -> Data:
    """
    Build a PyG Data object from PLV matrices and node features.

    Args:
        plv_matrices  : { band: (n_ch, n_ch) }
        node_features : (n_ch, 4)
        label         : int 0/1/2
        threshold     : minimum PLV to keep an edge

    Returns:
        PyG Data object
    """
    n_ch = node_features.shape[0]

    # Stack PLV across bands → (n_ch, n_ch, 4)
    plv_stack = np.stack([plv_matrices[b] for b in BANDS], axis=-1)

    # Average PLV across bands to decide which edges to keep
    plv_mean = plv_stack.mean(axis=-1)   # (n_ch, n_ch)

    # Find upper-triangle pairs above threshold (undirected graph)
    rows, cols = np.where((plv_mean > threshold) & (np.triu(np.ones((n_ch, n_ch)), k=1) > 0))

    if len(rows) == 0:
        # Fallback: keep top 10% of edges even if below threshold
        flat = plv_mean[np.triu_indices(n_ch, k=1)]
        cutoff = np.percentile(flat, 90)
        rows, cols = np.where((plv_mean >= cutoff) & (np.triu(np.ones((n_ch, n_ch)), k=1) > 0))

    # Edge attributes: PLV in each band for each edge
    edge_attr_np = plv_stack[rows, cols]   # (n_edges, 4)

    # Make undirected: add both directions
    edge_index = torch.tensor(
        np.stack([np.concatenate([rows, cols]),
                  np.concatenate([cols, rows])], axis=0),
        dtype=torch.long
    )

    edge_attr = torch.tensor(
        np.concatenate([edge_attr_np, edge_attr_np], axis=0),
        dtype=torch.float32
    )

    x = torch.tensor(node_features, dtype=torch.float32)
    y = torch.tensor([label], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def build_graph_dataset(processed_file: dict,
                        plv_list: list) -> list:
    """
    Build all graph Data objects for one processed .set file.

    Args:
        processed_file : output of preprocess.process_file()
        plv_list       : output of plv.batch_compute_plv()

    Returns:
        List of PyG Data objects, one per epoch
    """
    band_epoch_dict = processed_file["band_epochs"]
    label           = processed_file["label"]
    n_epochs        = processed_file["n_epochs"]

    graphs = []
    for i in range(n_epochs):
        node_feats   = compute_band_power(band_epoch_dict, i)
        plv_matrices = plv_list[i]
        graph        = plv_to_graph(plv_matrices, node_feats, label)
        graphs.append(graph)

    print(f"    Built {len(graphs)} graphs  |  "
          f"sample edges: {graphs[0].edge_index.shape[1]}  |  "
          f"label: {label}")
    return graphs