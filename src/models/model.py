"""
model.py
--------
FIGNN: Frequency-Informed Graph Neural Network

Architecture:
  1. BandWeightLayer  — 4 learnable scalars (one per band)
                        applied to edge_attr before attention
  2. GATConv layer 1  — attends over neighbours, edge-aware
  3. GATConv layer 2  — projects to latent dim
  4. GlobalMeanPool   — collapses 62 node vectors → 1 vector z
  5. Output: z        — the latent embedding, shape (batch, latent_dim)

Which GAT are we using:
  torch_geometric.nn.GATConv — the standard GAT from Veličković et al. 2018
  BUT we pass edge_dim so edge attributes (PLV) influence attention weights.
  Vanilla GAT only uses node features for attention. Edge-aware GAT is newer
  (PyG added this) and is what makes our approach different — we're not just
  connecting nodes, the connection strength (PLV) actively shapes attention.

How band weights make us different from existing work:
  Most EEG-GNN papers either:
    (a) use one band only (usually alpha or theta)
    (b) concatenate all bands and treat them equally
  We let the MODEL decide band importance via softmax weights.
  After training, you read out which band got the highest weight —
  that's a scientific finding, not just an engineering choice.
  If theta dominates (expected for N-back), that validates prior
  neuroscience literature. If it doesn't, that's also interesting.

  Wei et al. 2025 (closest paper) uses fixed graph topology.
  We learn both the graph weighting AND the band importance jointly.
  That's the core novelty claim.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class BandWeightLayer(nn.Module):
    """
    4 learnable scalars — one per frequency band.
    Applied to edge attributes before graph convolution.

    Why softmax: keeps weights positive and summing to 1,
    so they're interpretable as "how much of the signal
    comes from each band." After training, read these out
    and report them — it's a result, not just a hyperparameter.
    """

    def __init__(self, n_bands: int = 4):
        super().__init__()
        # Init equal — model learns from data which band matters
        self.raw_weights = nn.Parameter(torch.ones(n_bands))

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        edge_attr : (n_edges, 4)
        Returns   : (n_edges, 4) — each band column scaled by its weight
        """
        w = F.softmax(self.raw_weights, dim=0)   # (4,) summing to 1
        return edge_attr * w                      # broadcast over edges

    def get_weights(self) -> dict:
        """Call after training to read band importance."""
        w = F.softmax(self.raw_weights, dim=0).detach().cpu().numpy()
        return {band: float(w[i])
                for i, band in enumerate(["delta", "theta", "alpha", "beta"])}


class FIGNNEncoder(nn.Module):
    """
    Full encoder. Input: one graph (one 2-second EEG epoch).
    Output: z — a latent vector of shape (1, latent_dim).

    Hyperparameters explained:
      node_in_dim  : 4 — band power features per electrode
      edge_dim     : 4 — PLV values per band per edge
      hidden_dim   : 128 before 64 — size of intermediate node representations
      latent_dim   : 64 before 32 — size of z (the output embedding)
      n_heads      : 8 before 4  — GAT attention heads in layer 1
      dropout      : 0.3 — regularisation, prevents memorising subjects
    """
    # Change FIGNNEncoder defaults:
    def __init__(self,
                 node_in_dim : int = 4,
                 edge_dim    : int = 4,
                 hidden_dim  : int = 128,   # was 64
                 latent_dim  : int = 64,    # was 32
                 n_heads     : int = 8,     # was 4
                 dropout     : float = 0.3):
        super().__init__()

        self.band_weight = BandWeightLayer(n_bands=edge_dim)

        # GAT layer 1: node_in_dim → hidden_dim * n_heads
        # concat=True means n_heads outputs are concatenated
        self.gat1 = GATConv(
            in_channels  = node_in_dim,
            out_channels = hidden_dim,
            heads        = n_heads,
            edge_dim     = edge_dim,
            dropout      = dropout,
            concat       = True,
        )

        # GAT layer 2: hidden_dim*n_heads → latent_dim
        # concat=False means heads are averaged instead
        self.gat2 = GATConv(
            in_channels  = hidden_dim * n_heads,
            out_channels = latent_dim,
            heads        = 1,
            edge_dim     = edge_dim,
            dropout      = dropout,
            concat       = False,
        )

        self.norm1   = nn.LayerNorm(hidden_dim * n_heads)
        self.norm2   = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data) -> torch.Tensor:
        """
        data : PyG Batch or Data object

        Returns z : (batch_size, latent_dim)
        """
        x          = data.x           # (total_nodes, 4)
        edge_index = data.edge_index  # (2, total_edges)
        edge_attr  = data.edge_attr   # (total_edges, 4)
        batch      = data.batch       # (total_nodes,) — which graph each node belongs to

        edge_attr = edge_attr.clamp(0, 1)

        # Apply learnable band weights to edge features
        edge_attr = self.band_weight(edge_attr)

        # GAT layer 1
        x = self.gat1(x, edge_index, edge_attr)   # (total_nodes, hidden_dim*n_heads)
        x = self.norm1(x)
        x = F.elu(x)
        x = self.dropout(x)

        # GAT layer 2
        x = self.gat2(x, edge_index, edge_attr)   # (total_nodes, latent_dim)
        x = self.norm2(x)

        # Pool: collapse all 62 node vectors per graph into one vector
        z = global_mean_pool(x, batch)
        return F.normalize(z, p=2, dim=1)

    def get_band_importance(self) -> dict:
        return self.band_weight.get_weights()


def count_parameters(model: nn.Module) -> int:
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,}")
    return total