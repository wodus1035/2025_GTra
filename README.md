# Real-time trajectory inference from temporal single-cell expression via gene-module continuity and constrained transitions

GTra is a time-resolved trajectory inference framework for time-series single-cell RNA-seq data.  
It reconstructs **cell-state trajectories and dynamic gene programs** by modeling the continuity of co-expressed gene modules across physical time, without relying on pseudotime or predefined lineage structures.

---

## Overview

GTra directly leverages **longitudinal single-cell transcriptomic data** and infers trajectories by tracking how gene expression modules evolve and transition across adjacent time points.

<p align="center">
  <img src="./imgs/GTra_overview.png" width="900">
</p>

*Overview of the GTra framework.*

---

## Key concepts illustrated above

- **Longitudinal input data**  
  Time-series scRNA-seq profiles collected at discrete physical time points.

- **Gene-cluster identification within each time point**  
  Cell type–specific gene clusters are identified using gene–gene neighborhood graphs.

- **Inter-temporal transition scoring**  
  Gene clusters from adjacent time points are compared using  
  *Jaccard similarity* (gene overlap) and *cosine similarity* (expression directionality), yielding ranked transition scores.

- **Statistical testing and constrained transitions (Answer path)**  
  Transition edges are statistically validated and filtered using biologically motivated path constraints to remove implausible connections.

- **Trajectory reconstruction and gene module detection**  
  Directed transition networks are assembled into full trajectories, and genes along each trajectory are summarized into representative temporal expression modules.

---

## Quick start

GTra infers trajectories from time-series single-cell RNA-seq data by modeling transitions of gene expression modules across physical time.

### Input

GTra expects a time-resolved single-cell expression matrix provided as an `AnnData` object, with the following annotations:

- `adata.X`  
  Gene expression matrix (cells × genes)
- `adata.obs["time"]`  
  Discrete physical time points (e.g. 0, 3, 24, 72)
- `adata.obs["cell_type"]`  
  Cell type or population labels
- `adata.var_names`  
  Gene symbols

---

### Basic usage

```python
import gtra
import scanpy as sc

# Load time-series single-cell data
adata = sc.read_h5ad("example_time_series.h5ad")

# Initialize GTra object
g = gtra.GTra(
    adata,
    time_key="time",
    celltype_key="cell_type"
)

# Run trajectory inference
g.fit()

# Retrieve inferred trajectories
trajectories = g.get_trajectories()

# Extract gene modules along trajectories
modules = g.get_gene_modules()

# Plot inferred trajectories
g.plot_trajectory_graph()

# Plot gene module dynamics
g.plot_gene_modules()

---

## Applications

- Stimulus-response time-course scRNA-seq
- Developmental trajectory reconstruction
- Longitudinal disease progression analysis
- Patient-level trajectory integration

---
