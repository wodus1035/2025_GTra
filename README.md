# Real-time trajectory inference from temporal single-cell expression via gene-module continuity and constrained transitions

GTra is a time-resolved trajectory inference framework for time-series single-cell RNA-seq data.  
It reconstructs **cell-state trajectories and dynamic gene programs** by modeling the continuity of co-expressed gene modules across physical time, without relying on pseudotime or predefined lineage structures.

---

## Overview

GTra is designed to characterize **coordinated gene expression programs that evolve over physical time** in time-series single-cell RNA-seq data.  
While conventional trajectory inference methods primarily focus on ordering individual cells, GTra instead emphasizes how **groups of co-expressed genes are reused, reorganized, or replaced over time**, which is often difficult to capture using cell-centric or condition-based analyses.

To achieve this, GTra models trajectories by linking **gene clusters** rather than individual cells. At each time point, cell type–specific gene clusters are identified via graph-based co-expression analysis. Clusters from adjacent time points are then connected based on complementary measures of gene set overlap (Jaccard similarity) and directional consistency of expression changes (cosine similarity), resulting in a **directed inter-temporal network** that explicitly represents regulatory continuity over time.

Because such networks may admit multiple possible routes, GTra optionally supports **Answer path constraints**, which encode biologically feasible directions of regulatory progression. When provided, these constraints suppress implausible transitions such as spurious cycles or reversals of terminal states, while remaining optional for trajectory reconstruction.

Along inferred trajectories, GTra further summarizes dynamic regulatory programs into higher-order **gene modules**, capturing heterogeneous temporal patterns such as activation, repression, or transient responses. To ensure robustness, both cluster detection and trajectory construction are embedded within a **bootstrap-based statistical validation framework**, retaining only highly reproducible transitions.

Together, GTra reframes trajectory inference as the reconstruction and interpretation of **time-resolved gene regulatory programs**, providing a robust and interpretable framework for studying dynamic biological processes in development, perturbation, and disease.

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

## Quick start

GTra infers trajectories from time-series single-cell RNA-seq data by modeling transitions of gene expression modules across physical time.

Rather than ordering individual cells along a pseudotime axis, GTra reconstructs **directed trajectories of gene modules** by explicitly incorporating experimentally defined time points.

---

### Input data

GTra expects time-resolved single-cell transcriptomic data provided as an `AnnData` object with the following required annotations:

- **Expression matrix**  
  `adata.X` — gene expression matrix (cells × genes)

- **Physical time**  
  `adata.obs["time"]` — discrete time points (e.g. 0, 3, 24, 72)

- **Cell type labels**  
  `adata.obs["cell_type"]` — cell types or cell populations

- **Gene identifiers**  
  `adata.var_names` — gene symbols

---

### Output

GTra produces the following outputs:

- **Trajectory graph**  
  A directed network representing cell-state transitions across adjacent time points

- **Gene modules**  
  Trajectory-specific gene expression programs capturing coherent temporal dynamics

- **Transition statistics**  
  Statistically validated inter-temporal cluster transition scores

---

### Tutorials

Step-by-step tutorials and reproducible examples are provided in the `tutorials/` directory:

- Data preprocessing and input formatting  
- Trajectory reconstruction and visualization  
- Interpretation of gene modules and transition dynamics


---

## Applications

- Stimulus-response time-course scRNA-seq
- Developmental trajectory reconstruction
- Longitudinal disease progression analysis
- Patient-level trajectory integration

---
