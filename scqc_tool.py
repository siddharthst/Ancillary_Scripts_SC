#!/usr/bin/env python3
"""
scqc_tool.py — Single-Cell QC & Preprocessing Pipeline
========================================================
A command-line tool for quality control and preprocessing of single-cell
RNA-seq datasets. Supports multiple input formats, configurable filtering,
normalization, HVG selection, dimensionality reduction, clustering, and QC plots.

Supports batch processing of multiple files in one invocation.

Usage:
    python scqc_tool.py --input data.h5ad --format h5ad [OPTIONS]
    python scqc_tool.py --input matrix_dir/ --format 10x [OPTIONS]
    python scqc_tool.py --input matrix.csv --format csv [OPTIONS]
    python scqc_tool.py --input data.h5 --format h5 [OPTIONS]
    python scqc_tool.py --input data.h5ad --config my_config.yaml

Batch mode (multiple files):
    python scqc_tool.py --input a.h5ad b.h5ad c.h5ad --format h5ad
    python scqc_tool.py --input samples/*.h5ad --format h5ad --output-dir results/
    python scqc_tool.py --config batch_config.yaml   # with inputs: [a.h5ad, b.h5ad]
"""

import argparse
import sys
import os
import logging
import warnings
import glob

# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """
    Configure the root logger for the pipeline.

    Parameters
    ----------
    log_level : str
        Logging verbosity: DEBUG | INFO | WARNING | ERROR (default INFO).
    log_file : str or None
        If provided, log messages are also written to this file path.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  —  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)

    # Suppress noisy third-party loggers unless we are in DEBUG mode
    if level > logging.DEBUG:
        for noisy in ("anndata", "scanpy", "numba", "h5py", "matplotlib"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        warnings.filterwarnings("ignore")

    return logging.getLogger("scqc")


# Module-level logger — replaced by setup_logging() at runtime
log = logging.getLogger("scqc")


# ── Lazy imports (checked at runtime) ─────────────────────────────────────────

def _require(pkg, install_hint):
    try:
        return __import__(pkg)
    except ImportError:
        log.error("Package '%s' not found. Install with: %s", pkg, install_hint)
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path: str, fmt: str):
    """
    Load data from various formats into an AnnData object.

    For CSV/TSV input the expected layout is **genes × cells** (rows = genes,
    columns = cells). The matrix is transposed automatically so that the
    returned AnnData has cells as observations. If your file is already
    cells × genes, use format ``csv-cxg`` / ``tsv-cxg`` to skip the transpose.
    """
    _require("scanpy", "pip install scanpy")
    import scanpy as sc_mod

    fmt = fmt.lower()
    log.info("Loading data  format='%s'  path='%s'", fmt, path)

    if fmt == "h5ad":
        adata = sc_mod.read_h5ad(path)

    elif fmt == "10x":
        if os.path.isdir(path):
            adata = sc_mod.read_10x_mtx(path, var_names="gene_symbols", cache=True)
        else:
            adata = sc_mod.read_10x_h5(path)

    elif fmt in ("csv", "tsv"):
        sep = "\t" if fmt == "tsv" else ","
        import pandas as pd
        import anndata
        df = pd.read_csv(path, index_col=0, sep=sep)
        # Default: genes × cells → transpose to cells × genes
        adata = anndata.AnnData(df.T)

    elif fmt in ("csv-cxg", "tsv-cxg"):
        # Explicit cells-by-genes orientation — no transpose
        sep = "\t" if fmt.startswith("tsv") else ","
        import pandas as pd
        import anndata
        df = pd.read_csv(path, index_col=0, sep=sep)
        adata = anndata.AnnData(df)

    elif fmt == "h5":
        adata = sc_mod.read_10x_h5(path)

    else:
        log.error(
            "Unknown format '%s'. Choose: h5ad | 10x | csv | tsv | "
            "csv-cxg | tsv-cxg | h5", fmt
        )
        sys.exit(1)

    adata.var_names_make_unique()
    log.info("Loaded  %d cells × %d genes", adata.n_obs, adata.n_vars)
    return adata


def save_data(adata, out_path: str):
    """Save AnnData to h5ad with error handling for I/O failures."""
    log.info("Saving processed data → %s", out_path)
    try:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        adata.write_h5ad(out_path)
    except OSError as exc:
        log.error(
            "Failed to save output to '%s': %s. "
            "Check that the path is writable and has sufficient disk space.",
            out_path, exc,
        )
        # Attempt a fallback save to the current directory
        fallback = os.path.basename(out_path)
        if fallback != out_path:
            log.warning("Attempting fallback save → ./%s", fallback)
            try:
                adata.write_h5ad(fallback)
                log.info("Fallback save succeeded → ./%s", fallback)
                return
            except OSError:
                pass
        log.error("Could not save results. Pipeline output is lost.")
        sys.exit(1)
    log.info("Saved  %d cells × %d genes", adata.n_obs, adata.n_vars)


# ══════════════════════════════════════════════════════════════════════════════
# QC metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_qc_metrics(adata, mito_prefix: str = "MT-"):
    """Annotate cells with QC metrics (n_genes, n_counts, pct_mito)."""
    import scanpy as sc

    log.info("Computing QC metrics  mito_prefix='%s'", mito_prefix)
    mito_mask = adata.var_names.str.upper().str.startswith(mito_prefix.upper())
    adata.var["mt"] = mito_mask
    n_mito = int(mito_mask.sum())
    log.info("Mitochondrial genes detected: %d  (prefix='%s')", n_mito, mito_prefix)

    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    log.info(
        "QC metrics added: n_genes_by_counts, total_counts, pct_counts_mt"
    )
    log.debug(
        "Median n_genes=%.0f  median total_counts=%.0f  median pct_mito=%.2f",
        adata.obs["n_genes_by_counts"].median(),
        adata.obs["total_counts"].median(),
        adata.obs["pct_counts_mt"].median() if "pct_counts_mt" in adata.obs else 0.0,
    )
    return adata


# ══════════════════════════════════════════════════════════════════════════════
# Filtering
# ══════════════════════════════════════════════════════════════════════════════

def filter_cells_and_genes(
    adata,
    min_genes: int = 200,
    max_genes: int = None,
    min_cells: int = 3,
    max_pct_mito: float = 20.0,
    min_counts: int = None,
    max_counts: int = None,
):
    import scanpy as sc
    import numpy as np

    n_cells_before, n_genes_before = adata.n_obs, adata.n_vars
    log.info(
        "Filtering  —  start: %d cells, %d genes", n_cells_before, n_genes_before
    )

    # Gene-level filter
    sc.pp.filter_genes(adata, min_cells=min_cells)
    log.info(
        "Gene filter (min_cells=%d)  →  %d genes remain  (%d removed)",
        min_cells, adata.n_vars, n_genes_before - adata.n_vars,
    )

    # Cell-level filters — use `is not None` so that 0 is a valid threshold
    m = np.ones(adata.n_obs, dtype=bool)

    if min_genes is not None:
        before = m.sum()
        m &= adata.obs["n_genes_by_counts"] >= min_genes
        log.debug("  min_genes=%d  removed %d cells", min_genes, before - m.sum())
    if max_genes is not None:
        before = m.sum()
        m &= adata.obs["n_genes_by_counts"] <= max_genes
        log.debug("  max_genes=%d  removed %d cells", max_genes, before - m.sum())
    if min_counts is not None:
        before = m.sum()
        m &= adata.obs["total_counts"] >= min_counts
        log.debug("  min_counts=%d  removed %d cells", min_counts, before - m.sum())
    if max_counts is not None:
        before = m.sum()
        m &= adata.obs["total_counts"] <= max_counts
        log.debug("  max_counts=%d  removed %d cells", max_counts, before - m.sum())
    if max_pct_mito is not None and "pct_counts_mt" in adata.obs:
        before = m.sum()
        m &= adata.obs["pct_counts_mt"] <= max_pct_mito
        log.debug(
            "  max_pct_mito=%.1f  removed %d cells", max_pct_mito, before - m.sum()
        )

    adata = adata[m].copy()
    log.info(
        "Cell filters applied  →  %d cells remain  (%d removed)",
        adata.n_obs, n_cells_before - adata.n_obs,
    )
    log.info(
        "Filtering complete  —  end: %d cells, %d genes", adata.n_obs, adata.n_vars
    )
    return adata


# ══════════════════════════════════════════════════════════════════════════════
# Normalization
# ══════════════════════════════════════════════════════════════════════════════

def normalize(adata, target_sum: float = 1e4, log1p: bool = True):
    import scanpy as sc

    log.info("Normalizing to %s counts per cell ...", f"{int(target_sum):,}")
    sc.pp.normalize_total(adata, target_sum=target_sum)
    if log1p:
        sc.pp.log1p(adata)
        log.info("log1p transform applied")
    adata.raw = adata
    log.info("Raw counts stored in adata.raw")
    return adata


# ══════════════════════════════════════════════════════════════════════════════
# Highly variable genes
# ══════════════════════════════════════════════════════════════════════════════

def select_hvgs(
    adata,
    n_top_genes: int = 2000,
    flavor: str = "seurat",
    min_mean: float = 0.0125,
    max_mean: float = 3.0,
    min_disp: float = 0.5,
    subset: bool = False,
):
    """
    Identify (and optionally subset to) highly variable genes.

    Parameters
    ----------
    subset : bool
        If True, the AnnData object is subset in-place to HVGs only before
        returning. If False (default), the 'highly_variable' flag is added to
        adata.var but all genes are kept; PCA can then use
        mask_var='highly_variable' to restrict computation without discarding
        data.
    """
    import scanpy as sc

    def _run_hvg(flav):
        """Run highly_variable_genes with the appropriate kwargs."""
        if flav == "seurat_v3":
            sc.pp.highly_variable_genes(
                adata,
                flavor=flav,
                n_top_genes=n_top_genes,
                subset=False,
            )
        else:
            sc.pp.highly_variable_genes(
                adata,
                flavor=flav,
                n_top_genes=n_top_genes,
                min_mean=min_mean,
                max_mean=max_mean,
                min_disp=min_disp,
                subset=False,
            )

    log.info(
        "Selecting top %d highly variable genes  flavor='%s'  subset=%s",
        n_top_genes, flavor, subset,
    )

    try:
        _run_hvg(flavor)
    except KeyError as e:
        if flavor in ("seurat", "cell_ranger"):
            log.warning(
                "HVG flavor '%s' raised KeyError (%s). "
                "Known Scanpy/pandas compatibility issue — "
                "falling back to 'seurat_v3'.",
                flavor, e,
            )
            flavor = "seurat_v3"
            _run_hvg(flavor)
        else:
            raise

    n_hvg = int(adata.var["highly_variable"].sum())
    log.info(
        "%d highly variable genes identified  (flavor used: '%s')", n_hvg, flavor
    )

    if n_hvg == 0:
        log.warning("No highly variable genes found — skipping HVG subsetting.")
        return adata

    if subset:
        adata = adata[:, adata.var["highly_variable"]].copy()
        log.info(
            "HVG subset applied  →  matrix is now %d cells × %d genes",
            adata.n_obs, adata.n_vars,
        )
    else:
        log.info(
            "HVG subset skipped — 'highly_variable' flag stored in adata.var. "
            "PCA will use mask_var='highly_variable' to restrict computation."
        )

    return adata


# ══════════════════════════════════════════════════════════════════════════════
# Dimensionality reduction
# ══════════════════════════════════════════════════════════════════════════════

def run_pca(
    adata,
    n_pcs: int = 50,
    svd_solver: str = "arpack",
    zero_center: bool = False,
    use_hvg_mask: bool = True,
    scale: bool = True,
):
    """
    Optionally scale data and run PCA.

    Parameters
    ----------
    zero_center : bool
        Passed to sc.pp.scale. True centers each gene to mean 0 (memory-
        intensive for large sparse matrices). False (default) only scales
        variance without centering, which is safer for sparse data.
    use_hvg_mask : bool
        When True (default) and adata.var contains a 'highly_variable' column,
        passes mask_var='highly_variable' to sc.tl.pca so that PCA is computed
        on HVGs only without physically subsetting the matrix.
    scale : bool
        Whether to run sc.pp.scale before PCA. Default True. Set False to skip
        scaling (e.g. when using seurat_v3 HVG flavor on raw counts).
    """
    import scanpy as sc

    log.info(
        "Running PCA  n_pcs=%d  svd_solver='%s'  zero_center=%s  "
        "use_hvg_mask=%s  scale=%s",
        n_pcs, svd_solver, zero_center, use_hvg_mask, scale,
    )

    if scale:
        log.debug("Scaling data  zero_center=%s  max_value=10", zero_center)
        sc.pp.scale(adata, zero_center=zero_center, max_value=10)
    else:
        log.info("Scaling skipped (scale=False)")

    # Determine whether to pass mask_var
    mask_var = None
    if use_hvg_mask and "highly_variable" in adata.var.columns:
        mask_var = "highly_variable"
        n_masked = int(adata.var["highly_variable"].sum())
        log.info(
            "PCA restricted to %d HVGs via mask_var='highly_variable'", n_masked
        )
    elif use_hvg_mask and "highly_variable" not in adata.var.columns:
        log.debug(
            "use_hvg_mask=True but 'highly_variable' not in adata.var — "
            "running PCA on all genes"
        )

    pca_kwargs = dict(n_comps=n_pcs, svd_solver=svd_solver)
    if mask_var is not None:
        pca_kwargs["mask_var"] = mask_var

    sc.tl.pca(adata, **pca_kwargs)

    explained = adata.uns["pca"]["variance_ratio"]
    cumulative = float(explained[:n_pcs].sum()) * 100
    log.info(
        "PCA complete  —  cumulative variance explained by %d PCs: %.1f%%",
        n_pcs, cumulative,
    )
    log.debug(
        "Top-5 PC variance ratios: %s",
        "  ".join(f"PC{i+1}={v:.4f}" for i, v in enumerate(explained[:5])),
    )
    return adata


def run_umap(adata, n_neighbors: int = 15, n_pcs: int = 40):
    import scanpy as sc

    # Clamp n_pcs to the actual number of computed PCs
    if "X_pca" in adata.obsm:
        available_pcs = adata.obsm["X_pca"].shape[1]
        if n_pcs > available_pcs:
            log.warning(
                "Requested n_pcs=%d for neighbors but only %d PCs available. "
                "Clamping to %d.",
                n_pcs, available_pcs, available_pcs,
            )
            n_pcs = available_pcs

    log.info(
        "Building kNN graph  n_neighbors=%d  n_pcs=%d", n_neighbors, n_pcs
    )
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    log.info("Computing UMAP embedding ...")
    sc.tl.umap(adata)
    log.info("UMAP complete")
    return adata


# ══════════════════════════════════════════════════════════════════════════════
# Clustering
# ══════════════════════════════════════════════════════════════════════════════

def run_clustering(
    adata,
    method: str = "leiden",
    resolutions=None,
    n_iterations: int = -1,
    key_prefix: str = None,
):
    """
    Cluster cells at one or more resolutions using Leiden or Louvain.

    Results for each resolution are stored in separate adata.obs columns so
    they can all be explored and plotted independently.
    """
    import scanpy as sc

    method = method.lower()
    if method not in ("leiden", "louvain"):
        log.error("Unknown clustering method '%s'. Choose: leiden | louvain", method)
        sys.exit(1)

    if "neighbors" not in adata.uns:
        log.error(
            "Neighbor graph not found in adata.uns. "
            "Run PCA + neighbors (UMAP step) before clustering."
        )
        sys.exit(1)

    # Normalise resolutions to a list
    if resolutions is None:
        resolutions = [0.5]
    elif isinstance(resolutions, (int, float)):
        resolutions = [float(resolutions)]
    else:
        resolutions = [float(r) for r in resolutions]

    prefix = key_prefix or method
    keys_added = []

    log.info(
        "Running %s clustering  resolutions=%s",
        method, resolutions,
    )

    for res in resolutions:
        key = (
            prefix
            if len(resolutions) == 1
            else f"{prefix}_r{res:.2f}".rstrip("0").rstrip(".")
        )

        log.info("  resolution=%.4g  →  key='%s'", res, key)

        try:
            if method == "leiden":
                sc.tl.leiden(
                    adata,
                    resolution=res,
                    n_iterations=n_iterations,
                    key_added=key,
                    flavor='igraph',
                )
            else:
                sc.tl.louvain(
                    adata,
                    resolution=res,
                    key_added=key,
                )
        except ImportError as e:
            pkg = "leidenalg/igraph" if method == "leiden" else "louvain"
            log.error("%s\n  Install with: pip install %s", e, pkg)
            sys.exit(1)

        n_clusters = adata.obs[key].nunique()
        log.info(
            "  resolution=%.4g  →  %d clusters  (stored in adata.obs['%s'])",
            res, n_clusters, key,
        )

        counts = adata.obs[key].value_counts().sort_index()
        max_count = int(counts.max())
        log.info("  Cluster sizes  (resolution=%.4g):", res)
        for cid, cnt in counts.items():
            bar = "█" * int(cnt / max_count * 20)
            log.info("    %s: %6d cells  %s", str(cid).rjust(3), cnt, bar)

        keys_added.append(key)

    log.info(
        "Clustering complete  —  %d resolution(s) stored: %s",
        len(keys_added), keys_added,
    )
    return adata, keys_added


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline record (collects per-step stats for the flowchart)
# ══════════════════════════════════════════════════════════════════════════════

class PipelineRecord:
    """
    Lightweight accumulator that collects one entry per pipeline step.

    Each entry is a dict with keys:
        label   – short step name shown in the node header
        color   – hex fill colour for the node
        params  – list of (key, value) strings shown as parameters
        result  – list of result strings shown below the params
        n_cells – cell count *after* this step (None if not applicable)
        n_genes – gene count *after* this step (None if not applicable)
    """

    _COLORS = {
        "load":       "#1a3a5c",
        "qc":         "#1d5c63",
        "filter":     "#7b2d2d",
        "normalize":  "#2d5a27",
        "hvg":        "#4a3060",
        "pca":        "#2b4a6e",
        "umap":       "#1f4e4e",
        "cluster":    "#5c3d00",
        "save":       "#2a2a2a",
        "skipped":    "#444444",
    }

    def __init__(self):
        self.steps = []

    def add(self, label: str, category: str, params: list, result: list,
            n_cells=None, n_genes=None):
        self.steps.append({
            "label":   label,
            "color":   self._COLORS.get(category, "#333333"),
            "params":  params,
            "result":  result,
            "n_cells": n_cells,
            "n_genes": n_genes,
        })


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline flowchart — Graphviz implementation
# ══════════════════════════════════════════════════════════════════════════════

def _escape_html(text: str) -> str:
    """Escape characters that are special in Graphviz HTML labels."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
    )


def _result_color(text: str) -> str:
    """Pick a colour for a result line based on its content."""
    lower = text.lower()
    if "removed" in lower or "▼" in lower:
        return "#f85149"
    if "selected" in lower or "▲" in lower or "kept" in lower:
        return "#3fb950"
    if "skipped" in lower or "disabled" in lower:
        return "#8b949e"
    return "#58a6ff"


def plot_pipeline_flowchart(record: "PipelineRecord", out_path: str):
    """
    Draw a vertical pipeline flowchart using Graphviz and save to *out_path*.

    Each step is a node with an HTML-label table containing a coloured header,
    parameter key=value rows, and colour-coded result lines. Steps are
    connected by arrows.

    Requires the ``graphviz`` Python package and the ``dot`` binary.
    Falls back silently with a warning if graphviz is unavailable.
    """
    steps = record.steps
    if not steps:
        log.warning("PipelineRecord is empty — skipping flowchart.")
        return

    try:
        import graphviz
    except ImportError:
        log.warning(
            "graphviz Python package not installed — skipping flowchart. "
            "Install with: pip install graphviz  (also needs 'dot' binary)"
        )
        return

    # Determine output format from extension
    _, ext = os.path.splitext(out_path)
    fmt = ext.lstrip(".").lower() if ext else "png"
    if fmt not in ("png", "pdf", "svg"):
        fmt = "png"

    dot = graphviz.Digraph(
        name="pipeline",
        format=fmt,
        graph_attr={
            "rankdir": "TB",
            "bgcolor": "#0d1117",
            "fontname": "Helvetica",
            "pad": "0.4",
            "nodesep": "0.25",
            "ranksep": "0.35",
            "dpi": "180",
        },
        node_attr={
            "shape": "none",
            "fontname": "Courier",
            "fontsize": "10",
        },
        edge_attr={
            "color": "#30363d",
            "penwidth": "1.8",
            "arrowsize": "0.8",
        },
    )

    # ── Title node ────────────────────────────────────────────────────────────
    dot.node(
        "title",
        label='<<FONT FACE="Helvetica" POINT-SIZE="14" COLOR="#e6edf3">'
              '<B>scqc_tool  ·  Pipeline Summary</B></FONT>>',
        shape="none",
    )

    # ── Step nodes ────────────────────────────────────────────────────────────
    prev_id = "title"
    for idx, step in enumerate(steps):
        node_id = f"step_{idx}"
        color = step["color"]

        # Badge text
        badge = ""
        if step["n_cells"] is not None:
            parts = [f'{step["n_cells"]:,} cells']
            if step["n_genes"] is not None:
                parts.append(f'{step["n_genes"]:,} genes')
            badge = "  ·  ".join(parts)

        header_label = _escape_html(step["label"].upper())

        rows = []
        # Header
        rows.append(
            f'<TR>'
            f'<TD BGCOLOR="{color}" ALIGN="LEFT" COLSPAN="1">'
            f'<FONT COLOR="#ffffff" POINT-SIZE="9">'
            f'<B> {idx+1:02d}  {header_label} </B></FONT></TD>'
            f'<TD BGCOLOR="{color}" ALIGN="RIGHT">'
            f'<FONT COLOR="#c9d1d9" POINT-SIZE="7">'
            f' {_escape_html(badge)} </FONT></TD>'
            f'</TR>'
        )

        # Param rows
        for key_str, val_str in step["params"]:
            rows.append(
                f'<TR>'
                f'<TD ALIGN="LEFT"><FONT COLOR="#8b949e" POINT-SIZE="8">'
                f' {_escape_html(key_str)} </FONT></TD>'
                f'<TD ALIGN="LEFT"><FONT COLOR="#e6edf3" POINT-SIZE="8">'
                f' {_escape_html(val_str)} </FONT></TD>'
                f'</TR>'
            )

        # Result rows
        for res_str in step["result"]:
            rc = _result_color(res_str)
            rows.append(
                f'<TR>'
                f'<TD COLSPAN="2" ALIGN="LEFT">'
                f'<FONT COLOR="{rc}" POINT-SIZE="8">'
                f'<I> {_escape_html(res_str)} </I></FONT></TD>'
                f'</TR>'
            )

        table_html = (
            f'<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" '
            f'CELLPADDING="4" BGCOLOR="#161b22" COLOR="{color}" '
            f'STYLE="ROUNDED">'
            + "\n".join(rows)
            + '</TABLE>>'
        )

        dot.node(node_id, label=table_html)
        dot.edge(prev_id, node_id)
        prev_id = node_id

    # ── Footer ────────────────────────────────────────────────────────────────
    dot.node(
        "footer",
        label='<<FONT FACE="Helvetica" POINT-SIZE="7" COLOR="#484f58">'
              'generated by scqc_tool</FONT>>',
        shape="none",
    )
    dot.edge(prev_id, "footer", style="invis")

    # ── Render ────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out_base = out_path
    if out_path.endswith(f".{fmt}"):
        out_base = out_path[: -len(fmt) - 1]

    try:
        rendered = dot.render(out_base, cleanup=True)
        log.info("Pipeline flowchart → %s", rendered)
    except graphviz.backend.execute.ExecutableNotFound:
        log.warning(
            "Graphviz 'dot' binary not found on PATH — cannot render flowchart. "
            "Install with: apt install graphviz  (or brew install graphviz)"
        )
    except Exception as exc:
        log.warning("Could not render pipeline flowchart: %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
# QC plots
# ══════════════════════════════════════════════════════════════════════════════

def make_qc_plots(adata, out_dir: str):
    import scanpy as sc
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    log.info("Saving QC figures → %s/", out_dir)

    # ── Violin plots ──────────────────────────────────────────────────────────
    metrics = [c for c in ["n_genes_by_counts", "total_counts", "pct_counts_mt"]
               if c in adata.obs.columns]
    if metrics:
        sc.pl.violin(adata, metrics, jitter=0.4, multi_panel=True, show=False)
        vpath = os.path.join(out_dir, "qc_violin.png")
        plt.savefig(vpath, dpi=150, bbox_inches="tight")
        plt.close("all")
        log.info("  Violin plot        → %s", vpath)

    # ── Scatter: n_counts vs n_genes ─────────────────────────────────────────
    if "total_counts" in adata.obs and "n_genes_by_counts" in adata.obs:
        color = "pct_counts_mt" if "pct_counts_mt" in adata.obs else None
        sc.pl.scatter(
            adata, x="total_counts", y="n_genes_by_counts",
            color=color, show=False,
        )
        spath = os.path.join(out_dir, "qc_scatter_counts_vs_genes.png")
        plt.savefig(spath, dpi=150, bbox_inches="tight")
        plt.close("all")
        log.info("  Scatter plot       → %s", spath)

    # ── HVG dispersion plot ───────────────────────────────────────────────────
    if "highly_variable" in adata.var.columns:
        try:
            sc.pl.highly_variable_genes(adata, show=False)
            hpath = os.path.join(out_dir, "hvg_dispersion.png")
            plt.savefig(hpath, dpi=150, bbox_inches="tight")
            plt.close("all")
            log.info("  HVG dispersion     → %s", hpath)
        except Exception as exc:
            log.warning("Could not save HVG dispersion plot: %s", exc)

    # ── PCA variance explained ────────────────────────────────────────────────
    if "X_pca" in adata.obsm:
        sc.pl.pca_variance_ratio(
            adata, n_pcs=min(50, adata.obsm["X_pca"].shape[1]), show=False
        )
        ppath = os.path.join(out_dir, "pca_variance_ratio.png")
        plt.savefig(ppath, dpi=150, bbox_inches="tight")
        plt.close("all")
        log.info("  PCA variance ratio → %s", ppath)

    # ── UMAP (plain) ─────────────────────────────────────────────────────────
    if "X_umap" in adata.obsm:
        sc.pl.umap(adata, show=False)
        upath = os.path.join(out_dir, "umap.png")
        plt.savefig(upath, dpi=150, bbox_inches="tight")
        plt.close("all")
        log.info("  UMAP (plain)       → %s", upath)

    # ── UMAP coloured by cluster labels ──────────────────────────────────────
    if "X_umap" in adata.obsm:
        cluster_cols = [
            c for c in adata.obs.columns
            if c.startswith("leiden") or c.startswith("louvain")
        ]
        for col in cluster_cols:
            try:
                sc.pl.umap(
                    adata, color=col, show=False,
                    legend_loc="on data",
                    title=f"UMAP — {col}",
                )
                cpath = os.path.join(out_dir, f"umap_{col}.png")
                plt.savefig(cpath, dpi=150, bbox_inches="tight")
                plt.close("all")
                log.info("  UMAP %-17s → %s", f"({col})", cpath)
            except Exception as exc:
                log.warning("Could not save UMAP plot for '%s': %s", col, exc)

    log.info("All figures saved.")


# ══════════════════════════════════════════════════════════════════════════════
# Config / YAML support
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULT_CFG = {
    "format": "h5ad",
    "output": "processed.h5ad",
    "output_dir": None,
    "plot_dir": "qc_plots",
    "mito_prefix": "MT-",
    "filter": {
        "min_genes": 200,
        "max_genes": None,
        "min_cells": 3,
        "max_pct_mito": 20.0,
        "min_counts": None,
        "max_counts": None,
    },
    "normalize": {
        "enabled": True,
        "target_sum": 1e4,
        "log1p": True,
    },
    "hvg": {
        "enabled": True,
        "n_top_genes": 2000,
        "flavor": "seurat",
        "min_mean": 0.0125,
        "max_mean": 3.0,
        "min_disp": 0.5,
        "subset": False,
    },
    "pca": {
        "enabled": True,
        "n_pcs": 50,
        "svd_solver": "arpack",
        "zero_center": False,
        "use_hvg_mask": True,
        "scale": True,
    },
    "umap": {
        "enabled": True,
        "n_neighbors": 15,
        "n_pcs": 40,
    },
    "clustering": {
        "enabled": True,
        "method": "leiden",
        "resolutions": [0.5],
        "n_iterations": -1,
        "key_prefix": None,
    },
    "plots": {
        "enabled": True,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        log.error("PyYAML not installed. Run: pip install pyyaml")
        sys.exit(1)
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    cfg = _deep_merge(_DEFAULT_CFG, raw)
    log.info("Config loaded from %s", path)
    return cfg


def write_example_config(path: str = "scqc_config_example.yaml"):
    content = """\
# ── scqc_tool example configuration ──────────────────────────────────────────
# Single input:
input: data.h5ad              # path to input file
# Batch input (overrides 'input'):
# inputs:
#   - sample_a.h5ad
#   - sample_b.h5ad
#   - samples/*.h5ad          # glob patterns are expanded

format: h5ad                  # h5ad | 10x | csv | tsv | csv-cxg | tsv-cxg | h5
output: processed.h5ad        # output H5AD path (single-file mode)
output_dir: null              # output directory for batch mode (null = same dir)
plot_dir: qc_plots            # directory for QC figures

mito_prefix: "MT-"            # mitochondrial gene prefix (human: MT-, mouse: mt-)

# ── Logging ───────────────────────────────────────────────────────────────────
logging:
  level: INFO                 # DEBUG | INFO | WARNING | ERROR
  file: null                  # path to log file, or null for stdout only

# ── Filtering ─────────────────────────────────────────────────────────────────
filter:
  min_genes: 200
  max_genes: 6000             # set null to disable
  min_cells: 3
  max_pct_mito: 20.0
  min_counts: null
  max_counts: null

# ── Normalization ─────────────────────────────────────────────────────────────
normalize:
  enabled: true
  target_sum: 10000
  log1p: true

# ── Highly Variable Genes ─────────────────────────────────────────────────────
hvg:
  enabled: true
  n_top_genes: 2000
  flavor: seurat              # seurat | cell_ranger | seurat_v3
  min_mean: 0.0125
  max_mean: 3.0
  min_disp: 0.5
  subset: false               # true  → subset matrix to HVGs
                              # false → keep all genes, use mask_var in PCA

# ── PCA ───────────────────────────────────────────────────────────────────────
pca:
  enabled: true
  n_pcs: 50
  svd_solver: arpack          # arpack | randomized | auto
  zero_center: false          # true centers genes to mean 0 (memory-intensive)
  use_hvg_mask: true          # pass mask_var='highly_variable' when subset=false
  scale: true                 # false to skip sc.pp.scale before PCA

# ── UMAP ──────────────────────────────────────────────────────────────────────
umap:
  enabled: true
  n_neighbors: 15
  n_pcs: 40

# ── Clustering ────────────────────────────────────────────────────────────────
clustering:
  enabled: true
  method: leiden              # leiden (recommended) | louvain
  resolutions:                # one or more resolution values run consecutively
    - 0.3
    - 0.5
    - 1.0
  n_iterations: -1            # leiden only: -1 = run until stable
  key_prefix: null            # obs column prefix; defaults to method name

# ── Plots ─────────────────────────────────────────────────────────────────────
plots:
  enabled: true
"""
    with open(path, "w") as f:
        f.write(content)
    log.info("Example config written → %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(cfg: dict):
    """
    Execute the full preprocessing pipeline for a single input file.

    Returns the processed AnnData object.
    """
    log.info("═" * 58)
    log.info("  scqc_tool — Single-Cell QC & Preprocessing")
    log.info("═" * 58)

    rec = PipelineRecord()

    # ── Load ─────────────────────────────────────────────────────────────────
    adata = load_data(cfg["input"], cfg.get("format", "h5ad"))
    rec.add(
        label="Load Data",
        category="load",
        params=[
            ("input",  cfg["input"]),
            ("format", cfg.get("format", "h5ad")),
        ],
        result=[f"▲ {adata.n_obs:,} cells · {adata.n_vars:,} genes loaded"],
        n_cells=adata.n_obs,
        n_genes=adata.n_vars,
    )

    # ── QC metrics ───────────────────────────────────────────────────────────
    mito_prefix = cfg.get("mito_prefix", "MT-")
    adata = compute_qc_metrics(adata, mito_prefix=mito_prefix)
    n_mito = int(adata.var.get("mt", [False]).sum()) if "mt" in adata.var else 0
    rec.add(
        label="QC Metrics",
        category="qc",
        params=[("mito_prefix", mito_prefix)],
        result=[
            f"{n_mito} mitochondrial genes detected",
            "metrics: n_genes_by_counts, total_counts, pct_counts_mt",
        ],
        n_cells=adata.n_obs,
        n_genes=adata.n_vars,
    )

    # ── Plots (pre-filter) ───────────────────────────────────────────────────
    plot_cfg = cfg.get("plots", {})
    plot_dir = cfg.get("plot_dir", "qc_plots")
    if plot_cfg.get("enabled", True):
        pre_dir = os.path.join(plot_dir, "pre_filter")
        make_qc_plots(adata, pre_dir)

    # ── Filter ───────────────────────────────────────────────────────────────
    f = cfg.get("filter", {})
    _cells_before_filter = adata.n_obs
    _genes_before_filter = adata.n_vars
    adata = filter_cells_and_genes(
        adata,
        min_genes=f.get("min_genes", 200),
        max_genes=f.get("max_genes"),
        min_cells=f.get("min_cells", 3),
        max_pct_mito=f.get("max_pct_mito", 20.0),
        min_counts=f.get("min_counts"),
        max_counts=f.get("max_counts"),
    )
    _cells_removed = _cells_before_filter - adata.n_obs
    _genes_removed = _genes_before_filter - adata.n_vars
    _filter_params = [
        ("min_genes",     str(f.get("min_genes", 200))),
        ("max_genes",     str(f.get("max_genes", "—"))),
        ("min_cells",     str(f.get("min_cells", 3))),
        ("max_pct_mito",  f"{f.get('max_pct_mito', 20.0):.1f}%"),
        ("min_counts",    str(f.get("min_counts", "—"))),
        ("max_counts",    str(f.get("max_counts", "—"))),
    ]
    rec.add(
        label="Filter Cells & Genes",
        category="filter",
        params=_filter_params,
        result=[
            f"▼ {_cells_removed:,} cells removed  →  {adata.n_obs:,} remain",
            f"▼ {_genes_removed:,} genes removed  →  {adata.n_vars:,} remain",
        ],
        n_cells=adata.n_obs,
        n_genes=adata.n_vars,
    )

    # ── Normalize ─────────────────────────────────────────────────────────────
    norm_cfg = cfg.get("normalize", {})
    if norm_cfg.get("enabled", True):
        adata = normalize(
            adata,
            target_sum=norm_cfg.get("target_sum", 1e4),
            log1p=norm_cfg.get("log1p", True),
        )
        rec.add(
            label="Normalize",
            category="normalize",
            params=[
                ("target_sum", f"{int(norm_cfg.get('target_sum', 1e4)):,}"),
                ("log1p",      str(norm_cfg.get("log1p", True))),
            ],
            result=[
                "total-count normalisation applied",
                "raw counts stored in adata.raw",
            ],
            n_cells=adata.n_obs,
            n_genes=adata.n_vars,
        )
    else:
        # Store raw layer even when normalization is skipped, so downstream
        # DE tools that expect adata.raw will not break.
        adata.raw = adata
        log.info("Normalisation skipped — adata.raw set to current state.")
        rec.add(
            label="Normalize", category="skipped",
            params=[("enabled", "false")],
            result=["skipped (adata.raw still set)"],
        )

    # ── HVGs ─────────────────────────────────────────────────────────────────
    hvg_cfg = cfg.get("hvg", {})
    hvg_subset = hvg_cfg.get("subset", False)
    if hvg_cfg.get("enabled", True):
        adata = select_hvgs(
            adata,
            n_top_genes=hvg_cfg.get("n_top_genes", 2000),
            flavor=hvg_cfg.get("flavor", "seurat"),
            min_mean=hvg_cfg.get("min_mean", 0.0125),
            max_mean=hvg_cfg.get("max_mean", 3.0),
            min_disp=hvg_cfg.get("min_disp", 0.5),
            subset=hvg_subset,
        )
        n_hvg = (
            int(adata.var["highly_variable"].sum())
            if "highly_variable" in adata.var else adata.n_vars
        )
        rec.add(
            label="Highly Variable Genes",
            category="hvg",
            params=[
                ("n_top_genes", str(hvg_cfg.get("n_top_genes", 2000))),
                ("flavor",      hvg_cfg.get("flavor", "seurat")),
                ("min_mean",    str(hvg_cfg.get("min_mean", 0.0125))),
                ("max_mean",    str(hvg_cfg.get("max_mean", 3.0))),
                ("min_disp",    str(hvg_cfg.get("min_disp", 0.5))),
                ("subset",      str(hvg_subset)),
            ],
            result=[
                f"▲ {n_hvg:,} HVGs selected",
                f"matrix: {adata.n_obs:,} cells × {adata.n_vars:,} genes"
                + ("  (subset)" if hvg_subset else "  (mask only)"),
            ],
            n_cells=adata.n_obs,
            n_genes=adata.n_vars,
        )
    else:
        rec.add(
            label="Highly Variable Genes", category="skipped",
            params=[("enabled", "false")],
            result=["skipped"],
        )

    # ── PCA ───────────────────────────────────────────────────────────────────
    pca_cfg = cfg.get("pca", {})
    if pca_cfg.get("enabled", True):
        use_hvg_mask = pca_cfg.get("use_hvg_mask", True) and not hvg_subset
        adata = run_pca(
            adata,
            n_pcs=pca_cfg.get("n_pcs", 50),
            svd_solver=pca_cfg.get("svd_solver", "arpack"),
            zero_center=pca_cfg.get("zero_center", False),
            use_hvg_mask=use_hvg_mask,
            scale=pca_cfg.get("scale", True),
        )
        cum_var = (
            float(adata.uns["pca"]["variance_ratio"][
                :pca_cfg.get("n_pcs", 50)
            ].sum()) * 100
        )
        rec.add(
            label="PCA",
            category="pca",
            params=[
                ("n_pcs",       str(pca_cfg.get("n_pcs", 50))),
                ("svd_solver",  pca_cfg.get("svd_solver", "arpack")),
                ("zero_center", str(pca_cfg.get("zero_center", False))),
                ("use_hvg_mask", str(use_hvg_mask)),
                ("scale",       str(pca_cfg.get("scale", True))),
            ],
            result=[
                f"▲ {pca_cfg.get('n_pcs', 50)} PCs · "
                f"{cum_var:.1f}% variance explained"
            ],
            n_cells=adata.n_obs,
            n_genes=adata.n_vars,
        )
    else:
        rec.add(
            label="PCA", category="skipped",
            params=[("enabled", "false")],
            result=["skipped"],
        )

    # ── UMAP ──────────────────────────────────────────────────────────────────
    umap_cfg = cfg.get("umap", {})
    if umap_cfg.get("enabled", True):
        adata = run_umap(
            adata,
            n_neighbors=umap_cfg.get("n_neighbors", 15),
            n_pcs=umap_cfg.get("n_pcs", 40),
        )
        rec.add(
            label="UMAP",
            category="umap",
            params=[
                ("n_neighbors", str(umap_cfg.get("n_neighbors", 15))),
                ("n_pcs",       str(umap_cfg.get("n_pcs", 40))),
            ],
            result=[
                "kNN graph + UMAP embedding computed",
                "stored in adata.obsm['X_umap']",
            ],
            n_cells=adata.n_obs,
            n_genes=adata.n_vars,
        )
    else:
        rec.add(
            label="UMAP", category="skipped",
            params=[("enabled", "false")],
            result=["skipped"],
        )

    # ── Clustering ────────────────────────────────────────────────────────────
    cluster_cfg = cfg.get("clustering", {})
    if cluster_cfg.get("enabled", True):
        raw_res = cluster_cfg.get(
            "resolutions", cluster_cfg.get("resolution", 0.5)
        )
        adata, cluster_keys = run_clustering(
            adata,
            method=cluster_cfg.get("method", "leiden"),
            resolutions=raw_res,
            n_iterations=cluster_cfg.get("n_iterations", -1),
            key_prefix=cluster_cfg.get("key_prefix"),
        )
        cluster_results = []
        for k in cluster_keys:
            n_cl = adata.obs[k].nunique()
            cluster_results.append(f"▲ {n_cl} clusters  →  adata.obs['{k}']")
        rec.add(
            label="Clustering",
            category="cluster",
            params=[
                ("method", cluster_cfg.get("method", "leiden")),
                ("resolutions", str(
                    raw_res if isinstance(raw_res, list) else [raw_res]
                )),
                ("n_iterations", str(cluster_cfg.get("n_iterations", -1))),
            ],
            result=cluster_results,
            n_cells=adata.n_obs,
            n_genes=adata.n_vars,
        )
    else:
        rec.add(
            label="Clustering", category="skipped",
            params=[("enabled", "false")],
            result=["skipped"],
        )

    # ── Plots (post-processing) ──────────────────────────────────────────────
    if plot_cfg.get("enabled", True):
        post_dir = os.path.join(plot_dir, "post_processing")
        make_qc_plots(adata, post_dir)

    # ── Save ──────────────────────────────────────────────────────────────────
    out = cfg.get("output", "processed.h5ad")
    save_data(adata, out)
    rec.add(
        label="Save",
        category="save",
        params=[("output", out)],
        result=[
            f"saved {adata.n_obs:,} cells × {adata.n_vars:,} genes → {out}"
        ],
        n_cells=adata.n_obs,
        n_genes=adata.n_vars,
    )

    # ── Pipeline flowchart ────────────────────────────────────────────────────
    if plot_cfg.get("enabled", True):
        flowchart_path = os.path.join(plot_dir, "pipeline_flowchart.png")
        try:
            plot_pipeline_flowchart(rec, flowchart_path)
        except Exception as exc:
            log.warning("Could not save pipeline flowchart: %s", exc)

    log.info("═" * 58)
    log.info("  Pipeline complete!")
    log.info("═" * 58)
    return adata


# ══════════════════════════════════════════════════════════════════════════════
# Batch processing
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_input_paths(raw_paths) -> list:
    """
    Expand a list of paths / glob patterns into concrete file paths.

    Accepts a single string or a list of strings (each may be a glob).
    Returns a deduplicated, sorted list of existing paths.
    """
    if isinstance(raw_paths, str):
        raw_paths = [raw_paths]

    resolved = []
    for pat in raw_paths:
        expanded = glob.glob(pat)
        if not expanded:
            if os.path.exists(pat):
                resolved.append(os.path.abspath(pat))
            else:
                log.warning("Input path does not exist (skipping): %s", pat)
        else:
            resolved.extend(os.path.abspath(p) for p in expanded)

    seen = set()
    unique = []
    for p in sorted(resolved):
        if p not in seen:
            seen.add(p)
            unique.append(p)

    return unique


def _output_path_for(
    input_path: str,
    output_dir: str = None,
    suffix: str = "_processed",
) -> str:
    """Derive an output .h5ad path for a given input file."""
    base = os.path.splitext(os.path.basename(input_path))[0]
    fname = f"{base}{suffix}.h5ad"
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, fname)
    return os.path.join(os.path.dirname(input_path) or ".", fname)


def _plot_dir_for(input_path: str, base_plot_dir: str) -> str:
    """Return a per-sample subdirectory under *base_plot_dir*."""
    stem = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(base_plot_dir, stem)


def run_batch(cfg: dict, input_paths: list):
    """
    Run the pipeline on multiple input files sequentially.

    Each file gets its own output path and plot subdirectory. The shared
    config (filtering thresholds, normalization, etc.) is applied identically
    to every sample.
    """
    n = len(input_paths)
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║  BATCH MODE  —  %d file(s) to process                   ║", n)
    log.info("╚══════════════════════════════════════════════════════════╝")

    output_dir = cfg.get("output_dir")
    base_plot_dir = cfg.get("plot_dir", "qc_plots")
    results = {}
    failures = {}

    for i, path in enumerate(input_paths, 1):
        sample_name = os.path.basename(path)
        log.info("")
        log.info("━" * 58)
        log.info("  [%d/%d]  %s", i, n, sample_name)
        log.info("━" * 58)

        sample_cfg = cfg.copy()
        sample_cfg["input"] = path
        sample_cfg["output"] = _output_path_for(path, output_dir)
        sample_cfg["plot_dir"] = _plot_dir_for(path, base_plot_dir)

        try:
            adata = run_pipeline(sample_cfg)
            results[path] = {
                "output": sample_cfg["output"],
                "n_cells": adata.n_obs,
                "n_genes": adata.n_vars,
            }
        except Exception as exc:
            log.error("Pipeline failed for '%s': %s", sample_name, exc)
            failures[path] = str(exc)
            continue

    # ── Batch summary ─────────────────────────────────────────────────────────
    log.info("")
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║  BATCH SUMMARY                                          ║")
    log.info("╚══════════════════════════════════════════════════════════╝")

    for path, info in results.items():
        log.info(
            "  ✔  %s  →  %s  (%d cells × %d genes)",
            os.path.basename(path),
            info["output"],
            info["n_cells"],
            info["n_genes"],
        )
    for path, err in failures.items():
        log.info(
            "  ✘  %s  →  FAILED: %s", os.path.basename(path), err
        )

    log.info("")
    log.info(
        "  %d succeeded, %d failed out of %d total",
        len(results), len(failures), n,
    )

    return results, failures


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser():
    parser = argparse.ArgumentParser(
        prog="scqc_tool",
        description="Single-cell QC & preprocessing pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Config shortcut ──────────────────────────────────────────────────────
    parser.add_argument(
        "--config", metavar="YAML",
        help="Path to a YAML config file. Overrides all other flags."
    )
    parser.add_argument(
        "--write-config", metavar="PATH", nargs="?",
        const="scqc_config_example.yaml",
        help="Write an example YAML config and exit."
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    log_g = parser.add_argument_group("Logging")
    log_g.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level",
    )
    log_g.add_argument(
        "--log-file", metavar="PATH", default=None,
        help="Write log output to this file in addition to stdout",
    )

    # ── I/O ──────────────────────────────────────────────────────────────────
    io = parser.add_argument_group("Input / Output")
    io.add_argument(
        "--input", "-i", metavar="PATH", nargs="+",
        help=(
            "One or more input data paths. Supports multiple files for "
            "batch processing (e.g. --input a.h5ad b.h5ad)."
        ),
    )
    io.add_argument(
        "--format", "-f", metavar="FMT",
        choices=["h5ad", "10x", "csv", "tsv", "csv-cxg", "tsv-cxg", "h5"],
        default="h5ad",
        help="Input format",
    )
    io.add_argument(
        "--output", "-o", metavar="PATH", default="processed.h5ad",
        help="Output H5AD path (single-file mode)",
    )
    io.add_argument(
        "--output-dir", metavar="DIR", default=None,
        help=(
            "Output directory for batch mode. Each file is saved as "
            "<stem>_processed.h5ad inside this directory."
        ),
    )
    io.add_argument(
        "--plot-dir", metavar="DIR", default="qc_plots",
        help="Directory for QC figures",
    )

    # ── QC ───────────────────────────────────────────────────────────────────
    qc = parser.add_argument_group("QC metrics")
    qc.add_argument(
        "--mito-prefix", default="MT-",
        help="Mitochondrial gene prefix",
    )

    # ── Filtering ─────────────────────────────────────────────────────────────
    flt = parser.add_argument_group("Filtering")
    flt.add_argument("--min-genes", type=int, default=200)
    flt.add_argument("--max-genes", type=int, default=None)
    flt.add_argument("--min-cells", type=int, default=3)
    flt.add_argument("--max-pct-mito", type=float, default=20.0)
    flt.add_argument("--min-counts", type=int, default=None)
    flt.add_argument("--max-counts", type=int, default=None)

    # ── Normalization ─────────────────────────────────────────────────────────
    nrm = parser.add_argument_group("Normalization")
    nrm.add_argument("--no-normalize", action="store_true")
    nrm.add_argument("--target-sum", type=float, default=1e4)
    nrm.add_argument("--no-log1p", action="store_true")

    # ── HVG ───────────────────────────────────────────────────────────────────
    hvg = parser.add_argument_group("Highly variable genes")
    hvg.add_argument("--no-hvg", action="store_true")
    hvg.add_argument("--n-top-genes", type=int, default=2000)
    hvg.add_argument(
        "--hvg-flavor", default="seurat",
        choices=["seurat", "cell_ranger", "seurat_v3"],
    )
    hvg.add_argument("--min-mean", type=float, default=0.0125)
    hvg.add_argument("--max-mean", type=float, default=3.0)
    hvg.add_argument("--min-disp", type=float, default=0.5)
    hvg.add_argument(
        "--hvg-subset", action="store_true", default=False,
        help=(
            "Subset the matrix to HVGs after selection. "
            "If omitted (default), genes are flagged but the full matrix is "
            "kept; PCA uses mask_var='highly_variable' instead."
        ),
    )

    # ── PCA ───────────────────────────────────────────────────────────────────
    pca = parser.add_argument_group("PCA")
    pca.add_argument("--no-pca", action="store_true")
    pca.add_argument("--n-pcs", type=int, default=50)
    pca.add_argument(
        "--svd-solver", default="arpack",
        choices=["arpack", "randomized", "auto"],
        help="SVD solver ('randomized' is faster on large datasets)",
    )
    pca.add_argument(
        "--zero-center", action="store_true", default=False,
        help=(
            "Center genes to mean 0 before PCA. "
            "Memory-intensive for large sparse matrices; off by default."
        ),
    )
    pca.add_argument(
        "--no-hvg-mask", action="store_true", default=False,
        help="Do NOT pass mask_var='highly_variable' to sc.tl.pca.",
    )
    pca.add_argument(
        "--no-scale", action="store_true", default=False,
        help="Skip sc.pp.scale before PCA.",
    )

    # ── UMAP ──────────────────────────────────────────────────────────────────
    ump = parser.add_argument_group("UMAP")
    ump.add_argument("--no-umap", action="store_true")
    ump.add_argument("--n-neighbors", type=int, default=15)

    # ── Clustering ────────────────────────────────────────────────────────────
    clu = parser.add_argument_group("Clustering")
    clu.add_argument(
        "--no-clustering", action="store_true", help="Skip clustering"
    )
    clu.add_argument(
        "--cluster-method", default="leiden",
        choices=["leiden", "louvain"],
        help="Clustering algorithm",
    )
    clu.add_argument(
        "--resolution", type=float, nargs="+", default=[0.5],
        metavar="R",
        help=(
            "One or more resolution values to run consecutively. "
            "E.g. --resolution 0.3 0.5 1.0"
        ),
    )
    clu.add_argument(
        "--n-iterations", type=int, default=-1,
        help="Leiden only: iterations (-1 = until stable)",
    )
    clu.add_argument(
        "--cluster-key-prefix", default=None, metavar="PREFIX",
        help="Prefix for adata.obs column names storing cluster labels.",
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    plt_g = parser.add_argument_group("Plots")
    plt_g.add_argument("--no-plots", action="store_true")

    return parser


def args_to_cfg(args) -> dict:
    """Convert parsed argparse namespace to a pipeline config dict."""
    return _deep_merge(_DEFAULT_CFG, {
        "format": args.format,
        "output": args.output,
        "output_dir": args.output_dir,
        "plot_dir": args.plot_dir,
        "mito_prefix": args.mito_prefix,
        "filter": {
            "min_genes": args.min_genes,
            "max_genes": args.max_genes,
            "min_cells": args.min_cells,
            "max_pct_mito": args.max_pct_mito,
            "min_counts": args.min_counts,
            "max_counts": args.max_counts,
        },
        "normalize": {
            "enabled": not args.no_normalize,
            "target_sum": args.target_sum,
            "log1p": not args.no_log1p,
        },
        "hvg": {
            "enabled": not args.no_hvg,
            "n_top_genes": args.n_top_genes,
            "flavor": args.hvg_flavor,
            "min_mean": args.min_mean,
            "max_mean": args.max_mean,
            "min_disp": args.min_disp,
            "subset": args.hvg_subset,
        },
        "pca": {
            "enabled": not args.no_pca,
            "n_pcs": args.n_pcs,
            "svd_solver": args.svd_solver,
            "zero_center": args.zero_center,
            "use_hvg_mask": not args.no_hvg_mask,
            "scale": not args.no_scale,
        },
        "umap": {
            "enabled": not args.no_umap,
            "n_neighbors": args.n_neighbors,
            "n_pcs": args.n_pcs,
        },
        "clustering": {
            "enabled": not args.no_clustering,
            "method": args.cluster_method,
            "resolutions": args.resolution,
            "n_iterations": args.n_iterations,
            "key_prefix": args.cluster_key_prefix,
        },
        "plots": {
            "enabled": not args.no_plots,
        },
    })


def main():
    parser = build_parser()
    args = parser.parse_args()

    # ── Initialise logging before anything else ──────────────────────────────
    log_level = "INFO"
    log_file = None
    if args.config:
        try:
            import yaml
            with open(args.config) as fh:
                _peek = yaml.safe_load(fh) or {}
            _lc = _peek.get("logging", {})
            log_level = _lc.get("level", "INFO")
            log_file = _lc.get("file", None)
        except Exception:
            pass
    else:
        log_level = args.log_level
        log_file = args.log_file

    global log
    log = setup_logging(log_level=log_level, log_file=log_file)

    # ── Write example config and exit ────────────────────────────────────────
    if args.write_config is not None:
        write_example_config(args.write_config)
        sys.exit(0)

    # ── Load config from YAML or CLI flags ───────────────────────────────────
    if args.config:
        cfg = load_config(args.config)
    else:
        if not args.input:
            parser.error("--input is required (or use --config)")
        cfg = args_to_cfg(args)

    # ── Resolve input paths ──────────────────────────────────────────────────
    raw_inputs = cfg.get("inputs") or cfg.get("input")
    if raw_inputs is None:
        parser.error(
            "No input files specified (--input or 'input'/'inputs' in YAML)"
        )

    input_paths = _resolve_input_paths(raw_inputs)
    if not input_paths:
        log.error("No valid input files found after resolving paths.")
        sys.exit(1)

    # ── Single vs. batch dispatch ─────────────────────────────────────────────
    if len(input_paths) == 1:
        cfg["input"] = input_paths[0]
        run_pipeline(cfg)
    else:
        run_batch(cfg, input_paths)


if __name__ == "__main__":
    main()
