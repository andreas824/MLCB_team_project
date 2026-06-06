# === src/functions.py ===
import os, re, gzip
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import anndata as ad


def _parse_cell_string(x):
    """Parse one entry of the 'x' column.

    Format: '<donor>.<barcode>.<broad>.<fine>'
    e.g.    'F1.AAACCCACACCTCTGT-1.Mic.Mic1'
    Note the barcode itself contains no dots, but is suffixed '-1'.
    Returns (donor_raw, barcode, broad, fine).
    """
    parts = x.split('.')
    # Robust to any unexpected extra dots: donor is first, fine is last,
    # broad is second-to-last, barcode is whatever sits in between.
    donor = parts[0]
    fine = parts[-1]
    broad = parts[-2]
    barcode = '.'.join(parts[1:-2])
    return donor, barcode, broad, fine


def _normalize_donor(donor_raw):
    """Collapse technical-replicate tokens onto the physical donor.

    'M24_2' -> 'M24'  (same individual, two sequencing runs in the
    combined matrix). Leaves 'F1', 'M3', etc. untouched.
    """
    m = re.match(r'^([FM]\d+)(?:_\d+)?$', donor_raw)
    return m.group(1) if m else donor_raw


def build_condition_map(gse_female, gse_male):
    """Build {donor_label -> 'MDD'/'Control'} for all 71 donors.

    Females: GSM title is the donor label directly ('F1'...), diagnosis in
             characteristics_ch1 'group: Case'/'group: Control'.
    Males:   GSM title is '<n>: ...', donor label is 'M<n>', diagnosis in
             characteristics_ch1 'group: ...(MDD)'/'group: Control'.
    """
    cond = {}

    # --- females ---
    for gsm in gse_female.gsms.values():
        label = gsm.metadata['title'][0].strip()          # 'F1'
        if not label.startswith('F'):
            continue                                       # skip male GSMs if present
        chars = gsm.metadata.get('characteristics_ch1', [])
        grp = next((c for c in chars if c.lower().startswith('group')), '')
        val = grp.split(':', 1)[1].strip().lower()
        cond[label] = 'MDD' if val in ('case', 'mdd') or 'depress' in val else 'Control'

    # --- males ---
    for gsm in gse_male.gsms.values():
        num = int(gsm.metadata['title'][0].split(':', 1)[0].strip())
        label = f'M{num}'
        chars = gsm.metadata.get('characteristics_ch1', [])
        grp = next((c for c in chars if c.lower().startswith('group')), '')
        cond[label] = 'MDD' if ('mdd' in grp.lower() or 'depress' in grp.lower()) else 'Control'

    # M24 and M25 in GSE144136 metadata are the same physical donor
    # (combined matrix labels them M24 + M24_2, merged to M24 by _normalize_donor).
    # Drop the redundant M25 key so the map has exactly 71 donors.
    cond.pop('M25', None)

    return cond


def _read_mtx_streaming(mtx_fn, chunksize=5_000_000):
    """Read a gzipped MatrixMarket file into a CSR matrix (cells x genes)
    without blowing up memory.

    The .mtx stores genes x cells with 1-based (row, col, value) triplets.
    We read the triplets in chunks via pandas, accumulate COO arrays, and
    build the transposed (cells x genes) CSR at the end.
    """
    # First non-comment line after the %%MatrixMarket header gives dimensions.
    with gzip.open(mtx_fn, 'rt') as f:
        for line in f:
            if line.startswith('%'):
                continue
            n_genes, n_cells, nnz = map(int, line.split())
            break

    # Read the triplet body in chunks. skiprows covers header + dim line.
    # Count comment lines so skiprows is exact.
    n_comment = 0
    with gzip.open(mtx_fn, 'rt') as f:
        for line in f:
            if line.startswith('%'):
                n_comment += 1
            else:
                break
    skip = n_comment + 1   # + the dimension line

    rows = np.empty(nnz, dtype=np.int32)   # gene indices
    cols = np.empty(nnz, dtype=np.int32)   # cell indices
    vals = np.empty(nnz, dtype=np.float32)
    pos = 0
    reader = pd.read_csv(mtx_fn, sep=r'\s+', header=None, skiprows=skip,
                         dtype={0: np.int32, 1: np.int32, 2: np.float32},
                         chunksize=chunksize, compression='gzip')
    for chunk in reader:
        k = len(chunk)
        rows[pos:pos+k] = chunk[0].values - 1   # 1-based -> 0-based
        cols[pos:pos+k] = chunk[1].values - 1
        vals[pos:pos+k] = chunk[2].values
        pos += k

    # Build COO as genes x cells, then transpose to cells x genes CSR.
    coo = sp.coo_matrix((vals, (rows, cols)), shape=(n_genes, n_cells))
    return coo.T.tocsr()


def load_dataset(raw_dir, condition_map):
    """Load the combined GSE213982 matrix into a clean AnnData.

    Returns AnnData (cells x genes) with obs columns:
        donor_id, barcode, cell_type_broad, cell_type_fine, sex, condition, dataset
    Raw integer counts are kept in .X (and mirrored in .layers['counts']).
    """
    gse_dir = os.path.join(raw_dir, 'GSE213982')
    mtx_fn   = os.path.join(gse_dir, 'GSE213982_combined_counts_matrix.mtx.gz')
    cells_fn = os.path.join(gse_dir, 'GSE213982_combined_counts_matrix_cells_columns.csv.gz')
    genes_fn = os.path.join(gse_dir, 'GSE213982_combined_counts_matrix_genes_rows.csv.gz')

    # --- matrix: streaming read to stay under free-Colab RAM ---
    mat = _read_mtx_streaming(mtx_fn)    # cells x genes, CSR

    # --- genes (rows of original matrix) ---
    genes = pd.read_csv(genes_fn)['x'].astype(str).values

    # --- cells (columns of original matrix) ---
    cells = pd.read_csv(cells_fn)['x'].astype(str).values
    parsed = [_parse_cell_string(x) for x in cells]
    donor_raw = [p[0] for p in parsed]
    barcode   = [p[1] for p in parsed]
    broad     = [p[2] for p in parsed]
    fine      = [p[3] for p in parsed]

    donor_id = [_normalize_donor(d) for d in donor_raw]
    sex = ['female' if d.startswith('F') else 'male' for d in donor_id]
    condition = [condition_map[d] for d in donor_id]   # KeyError = unmapped donor (good: fail loud)

    obs = pd.DataFrame({
        'donor_id':        donor_id,
        'barcode':         barcode,
        'cell_type_broad': broad,
        'cell_type_fine':  fine,
        'sex':             sex,
        'condition':       condition,
        'dataset':         'GSE213982',
    }, index=cells)   # original full string stays as the obs index (unique)

    adata = ad.AnnData(X=mat, obs=obs,
                       var=pd.DataFrame(index=genes))
    adata.layers['counts'] = adata.X.copy()   # preserve raw counts before any normalization

    return adata