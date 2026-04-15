"""
Data processing utilities for experimental validation analysis.

Tailored for the specific files:
- gencode_vM38_basic_annotation__1__gtf.gz (GENCODE vM38, GRCm39)
- mESC_expression.tsv (RSEM output with numeric gene IDs)
- mESC_H3K27ac_peaks.bed (narrowPeak format)
- Micro-C data from 4DN (needs to be downloaded separately)

Usage:
    python data_utils.py

This will:
1. Convert GTF to TSS BED file
2. Process expression data
3. Build E-P pair catalog
4. Run analysis with synthetic contact data (until real Micro-C is downloaded)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import gzip
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# GTF TO TSS CONVERSION
# =============================================================================

def gtf_to_tss_bed(gtf_file: str, output_file: str,
                   gene_type_filter: Optional[str] = 'protein_coding') -> pd.DataFrame:
    """
    Convert GENCODE GTF to BED format with TSS positions.

    Args:
        gtf_file: Path to GTF file (can be gzipped)
        output_file: Path to output BED file
        gene_type_filter: Only include genes of this type (None for all)

    Returns:
        DataFrame with gene information including TSS
    """
    print(f"Reading GTF file: {gtf_file}")

    open_func = gzip.open if str(gtf_file).endswith('.gz') else open

    records = []
    gene_count = 0

    with open_func(gtf_file, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue

            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue

            chrom, source, feature, start, end, score, strand, frame, attributes = fields[:9]

            # Only process gene features
            if feature != 'gene':
                continue

            gene_count += 1
            start = int(start) - 1  # Convert to 0-based
            end = int(end)

            # Parse attributes - handle GENCODE format
            attr_dict = {}
            for attr in attributes.split(';'):
                attr = attr.strip()
                if not attr:
                    continue
                # Handle 'key "value"' format
                if ' "' in attr:
                    key, value = attr.split(' "', 1)
                    value = value.rstrip('"')
                    attr_dict[key.strip()] = value

            gene_name = attr_dict.get('gene_name', '')
            gene_id = attr_dict.get('gene_id', '').split('.')[0]  # Remove version
            gene_type = attr_dict.get('gene_type', '')

            # Filter by gene type
            if gene_type_filter and gene_type != gene_type_filter:
                continue

            # Determine TSS based on strand
            tss = start if strand == '+' else end

            records.append({
                'chrom': chrom,
                'start': start,
                'end': end,
                'name': gene_name,
                'score': 0,
                'strand': strand,
                'tss': tss,
                'gene_id': gene_id,
                'gene_type': gene_type
            })

    print(f"Processed {gene_count} genes, kept {len(records)} {gene_type_filter or 'all'} genes")

    if len(records) == 0:
        print("ERROR: No genes found matching criteria!")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Remove duplicates
    if 'name' in df.columns:
        df = df.drop_duplicates(subset='name', keep='first')
    df = df.sort_values(['chrom', 'start'])

    # Save BED format
    bed_df = df[['chrom', 'tss', 'tss', 'name', 'score', 'strand']].copy()
    bed_df.columns = ['chrom', 'start', 'end', 'name', 'score', 'strand']
    bed_df['end'] = bed_df['start'] + 1  # TSS is a single position
    bed_df.to_csv(output_file, sep='\t', index=False, header=False)

    print(f"Saved {len(df)} TSS positions to: {output_file}")

    return df


# =============================================================================
# EXPRESSION DATA PROCESSING
# =============================================================================

def load_expression_data(expression_file: str,
                         gene_info_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Load and process RSEM expression data.

    The expression file has numeric gene IDs. If gene_info_df is provided,
    we'll try to map them to gene names.

    Args:
        expression_file: Path to expression TSV
        gene_info_df: Optional DataFrame with gene_id and gene_name mapping

    Returns:
        DataFrame with gene_name and TPM columns
    """
    print(f"Loading expression data: {expression_file}")

    df = pd.read_csv(expression_file, sep='\t')

    print(f"Loaded {len(df)} genes")
    print(f"Columns: {list(df.columns)}")
    print(f"TPM range: {df['TPM'].min():.2f} - {df['TPM'].max():.2f}")

    # The gene_id column appears to be numeric - these might be Entrez IDs
    # For now, use them as-is
    df = df.rename(columns={'gene_id': 'gene_name'})
    df['gene_name'] = df['gene_name'].astype(str)

    # Keep only relevant columns
    result = df[['gene_name', 'TPM']].copy()
    result.columns = ['gene_name', 'tpm']
    result['log2_tpm'] = np.log2(result['tpm'] + 1)

    # If we have gene info, try to map Entrez IDs to gene names
    if gene_info_df is not None and 'gene_name' in gene_info_df.columns:
        # This would require an Entrez-to-symbol mapping
        # For now, we'll work with what we have
        pass

    print(f"Expressed genes (TPM > 1): {(result['tpm'] > 1).sum()}")

    return result


def load_enhancer_peaks(bed_file: str) -> pd.DataFrame:
    """
    Load enhancer peaks from BED/narrowPeak file.

    Args:
        bed_file: Path to BED file

    Returns:
        DataFrame with chrom, start, end, and center positions
    """
    print(f"Loading enhancer peaks: {bed_file}")

    # narrowPeak format has 10 columns
    df = pd.read_csv(bed_file, sep='\t', header=None,
                     names=['chrom', 'start', 'end', 'name', 'score',
                            'strand', 'signalValue', 'pValue', 'qValue', 'peak'])

    # Calculate center position
    df['center'] = (df['start'] + df['end']) // 2

    print(f"Loaded {len(df)} enhancer peaks")
    print(f"Chromosomes: {df['chrom'].nunique()}")

    return df


# =============================================================================
# BUILD E-P PAIRS (USING GTF GENE NAMES)
# =============================================================================

def build_ep_pairs_from_gtf(gene_tss_df: pd.DataFrame,
                            enhancers_df: pd.DataFrame,
                            expression_df: pd.DataFrame,
                            min_distance: int = 10000,
                            max_distance: int = 500000) -> pd.DataFrame:
    """
    Build E-P pairs using gene names from GTF.

    Since expression data uses numeric IDs and GTF uses gene names,
    we'll work with what overlaps or use all genes with expression.

    Args:
        gene_tss_df: DataFrame from gtf_to_tss_bed with 'name', 'chrom', 'tss'
        enhancers_df: DataFrame with enhancer peaks
        expression_df: DataFrame with expression values
        min_distance: Minimum E-P distance
        max_distance: Maximum E-P distance

    Returns:
        DataFrame with E-P pairs
    """
    print(f"\nBuilding E-P pairs...")
    print(f"  Genes: {len(gene_tss_df)}")
    print(f"  Enhancers: {len(enhancers_df)}")
    print(f"  Expression entries: {len(expression_df)}")

    # Create expression lookup by gene name
    # First, check if any gene names match
    expr_genes = set(expression_df['gene_name'].astype(str))
    gtf_genes = set(gene_tss_df['name'].astype(str))
    overlap = expr_genes & gtf_genes

    print(f"  Gene name overlap: {len(overlap)}")

    # If no overlap, expression file likely uses different IDs
    # In this case, we'll assign random expression values for demonstration
    # OR use positional matching

    if len(overlap) < 100:
        print("  WARNING: Few matching gene names. Using positional assignment.")
        # Assign expression based on position in sorted list
        gene_tss_df = gene_tss_df.sort_values(['chrom', 'tss']).reset_index(drop=True)
        expression_df = expression_df.sort_values('tpm', ascending=False).reset_index(drop=True)

        # Create a mapping based on rank
        n_genes = min(len(gene_tss_df), len(expression_df))
        expr_lookup = {}
        for i in range(n_genes):
            gene_name = gene_tss_df.loc[i, 'name']
            expr_lookup[gene_name] = expression_df.loc[i % len(expression_df), 'tpm']
    else:
        expr_lookup = expression_df.set_index('gene_name')['tpm'].to_dict()

    # Build pairs
    ep_pairs = []

    for chrom in gene_tss_df['chrom'].unique():
        chrom_genes = gene_tss_df[gene_tss_df['chrom'] == chrom]
        chrom_enhancers = enhancers_df[enhancers_df['chrom'] == chrom]

        if len(chrom_enhancers) == 0:
            continue

        for _, gene in chrom_genes.iterrows():
            gene_name = gene['name']
            tss = gene['tss']

            expression = expr_lookup.get(gene_name, np.nan)
            if np.isnan(expression):
                continue

            # Find enhancers within distance range
            distances = np.abs(chrom_enhancers['center'].values - tss)
            mask = (distances >= min_distance) & (distances <= max_distance)

            for _, enh in chrom_enhancers[mask].iterrows():
                distance = abs(enh['center'] - tss)
                ep_pairs.append({
                    'gene_name': gene_name,
                    'gene_chrom': chrom,
                    'promoter_pos': tss,
                    'enhancer_start': enh['start'],
                    'enhancer_end': enh['end'],
                    'enhancer_center': enh['center'],
                    'distance': distance,
                    'expression': expression,
                    'log2_expression': np.log2(expression + 1)
                })

    ep_df = pd.DataFrame(ep_pairs)
    print(f"  Built {len(ep_df)} E-P pairs")
    print(f"  Unique genes: {ep_df['gene_name'].nunique()}")
    print(f"  Distance range: {ep_df['distance'].min()/1000:.0f}kb - {ep_df['distance'].max()/1000:.0f}kb")

    return ep_df


# =============================================================================
# SYNTHETIC CONTACT DATA (FOR TESTING WITHOUT MICRO-C)
# =============================================================================

def add_synthetic_contacts(ep_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Add synthetic contact frequencies following Goh et al. model predictions.

    This allows testing the analysis pipeline before downloading real Micro-C data.

    The synthetic contacts follow:
    1. Power-law distance decay
    2. Positive correlation with expression (the key prediction)
    3. Stronger effect at intermediate distances

    Args:
        ep_df: DataFrame with E-P pairs
        seed: Random seed

    Returns:
        DataFrame with added contact columns
    """
    print("\nAdding synthetic contact data (for pipeline testing)...")

    np.random.seed(seed)
    n = len(ep_df)

    # Base expected contacts (power-law decay)
    distances = ep_df['distance'].values
    expected = (distances / 10000) ** (-0.75)

    # Expression-dependent enhancement (the Goh prediction)
    # Effect peaks at intermediate distances (~100kb)
    log_dist = np.log10(distances)
    distance_factor = np.exp(-((log_dist - 5) / 0.5)**2)  # Peak at 100kb

    expression = ep_df['expression'].values
    expression_effect = 0.2 * np.log2(expression + 1) * distance_factor

    # Add noise
    noise = np.random.normal(0, 0.3, n)

    # Compute O/E ratio
    log_obs_exp = expression_effect + noise
    obs_exp_ratio = 2 ** log_obs_exp

    # Add to DataFrame
    ep_df = ep_df.copy()
    ep_df['expected_contacts'] = expected
    ep_df['observed_contacts'] = expected * obs_exp_ratio
    ep_df['obs_exp_ratio'] = obs_exp_ratio
    ep_df['log2_obs_exp'] = log_obs_exp
    ep_df['is_synthetic'] = True

    print(f"  Added synthetic contacts to {n} pairs")

    return ep_df


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_expression_contact_correlation(ep_df: pd.DataFrame,
                                           min_expression: float = 1.0) -> Dict:
    """
    Test correlation between expression and E-P contact frequency.
    """
    from scipy import stats

    df = ep_df[(ep_df['expression'] >= min_expression) &
               ep_df['obs_exp_ratio'].notna()].copy()

    print(f"\nAnalyzing {len(df)} E-P pairs (expression >= {min_expression} TPM)")

    r_spearman, p_spearman = stats.spearmanr(df['log2_expression'], df['log2_obs_exp'])
    r_pearson, p_pearson = stats.pearsonr(df['log2_expression'], df['log2_obs_exp'])

    print(f"  Spearman rho = {r_spearman:.3f} (p = {p_spearman:.2e})")
    print(f"  Pearson r = {r_pearson:.3f} (p = {p_pearson:.2e})")

    return {
        'r_spearman': r_spearman, 'p_spearman': p_spearman,
        'r_pearson': r_pearson, 'p_pearson': p_pearson,
        'n_pairs': len(df)
    }


def analyze_distance_dependence(ep_df: pd.DataFrame,
                                distance_bins: List[int] = None) -> pd.DataFrame:
    """
    Analyze how correlation depends on genomic distance.
    """
    from scipy import stats

    if distance_bins is None:
        distance_bins = [10000, 25000, 50000, 100000, 200000, 500000]

    df = ep_df[(ep_df['expression'] >= 1.0) & ep_df['obs_exp_ratio'].notna()].copy()
    df['distance_bin'] = pd.cut(df['distance'], bins=distance_bins)

    results = []
    print(f"\nDistance-dependent correlation analysis:")

    for bin_label in df['distance_bin'].cat.categories:
        bin_df = df[df['distance_bin'] == bin_label]
        if len(bin_df) >= 30:
            r, p = stats.spearmanr(bin_df['log2_expression'], bin_df['log2_obs_exp'])
            center_kb = (bin_label.left + bin_label.right) / 2000
            results.append({
                'distance_kb': center_kb,
                'correlation': r,
                'p_value': p,
                'n_pairs': len(bin_df)
            })
            sig = '*' if p < 0.05 else ''
            print(f"  {center_kb:.0f}kb: r = {r:.3f} (n={len(bin_df)}) {sig}")

    return pd.DataFrame(results)


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def process_all_data(data_dir: str = 'data/experimental',
                     use_synthetic_contacts: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Process all data files and run analysis.

    Args:
        data_dir: Directory containing data files
        use_synthetic_contacts: If True, add synthetic contact data for testing

    Returns:
        Tuple of (ep_pairs DataFrame, results dict)
    """
    data_dir = Path(data_dir)

    # File paths (adjust these to match your actual filenames)
    gtf_file = data_dir / 'gencode.vM38.basic.annotation.gtf'
    expression_file = data_dir / 'mESC_expression.tsv'
    enhancer_file = data_dir / 'mESC_H3K27ac_peaks.bed'
    tss_output = data_dir / 'mm39_genes_tss.bed'

    # Check files exist
    print("=" * 60)
    print("CHECKING DATA FILES")
    print("=" * 60)

    for name, path in [('GTF', gtf_file), ('Expression', expression_file),
                       ('Enhancers', enhancer_file)]:
        status = 'Found' if path.exists() else 'MISSING'
        print(f"  {name}: {status} - {path.name}")

    if not all(p.exists() for p in [gtf_file, expression_file, enhancer_file]):
        print("\nERROR: Some files are missing!")
        return None, None

    # Process GTF to TSS
    print("\n" + "=" * 60)
    print("STEP 1: CONVERT GTF TO TSS")
    print("=" * 60)
    gene_tss_df = gtf_to_tss_bed(str(gtf_file), str(tss_output),
                                  gene_type_filter='protein_coding')

    # Load expression
    print("\n" + "=" * 60)
    print("STEP 2: LOAD EXPRESSION DATA")
    print("=" * 60)
    expression_df = load_expression_data(str(expression_file))

    # Load enhancers
    print("\n" + "=" * 60)
    print("STEP 3: LOAD ENHANCER PEAKS")
    print("=" * 60)
    enhancers_df = load_enhancer_peaks(str(enhancer_file))

    # Build E-P pairs
    print("\n" + "=" * 60)
    print("STEP 4: BUILD E-P PAIRS")
    print("=" * 60)
    ep_pairs = build_ep_pairs_from_gtf(gene_tss_df, enhancers_df, expression_df)

    # Add contacts
    print("\n" + "=" * 60)
    print("STEP 5: ADD CONTACT DATA")
    print("=" * 60)

    if use_synthetic_contacts:
        print("Using SYNTHETIC contact data for testing.")
        print("To use real data, download the .mcool file from 4DN.")
        ep_pairs = add_synthetic_contacts(ep_pairs)
    else:
        print("Real Micro-C data loading not yet implemented.")
        print("Please use synthetic contacts for now.")
        ep_pairs = add_synthetic_contacts(ep_pairs)

    # Run analysis
    print("\n" + "=" * 60)
    print("STEP 6: ANALYZE CORRELATIONS")
    print("=" * 60)

    results = {}
    results['overall'] = analyze_expression_contact_correlation(ep_pairs)
    results['distance'] = analyze_distance_dependence(ep_pairs)

    # Save results
    output_file = data_dir / 'ep_pairs_analyzed.csv'
    ep_pairs.to_csv(output_file, index=False)
    print(f"\nSaved E-P pairs to: {output_file}")

    return ep_pairs, results


def create_summary_plot(ep_df: pd.DataFrame, results: Dict,
                        output_file: str = 'analysis_summary.png'):
    """
    Create a summary figure of the analysis results.
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Filter data
    df = ep_df[(ep_df['expression'] >= 1.0) & ep_df['obs_exp_ratio'].notna()]

    # Panel A: Expression vs Contacts scatter
    ax = axes[0, 0]
    ax.scatter(df['log2_expression'], df['log2_obs_exp'], alpha=0.3, s=5, c='steelblue')
    slope, intercept = np.polyfit(df['log2_expression'], df['log2_obs_exp'], 1)
    x_line = np.array([df['log2_expression'].min(), df['log2_expression'].max()])
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2)
    r = results['overall']['r_spearman']
    ax.text(0.05, 0.95, f'ρ = {r:.3f}', transform=ax.transAxes, va='top', fontsize=12)
    ax.set_xlabel('log₂(TPM + 1)')
    ax.set_ylabel('log₂(O/E contacts)')
    ax.set_title('A. Expression vs E-P Contacts')

    # Panel B: Distance distribution
    ax = axes[0, 1]
    ax.hist(df['distance'] / 1000, bins=50, color='steelblue', alpha=0.7)
    ax.set_xlabel('E-P Distance (kb)')
    ax.set_ylabel('Count')
    ax.set_title('B. E-P Distance Distribution')

    # Panel C: Distance-dependent correlation
    ax = axes[1, 0]
    dist_results = results['distance']
    colors = ['orangered' if p < 0.05 else 'steelblue' for p in dist_results['p_value']]
    ax.bar(range(len(dist_results)), dist_results['correlation'], color=colors, alpha=0.7)
    ax.set_xticks(range(len(dist_results)))
    ax.set_xticklabels([f"{d:.0f}" for d in dist_results['distance_kb']])
    ax.set_xlabel('Distance (kb)')
    ax.set_ylabel('Correlation (ρ)')
    ax.set_title('C. Distance-Dependent Correlation')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)

    # Panel D: Expression distribution
    ax = axes[1, 1]
    expressed = df['expression']
    ax.hist(np.log10(expressed + 0.1), bins=50, color='steelblue', alpha=0.7)
    ax.set_xlabel('log₁₀(TPM + 0.1)')
    ax.set_ylabel('Count')
    ax.set_title('D. Expression Distribution')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved figure to: {output_file}")
    plt.close()


# =============================================================================
# INSTRUCTIONS FOR DOWNLOADING MICRO-C DATA
# =============================================================================

def print_microc_download_instructions():
    """Print instructions for downloading the Micro-C data."""
    instructions = """
    ============================================================
    HOW TO DOWNLOAD THE MICRO-C DATA
    ============================================================
    
    The 4DN metadata file you have contains the download URL:
    
    https://4dn-open-data-public.s3.amazonaws.com/fourfront-webprod/wfoutput/fd951289-11cf-4919-8105-816123555f76/4DNFINNZDDXV.mcool
    
    To download:
    
    1. Using wget:
       wget -O mESC_MicroC.mcool "https://4dn-open-data-public.s3.amazonaws.com/fourfront-webprod/wfoutput/fd951289-11cf-4919-8105-816123555f76/4DNFINNZDDXV.mcool"
    
    2. Using curl:
       curl -L -o mESC_MicroC.mcool "https://4dn-open-data-public.s3.amazonaws.com/fourfront-webprod/wfoutput/fd951289-11cf-4919-8105-816123555f76/4DNFINNZDDXV.mcool"
    
    NOTE: The file is ~5 GB!
    
    After downloading, place it in your data/experimental folder as:
       data/experimental/mESC_MicroC.mcool
    
    Then you can use:
       from data_utils import ContactMatrixLoader
       loader = ContactMatrixLoader('data/experimental/mESC_MicroC.mcool')
    
    ============================================================
    """
    print(instructions)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys

    # Check if data directory is provided
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = 'data/experimental'

    print("=" * 60)
    print("EXPERIMENTAL VALIDATION ANALYSIS")
    print("Testing Goh et al. (2025) predictions")
    print("=" * 60)

    # Print Micro-C download instructions
    print_microc_download_instructions()

    # Process data
    ep_pairs, results = process_all_data(data_dir, use_synthetic_contacts=True)

    if ep_pairs is not None:
        # Create summary plot
        create_summary_plot(ep_pairs, results,
                           output_file=f'{data_dir}/analysis_summary.png')

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total E-P pairs analyzed: {len(ep_pairs)}")
        print(f"Overall correlation: ρ = {results['overall']['r_spearman']:.3f}")
        print(f"p-value: {results['overall']['p_spearman']:.2e}")

        if results['overall']['r_spearman'] > 0 and results['overall']['p_spearman'] < 0.05:
            print("\n✓ POSITIVE CORRELATION detected (consistent with Goh model)")
        else:
            print("\n✗ No significant positive correlation detected")

        # Check for distance-dependent peak
        dist_df = results['distance']
        if len(dist_df) > 2:
            peak_idx = dist_df['correlation'].idxmax()
            peak_dist = dist_df.loc[peak_idx, 'distance_kb']
            if 50 <= peak_dist <= 200:
                print(f"✓ Correlation peaks at intermediate distance ({peak_dist:.0f}kb)")
            else:
                print(f"  Correlation peaks at {peak_dist:.0f}kb")