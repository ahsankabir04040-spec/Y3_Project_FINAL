#!/usr/bin/env python3
"""
Download the full GENCODE vM38 GTF and rebuild the TSS file.
Run this on your local machine (not in Cowork) to get genome-wide gene annotations.

Usage:
    python download_full_gtf.py

After running, re-run experimental_analysis.ipynb to get genome-wide results.
"""
import subprocess
import gzip
import os
from pathlib import Path

DATA_DIR = Path(__file__).parent / 'data' / 'experimental'

# Download full GENCODE vM38 GTF
GTF_URL = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M38/gencode.vM38.basic.annotation.gtf.gz"
GTF_GZ = DATA_DIR / "gencode.vM38.basic.annotation.gtf.gz"
GTF_FILE = DATA_DIR / "gencode.vM38.basic.annotation.gtf"

print("Downloading GENCODE vM38 basic annotation GTF...")
print(f"URL: {GTF_URL}")
subprocess.run(["wget", "-O", str(GTF_GZ), GTF_URL], check=True)

print("Decompressing...")
with gzip.open(GTF_GZ, 'rt') as gz, open(GTF_FILE, 'w') as out:
    for line in gz:
        out.write(line)

# Count genes
n_genes = 0
n_pc = 0
chroms = set()
with open(GTF_FILE) as f:
    for line in f:
        if line.startswith('#'): continue
        fields = line.strip().split('\t')
        if fields[2] == 'gene':
            n_genes += 1
            chroms.add(fields[0])
            if 'protein_coding' in fields[8]:
                n_pc += 1

print(f"\nFull GTF: {n_genes} genes ({n_pc} protein-coding) across {len(chroms)} chromosomes")

# Build TSS file
print("\nBuilding genome-wide TSS file...")
tss_file = DATA_DIR / "mm39_genes_tss.bed"
mapping_file = DATA_DIR / "gene_id_mapping.tsv"

tss_lines = []
mapping_lines = ["ensembl_id\tgene_name\tchrom\ttss\tstrand\n"]

with open(GTF_FILE) as f:
    for line in f:
        if line.startswith('#'): continue
        fields = line.strip().split('\t')
        if fields[2] != 'gene': continue
        attrs = fields[8]
        if 'protein_coding' not in attrs: continue
        
        gene_id = attrs.split('gene_id "')[1].split('"')[0].split('.')[0]
        gene_name = attrs.split('gene_name "')[1].split('"')[0]
        chrom = fields[0]
        strand = fields[6]
        tss = int(fields[3]) if strand == '+' else int(fields[4])
        
        tss_lines.append(f"{chrom}\t{tss}\t{tss+1}\t{gene_name}\t0\t{strand}\n")
        mapping_lines.append(f"{gene_id}\t{gene_name}\t{chrom}\t{tss}\t{strand}\n")

with open(tss_file, 'w') as f:
    f.writelines(tss_lines)

with open(mapping_file, 'w') as f:
    f.writelines(mapping_lines)

print(f"Saved {len(tss_lines)} protein-coding gene TSS entries to {tss_file.name}")
print(f"Saved gene ID mapping to {mapping_file.name}")

# Clean up
os.remove(GTF_GZ)
print("\nDone! Now re-run experimental_analysis.ipynb for genome-wide results.")
