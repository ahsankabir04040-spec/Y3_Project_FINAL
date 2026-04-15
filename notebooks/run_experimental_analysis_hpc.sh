#!/bin/bash
#PBS -N rna_experimental
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -j oe
#PBS -o /rds/general/user/aak223/home/

# -----------------------------------------------------------------------
# run_experimental_analysis_hpc.sh  --  PBS job script for Imperial CX3
# Runs experimental_analysis.ipynb and saves executed notebook + figures
#
# Submitting (from the notebooks/ directory on the cluster):
#   qsub run_experimental_analysis_hpc.sh
#
# Monitoring:
#   qstat -u $USER
#   tail -f /rds/general/user/aak223/home/rna_experimental.o<JOBID>
#
# Output:
#   Executed notebook:  notebooks/experimental_analysis_executed.ipynb
#   Summary figure:     images/analysis_summary.png
#   E-P pairs CSV:      data/experimental/ep_pairs_analyzed.csv
# -----------------------------------------------------------------------

echo "=============================================="
echo "Job ID  : $PBS_JOBID"
echo "Node    : $(hostname)"
echo "Date    : $(date)"
echo "Workdir : $PBS_O_WORKDIR"
echo "=============================================="

# -----------------------------------------------------------------------
# 1. Move to directory where the job was submitted from
# -----------------------------------------------------------------------
cd "$PBS_O_WORKDIR"

# -----------------------------------------------------------------------
# 2. Load conda and activate environment
# -----------------------------------------------------------------------
module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate rna_paper

# If the environment doesn't exist yet, create it:
# conda create -n rna_paper python=3.10 -y
# conda activate rna_paper
# pip install numpy scipy matplotlib pandas cooler h5py seaborn jupyter tqdm

# -----------------------------------------------------------------------
# 3. Verify key data files are present
# -----------------------------------------------------------------------
PROJ_ROOT="$(dirname "$PBS_O_WORKDIR")"   # one level up from notebooks/
DATA_DIR="$PROJ_ROOT/data/experimental"
GTF_FILE="$PROJ_ROOT/gencode.vM38.annotation.gtf"

echo ""
echo "Checking data files..."
for f in \
    "$DATA_DIR/mESC_MicroC.mcool" \
    "$DATA_DIR/mESC_expression.tsv" \
    "$DATA_DIR/mESC_H3K27ac_peaks.bed" \
    "$GTF_FILE"
do
    if [ -f "$f" ]; then
        SIZE=$(du -sh "$f" | cut -f1)
        echo "  FOUND  $f  ($SIZE)"
    else
        echo "  MISSING  $f"
    fi
done

# -----------------------------------------------------------------------
# 4. Run the notebook
# -----------------------------------------------------------------------
echo ""
echo "Running experimental_analysis.ipynb ..."
echo ""

python -m jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=10800 \
    --output experimental_analysis_executed.ipynb \
    experimental_analysis.ipynb

EXIT_CODE=$?

# -----------------------------------------------------------------------
# 5. Copy outputs to home directory for easy access
# -----------------------------------------------------------------------
RESULTS_DIR="/rds/general/user/aak223/home/rna_experimental_results"
mkdir -p "$RESULTS_DIR"

cp -v experimental_analysis_executed.ipynb "$RESULTS_DIR/" 2>/dev/null
cp -v "$PROJ_ROOT/images/analysis_summary.png" "$RESULTS_DIR/" 2>/dev/null
cp -v "$DATA_DIR/ep_pairs_analyzed.csv"        "$RESULTS_DIR/" 2>/dev/null

echo ""
echo "=============================================="
echo "Finished : $(date)"
echo "Exit code: $EXIT_CODE"
echo "Results saved to: $RESULTS_DIR"
echo "=============================================="

exit $EXIT_CODE
