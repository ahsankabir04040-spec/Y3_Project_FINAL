#!/bin/bash
#PBS -N fig2d_rouse
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=128:mem=128gb
#PBS -j oe
#PBS -o /rds/general/user/aak223/home/

# -----------------------------------------------------------------------
# run_figure2d_hpc.sh  --  PBS job script for Imperial CX3
# Runs the improved Fig 2d cell (3D Rouse contact probability heatmap)
# from reproduce_figure_2.ipynb on 64 cores.
#
# Submitting (from the notebooks/ directory on the cluster):
#   qsub run_figure2d_hpc.sh
#
# Monitoring:
#   qstat -u $USER
#   tail -f /rds/general/user/aak223/home/fig2d_rouse.o<JOBID>
#
# Output (written to images/ in the project root):
#   images/fig2d_contact_heatmap.png
#   images/fig2d_contact_data.npz
# -----------------------------------------------------------------------

echo "=============================================="
echo "Job ID  : $PBS_JOBID"
echo "Node    : $(hostname)"
echo "Date    : $(date)"
echo "Workdir : $PBS_O_WORKDIR"
echo "CPUs    : $(nproc)"
echo "=============================================="

# -----------------------------------------------------------------------
# 1. Move to the directory the job was submitted from
# -----------------------------------------------------------------------
cd "$PBS_O_WORKDIR"

# -----------------------------------------------------------------------
# 2. Load conda and activate environment
# -----------------------------------------------------------------------
module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate rna_paper

# -----------------------------------------------------------------------
# 3. Run the heatmap script directly (avoids running the full notebook
#    which has an unrelated numerical instability in the fig 2e cell)
# -----------------------------------------------------------------------
echo ""
echo "Running fig2d_heatmap.py ..."
echo ""

python fig2d_heatmap.py

EXIT_CODE=$?

# -----------------------------------------------------------------------
# 4. Copy outputs back to home directory for easy retrieval
# -----------------------------------------------------------------------
PROJ_ROOT="$(dirname "$PBS_O_WORKDIR")"   # one level up from notebooks/
RESULTS_DIR="/rds/general/user/aak223/home/fig2d_results"
mkdir -p "$RESULTS_DIR"

cp -v "$PROJ_ROOT/images/fig2d_contact_heatmap.png" "$RESULTS_DIR/" 2>/dev/null
cp -v "$PROJ_ROOT/images/fig2d_contact_data.npz"    "$RESULTS_DIR/" 2>/dev/null

echo ""
echo "=============================================="
echo "Finished : $(date)"
echo "Exit code: $EXIT_CODE"
echo "Results in: $RESULTS_DIR"
echo "=============================================="

exit $EXIT_CODE
