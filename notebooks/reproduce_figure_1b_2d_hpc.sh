#!/bin/bash
#PBS -N rna_fig1b_2d
#PBS -l walltime=08:00:00
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -j oe
#PBS -o /rds/general/user/aak223/home/

# -----------------------------------------------------------------------
# reproduce_figure_1b_2d_hpc.sh  —  PBS job script for Imperial CX3
# -----------------------------------------------------------------------
# Submitting (from the notebooks/ directory on the cluster):
#   qsub reproduce_figure_1b_2d_hpc.sh
#
# Monitoring:
#   qstat -u $USER
#   tail -f ../data/hpc_results_2d/rna_fig1b_2d.o<JOBID>
#
# Adjusting grid size: edit GRID_NX / GRID_NY below
# -----------------------------------------------------------------------

echo "=============================================="
echo "Job ID  : $PBS_JOBID"
echo "Node    : $(hostname)"
echo "Date    : $(date)"
echo "Workdir : $PBS_O_WORKDIR"
echo "=============================================="

# -----------------------------------------------------------------------
# 1. Move to the directory where you submitted the job from
# -----------------------------------------------------------------------
cd "$PBS_O_WORKDIR"

# -----------------------------------------------------------------------
# 2. Load conda and activate the environment
# -----------------------------------------------------------------------
module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate rna_paper

# -----------------------------------------------------------------------
# 3. Settings
# -----------------------------------------------------------------------
GRID_NX=200
GRID_NY=200
N_WORKERS=4
OUTPUT_DIR="/rds/general/user/aak223/home/rna_results"  # results saved here

mkdir -p "$OUTPUT_DIR"

echo "Grid    : ${GRID_NX} x ${GRID_NY}"
echo "Workers : ${N_WORKERS}"
echo "Output  : ${OUTPUT_DIR}"
echo ""

# -----------------------------------------------------------------------
# 4. Run the simulation
# -----------------------------------------------------------------------
python reproduce_figure_1b_2d_hpc.py \
    --nx        "$GRID_NX"    \
    --ny        "$GRID_NY"    \
    --n_workers "$N_WORKERS"  \
    --output_dir "$OUTPUT_DIR"

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Finished : $(date)"
echo "Exit code: $EXIT_CODE"
echo "=============================================="

exit $EXIT_CODE
