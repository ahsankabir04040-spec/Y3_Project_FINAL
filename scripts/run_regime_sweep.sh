#!/bin/bash
#PBS -N regime_sweep
#PBS -l walltime=08:00:00
#PBS -l select=1:ncpus=16:mem=48gb
#PBS -j oe
#PBS -o /rds/general/user/aak223/home/

# -----------------------------------------------------------------------
# run_regime_sweep.sh — PBS job script for Imperial CX3
# 273 simulations (21 k_p x 13 c^-) at 64x64, t_final=200
# 32 parallel workers — estimated ~2-3 hours
#
# Submit:
#   cd /rds/general/user/aak223/home
#   qsub run_regime_sweep.sh
# -----------------------------------------------------------------------

echo "=============================================="
echo "Job ID  : $PBS_JOBID"
echo "Node    : $(hostname)"
echo "Date    : $(date)"
echo "=============================================="

cd "$PBS_O_WORKDIR"

module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate rna_paper

python run_regime_sweep.py \
    --mode    hpc          \
    --workers 16           \
    --out     /rds/general/user/aak223/home/sweep_results.npz \
    --png     /rds/general/user/aak223/home/fig_regime_map.png

echo ""
echo "=============================================="
echo "Finished : $(date)"
echo "=============================================="
