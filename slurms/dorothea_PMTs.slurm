#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=dorothea_PMTs
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1              # nodes per instance
#SBATCH --ntasks=1             # tasks per instance
#SBATCH --array=1-50           # instance indexes
#SBATCH --cpus-per-task=1
#SBATCH --mem=200000

#SBATCH --error=errors/%x-%A_%a.err

#SBATCH --mail-user=mariandelbarrio@gmail.com
#SBATCH --mail-type=ALL

echo "JOB TO RUN IC's DOROTHEA"

echo "Slurm job id is ${SLURM_JOB_ID}"
echo "Array job id is ${SLURM_ARRAY_JOB_ID}"
echo "Instance index is ${SLURM_ARRAY_TASK_ID}."

cd /scratch/marian/IC;
pwd

ic_activate

cp $ICDIR/config/dorothea.conf $ICDIR/config/dorothea_${SLURM_ARRAY_TASK_ID}.conf

RUN_NAME="20250624_Next100_1Kev_ELoff_Kr"

# Input file
sed -i "s/IN_FILE.hypathia.h5/${RUN_NAME}_${SLURM_ARRAY_TASK_ID}_PMTs.hypathia.h5/g" $ICDIR/config/dorothea_${SLURM_ARRAY_TASK_ID}.conf
# Output file
sed -i "s/OUT_FILE.dorothea.h5/${RUN_NAME}_${SLURM_ARRAY_TASK_ID}_PMTs.dorothea.h5/g" $ICDIR/config/dorothea_${SLURM_ARRAY_TASK_ID}.conf
# Geometry
sed -i "s/next100fibers/next100/g" $ICDIR/config/dorothea_${SLURM_ARRAY_TASK_ID}.conf


# Define config file used (INPUT and OUTPUT files specified here)
CONFIG_FILE="$ICDIR/config/dorothea_${SLURM_ARRAY_TASK_ID}.conf"
cat ${CONFIG_FILE}

# Define output file based on input filename (optional customization)
OUTPUT_DIR="/scratch/marian/ic-course/data"

# Run your Python script
city dorothea ${CONFIG_FILE}

rm $ICDIR/config/dorothea_${SLURM_ARRAY_TASK_ID}.conf


cd ${OUTPUT_DIR}
pwd
ls -lsrth *.dorothea.h5
# mv ${OUTPUT_FILE}  /data/marian/cigar/outputs


