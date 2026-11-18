#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=detsim
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1              # nodes per instance
#SBATCH --ntasks=1             # tasks per instance
#SBATCH --array=17-40           # instance indexes
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000000

#SBATCH --error=errors/%x-%A_%a.err

#SBATCH --mail-user=mariandelbarrio@gmail.com
#SBATCH --mail-type=ALL

echo "JOB TO RUN IC's DETSIM"

echo "Slurm job id is ${SLURM_JOB_ID}"
echo "Array job id is ${SLURM_ARRAY_JOB_ID}"
echo "Instance index is ${SLURM_ARRAY_TASK_ID}."

cd /scratch/marian/IC;
pwd

ic_activate

cp $ICDIR/config/detsim_next100.conf $ICDIR/config/detsim_next100_${SLURM_ARRAY_TASK_ID}.conf

RUN_NAME="20250626_Next100_50Kev_ELoff_Kr"

# Input file
sed -i "s/IN_FILE.next.h5/${RUN_NAME}_${SLURM_ARRAY_TASK_ID}.next.h5/g" $ICDIR/config/detsim_next100_${SLURM_ARRAY_TASK_ID}.conf
# Output file
sed -i "s/OUT_FILE.detsim.h5/${RUN_NAME}_${SLURM_ARRAY_TASK_ID}.detsim.h5/g" $ICDIR/config/detsim_next100_${SLURM_ARRAY_TASK_ID}.conf


# Define config file used (INPUT and OUTPUT files specified here)
CONFIG_FILE="$ICDIR/config/detsim_next100_${SLURM_ARRAY_TASK_ID}.conf"
cat ${CONFIG_FILE}

# Define output file based on input filename (optional customization)
OUTPUT_DIR="/scratch/marian/IC/outputs"

# Run your Python script
city detsim ${CONFIG_FILE}

rm $ICDIR/config/detsim_next100_${SLURM_ARRAY_TASK_ID}.conf


cd ${OUTPUT_DIR}
pwd
ls -lsrth *.detsim.h5
# mv ${OUTPUT_FILE}  /data/marian/cigar/outputs


