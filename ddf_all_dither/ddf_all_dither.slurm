#!/bin/bash

#SBATCH --account=rubin:developers      # Account name
#SBATCH --partition=milano
#
#SBATCH --job-name=ddf_all_dither5.0
#SBATCH --output=output-%j.txt
#SBATCH --error=output-%j.txt
#
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20g
#
#SBATCH --time=1-14:10:00
#SBATCH --chdir=/sdf/data/rubin/shared/fbs_sims/sims_featureScheduler_runs5.0/ddf_all_dither
#SBATCH --mail-user=yoachim@uw.edu
#
##turn on e-mail notification
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yoachim@uw.edu

## Set up the evironment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate rubin12
export OPENBLAS_NUM_THREADS=1

# cat ddf_all_dither.sh | parallel -j 5

rm maf.sh
 
ls *10yrs.db | xargs -I'{}' echo "scimaf_dir --db '{}'" > maf.sh
ls *10yrs.db | xargs -I'{}' echo "glance_dir --db '{}'" >> maf.sh
ls *10yrs.db | xargs -I'{}' echo "ddf_dir --db '{}'" >> maf.sh
ls *10yrs.db | xargs -I'{}' echo "metadata_dir --db '{}'" >> maf.sh

generate_ss
cat ss_script.sh >> maf.sh
 
cat maf.sh | parallel -j 10
