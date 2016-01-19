#!/bin/bash
#SBATCH -N 1
#SBATCH -t 119:59:00 
#SBATCH -p fat                                                                                                                                                                              

cd /home/edwin/github/edwinkost/PCR-GLOBWB/model
python parallelPCR-GLOBWB_without_prefactors.py ../config/setup_05min_pcrglobwb_only_cartesius_parallel_natural_kinematic_wave.ini no_debug

# pcrglobwb only (natural, kinematic wave, without modflow) - start
