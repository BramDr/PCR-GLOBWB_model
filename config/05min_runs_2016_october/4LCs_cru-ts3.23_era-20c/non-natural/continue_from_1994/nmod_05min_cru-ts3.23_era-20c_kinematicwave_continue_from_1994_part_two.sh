#!/bin/bash
#SBATCH -N 1
#SBATCH -t 119:59:00 
#SBATCH -p fat

# load a special version of pcraster
. /home/edwin/bin-special/pcraster-4.1.0-beta-20151027_x86-64_gcc-4/bashrc_special_pcraster_modflow

cd /home/edwinhs/github/edwinkost/PCR-GLOBWB/model
python parallel_pcrglobwb_runner.py ../config/05min_runs_2016_october/4LCs_cru-ts3.23_era-20c/non-natural/continue_from_1994/nmod_05min_cru-ts3.23_era-20c_kinematicwave_continue_from_1994_part_one.sh /home/sutanudjaja/github/edwinkost/PCR-GLOBWB/config/05min_runs_2016_october/4LCs_cru-ts3.23_era-20c/non-natural/continue_from_1994/setup_05min_cru-ts3.23_era-20c_kinematicwave_continue_from_1994_part_one.ini

