import os
import numpy as np
import pandas as pd

# Create independent random market configurations for the surgery signal sweep.

SETS_PER_SIGNAL = 5000
SIGNAL_COUNTS = range(31)

TARGET_SPECIALTIES = {
    "Surgery (Categorical)",
}


SEED = 7
np.random.seed(SEED)

OUTPUT_DIRECTORY = "constants/surgery_sensitivity/"
# check if directory exists, if not create it
if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)

# NOTE: STANDARD CASE IS GAMMA SHAPE 40 SCALE 72/40
# will also test mean 30, shape 10

# gamma distribution parameters
GAMMA_SHAPE_APPLICATIONS = 10
GAMMA_SCALE_APPLICATIONS = 30/GAMMA_SHAPE_APPLICATIONS
GAMMA_SHAPE_INTERVIEWS_PER_POSITION = 8
GAMMA_SCALE_INTERVIEWS_PER_POSITION = 12/8

# percent deviations
P_DEVIATION_PROGRAMS = 0.05
P_DEVIATION_POSITIONS = 0.10
P_DEVIATION_APPLICANTS = 0.20

# https: // www.nrmp.org/wp-content/uploads/2025/05/Main_Match_Results_and_Data_20250529_FINAL.pdf
nrmp_data = pd.read_csv('constants/nrmp_base_data.csv')
nrmp_dict = nrmp_data.to_dict(orient='records')

final_sets = []
for program in nrmp_dict:
    name = program['Program']
    if name not in TARGET_SPECIALTIES:
        continue
    n_programs = program['n_programs']
    n_applicants = program['n_applicants']
    n_positions = program['n_positions']
    for signals in SIGNAL_COUNTS:
        for set_index in range(SETS_PER_SIGNAL):
            simulated_programs = int(
                np.random.uniform(
                    n_programs * (1-P_DEVIATION_PROGRAMS),
                    n_programs * (1+P_DEVIATION_PROGRAMS)))
            simulated_positions = int(
                np.random.uniform(
                    n_positions * (1-P_DEVIATION_POSITIONS),
                    n_positions * (1+P_DEVIATION_POSITIONS)))
            simulated_applicants = int(
                np.random.uniform(
                    n_applicants * (1-P_DEVIATION_APPLICANTS),
                    n_applicants * (1+P_DEVIATION_APPLICANTS)))
            applicant_applications = np.random.gamma(
                GAMMA_SHAPE_APPLICATIONS,
                GAMMA_SCALE_APPLICATIONS,
                size=simulated_applicants).astype(int)
            interviews_per_position = np.random.gamma(
                GAMMA_SHAPE_INTERVIEWS_PER_POSITION,
                GAMMA_SCALE_INTERVIEWS_PER_POSITION,
                size=simulated_programs).astype(int)
            final_sets.append({
                'program': name,
                'signals': signals,
                'n_programs': simulated_programs,
                'n_positions': simulated_positions,
                'n_applicants': simulated_applicants,
                'sim_id': f"{name}_{signals}_{set_index}",
                'applications_list': list(applicant_applications),
                'interviews_per_pos_list': list(interviews_per_position)
            })

# Convert to DataFrame and export
# parquet used so we don't need to convert data types to store lists, and
# easiet to load line by line
df_final = pd.DataFrame(final_sets)
df_final.to_parquet(OUTPUT_DIRECTORY + 'constants.parquet',
                    index=False, engine='pyarrow')
