from Kinematics import *
import numpy as np
import os
import json
from pathlib import Path
from util_pushover import convert_model_unit, initialize_contact_force, cal_gap_3d_elastic, \
    adjust_ft_c, write_intermediate_model, _update_contp_force_3d, _update_elem_disp_3d, _displace_model_3d, \
    compute_fracture_distance, recalculate_elasticity
import tqdm
import argparse

save_results = True

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", type=str, required=True,
                   help="Input directory containing load_config.json")
parser.add_argument("--model-dir", type=str, required=True,
                   help="Directory containing the model CSV files")
parser.add_argument("--output-dir", type=str, required=True,
                   help="Output directory for pushover results")
parser.add_argument("--load", type=str, default="load_config.json",
                   help="Configuration file of loading protocol")

args = parser.parse_args()

# Set up paths
input_dir = Path(args.input_dir)
model_dir = Path(args.model_dir)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------
# Load input
# --------------------------------------------------------------------
# Read load protocol
with open(input_dir / args.load) as f:
    load_protocol = json.load(f)

# Load model
set_dimension(3)
model = Model()
model.from_csv(model_dir)

# --------------------------------------------------------------------
# Model preprocessing
# --------------------------------------------------------------------
# for p in model.contps.values():
#     p.cont_type.fc = min(p.cont_type.fc,65.5*1e6)
#     p.cont_type.lamda = 0.2
#     if model.elems[p.cand].type.startswith('mortar') and model.elems[p.anta].type.startswith('stone'):
#         #p.cont_type.fc*=1000
#         #p.cont_type.fc = min(p.cont_type.fc,65.6*1e6)
#         p.cont_type.mu = 0.4
#         # p.cont_type.cohesion = p.cont_type.cohesion*3*0.5/5
#         # p.cont_type.ft =p.cont_type.ft*3*0.5/5
#         # p.cont_type.cohesion = 0.039*1e6
#         # p.cont_type.ft = 0.039*1e6
#         p.cont_type.cohesion = 0.05*1e6/3
#         p.cont_type.ft = 0.05*1e6/3
#         p.cont_type.Gf1= 0.02*1e3/3
#         p.cont_type.Gf2 = 0.005*1e3/3
#         p.cont_type.Gc = 5.29*1e3
#     elif model.elems[p.anta].type.startswith('mortar') and model.elems[p.cand].type.startswith('stone'):
#         #p.cont_type.fc*=1000
#         #p.cont_type.fc = min(p.cont_type.fc,65.6*1e6)
#         p.cont_type.mu = 0.4
#         # p.cont_type.mu =1.0
#         # p.cont_type.cohesion = p.cont_type.cohesion*3*0.5/5
#         # p.cont_type.ft = p.cont_type.ft*3*0.5/5
#         p.cont_type.cohesion = 0.05*1e6/3
#         p.cont_type.ft = 0.05*1e6/3
#         p.cont_type.Gf1= 0.02*1e3/3
#         p.cont_type.Gf2 = 0.005*1e3/3
#         p.cont_type.Gc = 5.29*1e3
#     elif model.elems[p.anta].type.startswith('stone') and model.elems[p.cand].type.startswith('stone'):
#         #p.cont_type.fc*=1000
#         p.cont_type.mu = 0
#         p.cont_type.cohesion = 0
#         p.cont_type.ft = 0
#         p.cont_type.Gf1= 0*1e3
#         p.cont_type.Gf2 = 0*1e3
#         p.cont_type.Gc = 27.77*1e3
#     elif model.elems[p.anta].type.startswith('mortar') and model.elems[p.cand].type.startswith('mortar'):
#         p.cont_type.mu = 0.4
#         p.cont_type.cohesion = 0.05*1e6
#         p.cont_type.ft = 0.05*1e6/1
#         p.cont_type.Gf1= 0.02*1e3
#         p.cont_type.Gf2 = 0.005*1e3
#         p.cont_type.Gc = 5.29*1e3
#         # if np.random.rand()<0.2:
#         #     p.section_h = 0.001*p.section_h
#     else:
#         pass
        
# Recalculate elasticity of interface by conjugating elasticity from two contact bodies
#model.contps = recalculate_elasticity(model)
model = compute_fracture_distance(model)

# Convert unit to kN and mm
model = convert_model_unit(model, force_convert=1, length_convert=1e3)  # N, mm
model = convert_model_unit(model, force_convert=1e-3, length_convert=1)  # kN, mm

# Plot the initial model
if save_results:
    viewer = VtkRenderer(model.elems, model.contps)
    viewer.plot_displaced_points(factor=0, filename=str(output_dir / 'initial'))

# Set where to apply axial load
beamID = int(load_protocol["beamID"])  # beam
dl_dir = int(load_protocol['dead_load_dim'])
model.elems[beamID].dl[dl_dir] = float(load_protocol['dead_load_value'])

# Initialize contact forces to zero
initialize_contact_force(model.contps)

# --------------------------------------------------------------------
# Pushover
# --------------------------------------------------------------------
# Parameters configuration
current_step_size = float(load_protocol['step_size'])
max_iteration = int(float(load_protocol['max_disp'])/current_step_size)+1
write_freq = int(load_protocol['write_freq'])
control_dof = int(load_protocol['control_dof'])

# Create result.txt
with open(output_dir / 'force_displacement.txt', 'w+') as f:
    f.write('force, displacement\n')

# Pushover iteration
for i in tqdm.tqdm(range(max_iteration)):
    # Load
    A_matrix = cal_A_global_3d(model.elems, model.contps, sparse=True)
    solution = solve_elastic_finitefc_associative_3d_disp_control(
        model.elems, model.contps,
        Aglobal=A_matrix, thickness_dict=None,
        control_element="beam", control_dof=control_dof,
        control_displacement=current_step_size
    )
    
    # Non-convergence status check
    if solution['convergence'] == False:
        if save_results:
            viewer = VtkRenderer(model.elems, model.contps)
            # viewer.plot_displaced_points_as_springs(
            #     scale=1, 
            #     filename=str(output_dir / f'associative_elastic_iter{i}_springs')
            # )
            viewer.plot_displaced_points(factor=0, filename=str(output_dir / f'associative_elastic_iter{i}_points'))
        break
    
    # Converged result
    live_load_of_this_step = -solution['imposed_force']
    
    # Save results from solution to model
    _update_contp_force_3d(model.contps, solution['contact_forces'])
    _update_elem_disp_3d(model.contps, model.elems, solution["displacements"])
    
    with open(output_dir / 'force_displacement.txt', 'a') as f:
        #f.write(f'0, 0\n')
        f.write(f'{live_load_of_this_step}, {model.elems[beamID].displacement[control_dof]}\n')
    
    if i % write_freq == 0 and save_results:
        write_intermediate_model(model, str(output_dir), i)
        viewer = VtkRenderer(model.elems, model.contps)
        # viewer.plot_displaced_points_as_springs(
        #     scale=1,
        #     filename=str(output_dir / f'associative_elastic_iter{i}_springs')
        # )
        viewer.plot_displaced_points(factor=0, filename=str(output_dir / f'associative_elastic_iter{i}_points'))