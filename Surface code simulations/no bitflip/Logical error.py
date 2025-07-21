import json
import numpy as np
from stimbposd import BPOSD
import copy
import stim
import sys
import time
from collections import Counter

start = time.time()

# number of rounds of X stabilizer measurement
rounds = int(sys.argv[1])
# same jobs launched in parallel
job_number = int(sys.argv[2])

# -------------------------------
# Load code 
# -------------------------------

dict_load = json.load(open('../CNOT order/UnfoldedCode.json', 'r'))

# stabilizers of unfolded code
stabilizers = dict_load['stabilzers']
# stabilizer types (X or Z)
stabilizer_type = np.array(dict_load['stabilizer_type'])
# qubits hosting the 15 XL rotations
rotations = np.array(dict_load['rotations'])
# information qubits where XL = X
info_qubits = dict_load['info_qubits']
# repetition code outputting the magic state
out_qubits = dict_load['out_qubits']
# the 5 logical qubits
Logical_qubits = dict_load['logical_qubits']

# distance against bit-flip errors
dx = dict_load['dx']
# distance against phase-flip errors
dz = dict_load['dz']

# number of ancilla qubits
nb_ancilla = len(stabilizers)

# stabilizers list (without timestep formatting)
stabilizers_flatten = []

for i in range(len(stabilizers)):
    stabilizers_flatten.append([])
    for j in range(len(stabilizers[i])): 
        for k in range(len(stabilizers[i][j])):
            if type(stabilizers[i][j][k]) == int:
                stabilizers_flatten[i].append(stabilizers[i][j][k])

stabilizers_flatten = [list(set(sublist)) for sublist in stabilizers_flatten]
max_len = max(len(sublist) for sublist in stabilizers_flatten)
stabilizers_flatten = np.array([sublist + [np.nan] * (max_len - len(sublist)) for sublist in stabilizers_flatten])

# number of data qubits
nb_data = int(np.nanmax(stabilizers_flatten)) + 1

# stabilizers involved in the surface code only
out_stabilizers = np.where(np.all(np.isin(stabilizers_flatten,out_qubits) | np.isnan(stabilizers_flatten),axis = 1))[0]

# Z stabilizer of the surface code that should not be measure 
# because of lattice surgery
merged_Z_stab = np.argmax(stabilizer_type == 'Z')

# -------------------------------
# Initial projection byproducts
# -------------------------------

# As first round of X stabilizer is random
# tracks the byproduct for each qubit
byproducts = [[] for _ in range(nb_data)]

# iteratively computes the byproducts
processed_qubits = np.zeros(nb_data,dtype = int)
# information qubits are chosen to have no byproduct
processed_qubits[info_qubits] = 1

while np.any(processed_qubits == 0):
            
    # iterate through all stabilizers
    for shape_index, shape in enumerate(stabilizers_flatten):
        
        if stabilizer_type[shape_index] == 'X':

            shape = shape[~np.isnan(shape)].astype(int)
            shape_processed = processed_qubits[shape]

            # if only one qubit in the stabilizer has no byproduct yet
            if np.sum(shape_processed) == len(shape_processed)-1:

                next_qubit_shape_index = np.where(shape_processed == 0)[0][0]
                next_qubit_data_index = shape[next_qubit_shape_index]

                # take sum%2 of the other qubit byproducts
                for qubit in shape:
                    if qubit != next_qubit_data_index:
                        byproducts[next_qubit_data_index] += byproducts[qubit]

                # add current stabilizer in the byproduct
                byproducts[next_qubit_data_index].append(shape_index)

                processed_qubits[next_qubit_data_index] = 1

# remove duplicate stabilizers
for index,sublist in enumerate(byproducts):
    count = Counter(sublist)
    new_sublist = [num for num, freq in count.items() if freq % 2 == 1]
    byproducts[index] = new_sublist

# -------------------------------
# Gate order
# -------------------------------

def Gate_Order(rounds):
    
    # first round of stabilizer measurements
    stabilizers_first = []

    for i in range(len(stabilizers)):
        stabilizers_first.append([])
        for j in range(len(stabilizers[i])):
            stabilizers_first[i].append(stabilizers[i][j])
            if stabilizers[i][j] == ['M']:
                break

    for i in range(len(stabilizers_first)):
        while len(stabilizers_first[i]) < 6:
            stabilizers_first[i].append([])

    # middle rounds
    stabilizers_middle = []

    for i in range(len(stabilizers)):
        stabilizers_middle.append([])
        for j in range(len(stabilizers[i])):
            stabilizers_middle[i].append(stabilizers[i][j])
            if stabilizers[i][j] == ['M']:
                break

    for i in range(len(stabilizers_middle)):
        stabilizers_middle[i] = stabilizers_middle[i][-6:]

    stabilizers_middle_end = copy.deepcopy(stabilizers_middle)

    for i in range(len(stabilizers_middle)):
        while len(stabilizers_middle[i]) < 6:
            stabilizers_middle[i].append([])

    # last round
    stabilizers_final = []

    for i in range(len(stabilizers)):
        stabilizers_final.append([])
        bool_measure = False
        for j in range(len(stabilizers[i])):
            if bool_measure:
                stabilizers_final[i].append(stabilizers[i][j])
            if stabilizers[i][j] == ['M']:
                bool_measure = True

    stabilizers_final_only = []

    for i in range(len(stabilizers)):
        stabilizers_final_only.append([])
        for j in range(len(stabilizers[i])-1,-1,-1):
            stabilizers_final_only[i].insert(0,stabilizers[i][j])
            if stabilizers[i][j] == ['P']:
                break

    max_length = max(len(stab) for stab in stabilizers_final_only)

    for stab in stabilizers_final_only:
        while len(stab) < max_length:
            stab.insert(0, [])

    # construct gate order from stabilizers first, middle and final
    if rounds == 1:
        gate_order = copy.deepcopy(stabilizers_final_only)
    elif rounds == 2:
        gate_order = stabilizers
    else:
        gate_order = copy.deepcopy(stabilizers_first)

        for i in range(rounds-3):
            for j in range(nb_ancilla):
                gate_order[j] += stabilizers_middle[j]

        for j in range(nb_ancilla):
            gate_order[j] += stabilizers_middle_end[j]

        for j in range(nb_ancilla):
            gate_order[j] += stabilizers_final[j]

        max_step = max([len(gate_order[j]) for j in range(nb_ancilla)])
        for j in range(nb_ancilla):
            while len(gate_order[j]) < max_step:
                gate_order[j].append([])
                
    return gate_order

# -------------------------------
# Stim circuit
# -------------------------------

def CreateCircuit(rounds):
    
    # store measurement number on ancilla qubits
    measurement_number = [[] for _ in range(nb_ancilla + len(stabilizers) - len(out_stabilizers))]
    counter = 0
    
    gate_order = Gate_Order(rounds)
    
    circuit = stim.Circuit()
    
    # preparation data
    circuit.append('R',range(nb_data))
    
    # iterate through timesteps
    for i in range(len(gate_order[0])):   
        
        # list of detectors to add at step i
        detectors_to_add = []
        
        for j in range(nb_ancilla):
            
            # preparation
            if gate_order[j][i] == ['P']:
                # X stabilizers
                if stabilizer_type[j] == "X":
                    circuit.append('RX',nb_data + j)
                    if i+1<len(gate_order[0]) and gate_order[j][i+1] == ['Fup']:
                        circuit.append('R',nb_data + nb_ancilla + j)
                # Z stabilizers
                if stabilizer_type[j] == "Z":
                    if j!= merged_Z_stab:
                        circuit.append('R',nb_data + j)
                    
            # measurement
            if gate_order[j][i] == ['M']:
                # X stabilizers
                if stabilizer_type[j] == "X":
                    circuit.append('MX',nb_data + j)
                    measurement_number[j].append(counter)
                    if len(measurement_number[j])>1:
                        detectors_to_add.append([measurement_number[j][-1], measurement_number[j][-2]])
                    counter += 1
                    # flag qubit measurement
                    if i-1>0 and gate_order[j][i-1] == ['Fdo']:
                        circuit.append('M',nb_data + nb_ancilla + j)
                        measurement_number[nb_ancilla + j].append(counter)
                        detectors_to_add.append([measurement_number[nb_ancilla + j][-1]])
                        counter += 1
                # Z stabilizers
                if stabilizer_type[j] == "Z":
                    if j!= merged_Z_stab:
                        circuit.append('M',nb_data + j)
                        measurement_number[j].append(counter)
                        # store detector
                        if len(measurement_number[j])>1:
                            detectors_to_add.append([measurement_number[j][-1], measurement_number[j][-2]])
                        else:
                            detectors_to_add.append([measurement_number[j][-1]])
                        counter += 1
                    
            # flag qubit
            if gate_order[j][i] == ['Fup'] or gate_order[j][i] == ['Fdo']:
                circuit.append('CX',[nb_data + j, nb_data + nb_ancilla + j])
                
            # CNOT gate
            if len(gate_order[j][i])>=1 and type(gate_order[j][i][0]) == int:
                # single CNOT gate
                if len(gate_order[j][i])==1:
                    if stabilizer_type[j] == 'X':
                        circuit.append('CX',[nb_data + j,gate_order[j][i][0]])
                    if stabilizer_type[j] == 'Z':
                        if j!= merged_Z_stab:
                            circuit.append('CX',[gate_order[j][i][0],nb_data + j])
                # two CNOT gates with flag qubit
                if len(gate_order[j][i])==2:
                    circuit.append('CX',[nb_data + j,gate_order[j][i][0]])
                    circuit.append('CX',[nb_data + nb_ancilla + j,gate_order[j][i][1]])
                # flag qubit preparation
                if i+1<len(gate_order[0]) and gate_order[j][i+1] == ['Fup']:
                    circuit.append('R',nb_data + nb_ancilla + j)
                # flag qubit measurement
                if i-1>0 and gate_order[j][i-1] == ['Fdo']:
                    circuit.append('M',nb_data + nb_ancilla + j)
                    measurement_number[nb_ancilla + j].append(counter)
                    detectors_to_add.append([measurement_number[nb_ancilla + j][-1]])
                    counter += 1

            # flag operation during idle ancilla time
            if len(gate_order[j][i])==0:
                # flag qubit preparation
                if i+1<len(gate_order[0]) and gate_order[j][i+1] == ['Fup']:
                    circuit.append('R',nb_data + nb_ancilla + j)
                # flag qubit measurement
                if i-1>0 and gate_order[j][i-1] == ['Fdo']:
                    circuit.append('M',nb_data + nb_ancilla + j)
                    measurement_number[nb_ancilla + j].append(counter)
                    detectors_to_add.append([measurement_number[nb_ancilla + j][-1]])
                    counter += 1
                       
        ## add detectors
        # total number of measurement up to step i
        tot_meas = sum(len(sublist) for sublist in measurement_number)
        for detector in detectors_to_add:
            if len(detector) ==1:
                circuit.append('DETECTOR',[stim.target_rec(-tot_meas+detector[0])])
            if len(detector) ==2:
                circuit.append('DETECTOR',[stim.target_rec(-tot_meas+detector[0]),stim.target_rec(-tot_meas+detector[1])])

        if i==len(gate_order[0])-3:
            circuit.append('I',out_qubits[0])
                
        circuit.append('TICK')

    # total number of ancilla measurement
    tot_meas = sum(len(sublist) for sublist in measurement_number)
        
    # Transversal X^1/2 gate 
    circuit.append('SQRT_X',rotations)
            
    circuit.append('TICK')
        
    # Z measurements unfolded code
    circuit.append('M',[i for i in range(nb_data) if i not in out_qubits])
    
    # distillation check of logical qubit 1 and 4
    for i in [1,3]:
        # raw observable
        raw_observable = [stim.target_rec(-nb_data+len(out_qubits)+j) for j in Logical_qubits[i]]
                  
        ## normalize with byproducts
        updated_logical_qubits = np.intersect1d(rotations,np.array(Logical_qubits[i]))   
        Logical_Qubit_byproduct = [stab for j in updated_logical_qubits for stab in byproducts[j]]   
        # keep only odd counter of stabilizers
        count = Counter(Logical_Qubit_byproduct)
        Logical_Qubit_byproduct = [num for num, freq in count.items() if freq % 2 == 1]
        # first round of random X stabilizers byproducts
        first_round_byproduct = [stim.target_rec(-nb_data+len(out_qubits)-tot_meas+measurement_number[j][0]) for j in Logical_Qubit_byproduct]
            
        # distillation observable
        circuit.append('OBSERVABLE_INCLUDE',raw_observable + first_round_byproduct,i)
        
    circuit.append('TICK')
    
    ## perfect round surface code stabilizer
    # preparation ancilla surface code
    for j in out_stabilizers:
        if stabilizer_type[j] == "X":
            circuit.append('RX',nb_data + j)
        if stabilizer_type[j] == "Z":
            circuit.append('R',nb_data + j)
    
    # CX gates
    for k in range(4):
        for j in out_stabilizers:
            if len(stabilizers[j][1+k])>0 and type(stabilizers[j][1+k][0]) == int:
                if stabilizer_type[j] == "X":
                    circuit.append('CX',[nb_data + j,stabilizers[j][1+k][0]])
                if stabilizer_type[j] == "Z":
                    circuit.append('CX',[stabilizers[j][1+k][0],nb_data + j])
        circuit.append('TICK')
    
    # measurement ancilla
    for j in out_stabilizers:
        if stabilizer_type[j] == "X":
            circuit.append('MX',nb_data + j)
        if stabilizer_type[j] == "Z":
            circuit.append('M',nb_data + j)
            
    # detector
    for index,j in enumerate(out_stabilizers):
        if j != merged_Z_stab:
            circuit.append('DETECTOR',[stim.target_rec(-len(out_stabilizers) + index),
                                       stim.target_rec(-len(out_stabilizers) -nb_data +len(out_qubits) -tot_meas + measurement_number[j][-1])])
            
    # distillation check of logical qubit 0 and 2
    for i in [0,2]:
        logical_qubit_unfolded_code = [qubit for qubit in Logical_qubits[i] if qubit < nb_data-len(out_qubits)]

        # raw observable
        raw_observable = [stim.target_rec(-nb_data+len(out_qubits)-len(out_stabilizers)+j) for j in logical_qubit_unfolded_code]

        # first stabilizer round normalization    
        updated_logical_qubits = np.intersect1d(rotations,logical_qubit_unfolded_code)
        Logical_Qubit_byproduct = [stab for j in updated_logical_qubits for stab in byproducts[j]]   
        # remove even occurrences
        count = Counter(Logical_Qubit_byproduct)
        Logical_Qubit_byproduct = [num for num, freq in count.items() if freq % 2 == 1]

        # create observable normalization  
        first_round_byproduct = [stim.target_rec(-nb_data+len(out_qubits)-len(out_stabilizers)-tot_meas+measurement_number[j][0]) for j in Logical_Qubit_byproduct]

        # Z stabilizer measured again at the end
        stab_Z = [stim.target_rec(-len(out_stabilizers)+(int(dx/2)+1)*(dz-1))]

        circuit.append('OBSERVABLE_INCLUDE',raw_observable + first_round_byproduct + stab_Z,i)

    # Y measurement surface code
    circuit.append('MY',[nb_data - len(out_qubits) + i for i in range(dx*dz)])
    
    ## Y observable
    # raw observable
    raw_observable = [stim.target_rec(-dx*dz+i) for i in range(dx*dz)]
    
    # Z feedback from unfolded code
    feedbackZ_qubits = np.setdiff1d(Logical_qubits[4],out_qubits)
    feedbackZ_observable = [stim.target_rec(-nb_data -len(out_stabilizers) + qubit) for qubit in feedbackZ_qubits]
        
    # Z feedback round normalization
    updated_logical_qubits = np.intersect1d(Logical_qubits[4],rotations)
    Logical_Qubit_byproduct = [stab for j in updated_logical_qubits for stab in byproducts[j]]
    # remove even occurrences
    count = Counter(Logical_Qubit_byproduct)
    Logical_Qubit_byproduct = [num for num, freq in count.items() if freq % 2 == 1]
    byproduct_observable = [stim.target_rec(-tot_meas -nb_data -len(out_stabilizers) +measurement_number[j][0]) for j in Logical_Qubit_byproduct]    
    
    # surface code byproduct
    Logical_Qubit_byproduct = [stab for j in out_qubits for stab in byproducts[j]]
    count = Counter(Logical_Qubit_byproduct)
    Logical_Qubit_byproduct = [num for num, freq in count.items() if freq % 2 == 1]
    surface_code_byproduct = [stim.target_rec(-tot_meas -nb_data -len(out_stabilizers) +measurement_number[j][0]) for j in Logical_Qubit_byproduct]    
    
    circuit.append('OBSERVABLE_INCLUDE',raw_observable + byproduct_observable + feedbackZ_observable + surface_code_byproduct,4)

    return circuit, measurement_number

# -------------------------------
# Add noise
# -------------------------------

def AddNoise(circuit,px,pz):

    # circuit with noise to be added
    result = stim.Circuit()

    # qubit acted on by gates in current timestep
    qubit_processed = []
    
    # remove the noise in final round of perfect stabilizer measurement of surface code
    final_round = False

    for instruction_index,instruction in enumerate(circuit):
        
        # ignore final round instruction if in unfolded code
        if len(instruction.targets_copy())>0:
            qv = instruction.targets_copy()[0].qubit_value
            if qv is not None:
                ignore_final_round = qv < out_qubits[0] or nb_data <= qv < nb_data + out_stabilizers[0] or qv >= nb_data + nb_ancilla
        
        # preparation
        if instruction.name == 'RX' or instruction.name == 'R':
            result.append(instruction)
            if final_round == False or ignore_final_round:
                result.append("PAULI_CHANNEL_1",[target.qubit_value for target in instruction.targets_copy()],[px,0,pz])
            qubit_processed += [target.qubit_value for target in instruction.targets_copy()]

        # measurement
        elif instruction.name == 'MX' or instruction.name == 'M':
            if final_round == False or ignore_final_round:
                result.append("PAULI_CHANNEL_1",[target.qubit_value for target in instruction.targets_copy()],[px,0,pz])
            result.append(instruction)
            qubit_processed += [target.qubit_value for target in instruction.targets_copy()]

        # CNOT
        elif instruction.name == 'CX':       
            result.append(instruction)
            if final_round == False or ignore_final_round:
                result.append("PAULI_CHANNEL_2",[target.qubit_value for target in instruction.targets_copy()],
				[px/3,0,pz/3,px/3,px/3,0,0,0,0,0,0,pz/3,0,0,pz/3])
            qubit_processed += [target.qubit_value for target in instruction.targets_copy()]
            
        # X^1/2
        elif instruction.name == 'SQRT_X':
            result.append(instruction)
            result.append("DEPOLARIZE1",[target.qubit_value for target in instruction.targets_copy()],pz)
            qubit_processed += [target.qubit_value for target in instruction.targets_copy()]
            
        # add idle noise on unacted qubits
        elif instruction.name == 'TICK':
            for i in range(circuit.num_qubits):
                if i not in qubit_processed:
                    ignore_final_round = i < out_qubits[0] or nb_data <= i < nb_data + out_stabilizers[0] or i >= nb_data + nb_ancilla
                    if final_round == False or ignore_final_round:
                        result.append("PAULI_CHANNEL_1",i,[px,0,pz])
            result.append(instruction)
            qubit_processed = []
            
        # I instruction signaling final round
        elif instruction.name == 'I':
            result.append(instruction)
            final_round = True
            
        else:
            result.append(instruction)
            qubit_processed += [target.qubit_value for target in instruction.targets_copy()]

    circuit = result
    
    return circuit
       
# -------------------------------
# simulation
# -------------------------------

# build circuit
circuit, detector_order = CreateCircuit(rounds = rounds)
circuit = AddNoise(circuit,px = 1e-20, pz = 1e-3)

# sample circuit
sampler = circuit.compile_detector_sampler()
batch_size = 10_000_000
detection_events, observables = sampler.sample(batch_size, separate_observables=True)

# decode samples
dem = circuit.detector_error_model(approximate_disjoint_errors = True)
decoder = BPOSD(dem)

predicted_observables = decoder.decode_batch(detection_events)

### remove first round from detector order
for index,sublist in enumerate(detector_order):
    if index<len(stabilizer_type) and stabilizer_type[index] == "X":
        removed_element = sublist[0]
        sublist.remove(removed_element)

        # reduce numbers higher by 1
        for i in range(len(detector_order)):
            for j in range(len(detector_order[i])):
                if detector_order[i][j]>removed_element:
                    detector_order[i][j] -= 1

# -------------------------------
# calculate results
# -------------------------------

result_dict = {}

result_dict['batch_size'] = batch_size

accepted = np.all(observables[:,:4] == predicted_observables[:,:4], axis = 1)

flag_index = [item for sublist in detector_order[len(stabilizers):] for item in sublist]
noflag = ~np.any(detection_events[:,flag_index],axis = 1)

fail = (observables[:,4] != predicted_observables[:,4])

# no preselection
result_dict['preselected'] = [batch_size]
result_dict['preselected&noflag'] = [int(np.sum(noflag))]
result_dict['preselected&accepted'] = [int(np.sum(accepted&noflag))]
result_dict['preselected&accepted&noflag'] = [int(np.sum(accepted&noflag))]
result_dict['preselected&accepted&fail'] = [int(np.sum(accepted&fail))]
result_dict['preselected&accepted&noflag&fail'] = [int(np.sum(accepted&noflag&fail))]

# preselection
for preselected_rounds in range(1,rounds):
    last_round_index = np.array(detector_order[:len(stabilizers)-len(out_stabilizers)])[:,-preselected_rounds:].flatten()
    preselect = np.all(detection_events[:,last_round_index] == False,axis = 1)

    result_dict['preselected'].append(int(np.sum(preselect)))
    result_dict['preselected&noflag'].append(int(np.sum(preselect&noflag)))
    result_dict['preselected&accepted'].append(int(np.sum(preselect&accepted&noflag)))
    result_dict['preselected&accepted&noflag'].append(int(np.sum(preselect&accepted&noflag)))
    result_dict['preselected&accepted&fail'].append(int(np.sum(preselect&accepted&fail)))
    result_dict['preselected&accepted&noflag&fail'].append(int(np.sum(preselect&accepted&noflag&fail)))
                
# -------------------------------
# save
# -------------------------------

with open(f"Results/result_{rounds}_{job_number}.json", "w") as file:
    json.dump(result_dict, file)

end = time.time()
print(f"Execution time: {end - start} seconds")


