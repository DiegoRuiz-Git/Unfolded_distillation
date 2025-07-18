import json
import numpy as np
from stimbposd import BPOSD
import stim
import sys
import time
from collections import Counter

start = time.time()

# number of rounds of X stabilizer measurements
rounds = int(sys.argv[1])
# same jobs launched in parallel
job_number = int(sys.argv[2])

# -------------------------------
# Load code 
# -------------------------------

dict_load = json.load(open('../CNOT order/UnfoldedCode_d=11.json', 'r'))

# stabilizers of unfolded code
stabilizers = np.array(dict_load['stabilzers'])
# qubits hosting the 15 XL rotations
rotations = np.array(dict_load['rotations'])
# information qubits where XL = X
info_qubits = dict_load['info_qubits']
# repetition code outputting the magic state
out_qubits = dict_load['out_qubits']
# the 5 logical qubits
Logical_qubits = dict_load['logical_qubits']

nb_data = int(np.nanmax(stabilizers)+1)
nb_ancilla = len(stabilizers)

# -------------------------------
# CNOT order
# -------------------------------

def Create_Cnot_order(stabilizers,rounds):
    Cnot_order = np.full((stabilizers.shape[0],(stabilizers.shape[1]+2)*rounds + rounds%2*2 + 3),np.nan)
    
    # unfolded code
    for i in range(rounds):
        Cnot_order[:-len(out_qubits)+1,i*6+1:i*6+5] = stabilizers[:-len(out_qubits)+1,:]

    # indicator for SQRT_X
    Cnot_order[0,int(rounds/2)*6-1] = -1
        
    # repetition code
    i = 0
    while 4*i < Cnot_order.shape[1] - 3:
        Cnot_order[-len(out_qubits)+1:,i*4+1:i*4+3] = stabilizers[-len(out_qubits)+1:,:2]
        i+=1
        
    # change output qubit cnot order in case of CNOT collision
    for j in range(Cnot_order.shape[1]):
        if len(np.where(Cnot_order[:,j] == 18)[0]) == 2:
            Cnot_order[-len(out_qubits)+1:,[j,j+1]] = Cnot_order[-len(out_qubits)+1:,[j+1,j]]
    
    return Cnot_order

# -------------------------------
# Initial projection byproducts
# -------------------------------

# As first round of X stabilizer is random
# tracks the byproduct for each qubit
byproducts = [[] for _ in range(nb_data-len(out_qubits)+1)]

# iteratively computes the byproducts
processed_qubits = np.zeros(nb_data-len(out_qubits)+1,dtype = int)
# information qubits are chosen to have no byproduct
processed_qubits[info_qubits] = 1

while np.any(processed_qubits == 0):
        
    # iterate through all stabilizers
    for shape_index, shape in enumerate(stabilizers[:-len(out_qubits)+1]):

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

# remove duplicate stabilizers in byproducts
for index,sublist in enumerate(byproducts):
    count = Counter(sublist)
    new_sublist = [num for num, freq in count.items() if freq % 2 == 1]  
    byproducts[index] = new_sublist

# -------------------------------
# Stim circuit
# -------------------------------

def CreateCircuit(rounds):
    
    # store measurement number on each ancilla qubit
    measurement_number = [[] for _ in range(nb_ancilla)]
    counter = 0

    circuit = stim.Circuit()
    
    Cnot_order = Create_Cnot_order(stabilizers,rounds)
        
    # iterate through timesteps
    for i in range(Cnot_order.shape[1]):
        
        # list of detectors to add at step i
        detectors_to_add = []

        # preparation data
        if i==0:
            circuit.append('R',range(nb_data))

        # gates
        for j in range(nb_ancilla):
            # preparation ancilla
            if i!=Cnot_order.shape[1]-1 and np.isnan(Cnot_order[j][i]) and not np.isnan(Cnot_order[j][i+1]):
                circuit.append('RX',nb_data + j)
            # CNOT
            if not np.isnan(Cnot_order[j][i]) and Cnot_order[j][i] != -1 :
                circuit.append('CX',[nb_data + j,int(Cnot_order[j][i])])
            # X^1/4 gate
            if Cnot_order[j][i] == -1:
                circuit.append('SQRT_X',rotations)
            # measurement
            if i!=0 and (np.isnan(Cnot_order[j][i]) or Cnot_order[j][i] == -1) and not np.isnan(Cnot_order[j][i-1]) and not Cnot_order[j][i-1] == -1:
                circuit.append('MX',[nb_data + j])
                measurement_number[j].append(counter)
                # store detector
                if len(measurement_number[j]) > 1:
                    detectors_to_add.append([measurement_number[j][-1], measurement_number[j][-2]])
                counter += 1
                
        ## add detectors
        # total number of measurement up to step i
        tot_meas = sum(len(sublist) for sublist in measurement_number)
        for detector in detectors_to_add:
            circuit.append('DETECTOR',[stim.target_rec(-tot_meas+detector[0]),stim.target_rec(-tot_meas+detector[1])])
        
        ## Transversal X^1/2 gate 
        # if all gates performed on unfolded code
        if i>=2 and np.all(np.isnan(Cnot_order[0][i-1:])) and not np.all(np.isnan(Cnot_order[0][i-2:])):
            circuit.append('M',[qubit for qubit in range(nb_data) if qubit not in out_qubits])
            
        ## Y measurement repetition code
        # S gates
        if i>=2 and np.all(np.isnan(Cnot_order[-1][i-1:])) and not np.all(np.isnan(Cnot_order[-1][i-2:])):
            circuit.append('S',out_qubits)
            
        # CZ between every pair of qubits
        if i>=3 and np.all(np.isnan(Cnot_order[-1][i-2:])) and not np.all(np.isnan(Cnot_order[-1][i-3:])):
            for j in range(len(out_qubits)):
                for k in range(j+1,len(out_qubits)):
                    circuit.append('CZ',[out_qubits[j],out_qubits[k]])

        # X measurements
        if i>=4 and np.all(np.isnan(Cnot_order[-1][i-3:])) and not np.all(np.isnan(Cnot_order[-1][i-4:])):
            circuit.append('MX',out_qubits)
 
        circuit.append('TICK')

    ## detectors of the output repetition code
    # total number of ancilla measurement
    tot_meas = sum(len(sublist) for sublist in measurement_number)
    for i in range(len(stabilizers)-(len(out_qubits)-1),len(stabilizers)):
        circuit.append('DETECTOR',[stim.target_rec(i-1-len(stabilizers)),stim.target_rec(i-len(stabilizers))] 
                                + [stim.target_rec(-nb_data-tot_meas+measurement_number[i][-1])])
        
    # distillation check of logical qubit 0,1,3,4
    for i in [0,1,3,4]:
        # update numbering because of output qubit not measured
        updated_logical_qubits = np.array(Logical_qubits[i])
        for qubit in out_qubits:
            updated_logical_qubits = np.where(
                updated_logical_qubits > qubit, updated_logical_qubits - 1, updated_logical_qubits
                                            )
            
        # raw observable
        raw_observable = [stim.target_rec(-nb_data+int(j)) for j in updated_logical_qubits]
                
        ## normalize with byproducts
        updated_logical_qubits = np.intersect1d(Logical_qubits[i],rotations)
        Logical_Qubit_byproduct = [stab for j in updated_logical_qubits for stab in byproducts[j]]
        # keep only odd counter of stabilizers
        count = Counter(Logical_Qubit_byproduct)  
        Logical_Qubit_byproduct = [num for num, freq in count.items() if freq % 2 == 1]
        # first round of random X stabilizers
        random_first_stab = np.array(measurement_number[:-len(out_qubits)+1])[:,:1].flatten()
        byproduct_observable = [stim.target_rec(-nb_data-tot_meas+random_first_stab[j]) for j in Logical_Qubit_byproduct]
        
        circuit.append('OBSERVABLE_INCLUDE',raw_observable + byproduct_observable, i if i<=1 else i-1)
            
    ### Y observable out qubit
    # raw observable
    raw_observable = [stim.target_rec(-len(out_qubits))]

    # normalize with byproducts
    updated_logical_qubits = np.intersect1d(Logical_qubits[2],rotations)
    Logical_Qubit_byproduct = [stab for j in updated_logical_qubits for stab in byproducts[j]]
    # keep only odd counter of stabilizers
    count = Counter(Logical_Qubit_byproduct)  
    Logical_Qubit_byproduct = [num for num, freq in count.items() if freq % 2 == 1]
    # first round of random X stabilizers
    byproduct_observable = [stim.target_rec(-nb_data-tot_meas+random_first_stab[j]) for j in Logical_Qubit_byproduct]
    
    ## feedback X from Z measurement unfolded code
    feedbackZ_qubits = np.setdiff1d(Logical_qubits[2],out_qubits)
    # update numbering because of output qubit not measured
    for qubit in out_qubits:
        feedbackZ_qubits = np.where(feedbackZ_qubits > qubit, feedbackZ_qubits - 1, feedbackZ_qubits)    
    feedbackZ_observable = [stim.target_rec(-nb_data + qubit) for qubit in feedbackZ_qubits]
        
    circuit.append('OBSERVABLE_INCLUDE',raw_observable + byproduct_observable + feedbackZ_observable,4)

    return circuit, measurement_number

# -------------------------------
# Add noise
# -------------------------------

def AddNoise(circuit,p):

    # circuit with noise to be added
    result = stim.Circuit()

    # qubit acted on by gates in current timestep
    qubit_processed = []
    
    for instruction_index,instruction in enumerate(circuit):
                
        # preparation
        if instruction.name == 'RX':
            result.append(instruction)
            result.append("Z_ERROR",[target.qubit_value for target in instruction.targets_copy()],p)
            qubit_processed += [target.qubit_value for target in instruction.targets_copy()]

        # measurement
        elif instruction.name == 'MX':
            result.append("Z_ERROR",[target.qubit_value for target in instruction.targets_copy()],p)
            result.append(instruction)
            qubit_processed += [target.qubit_value for target in instruction.targets_copy()]

        # CNOT
        elif instruction.name == 'CX':
            result.append(instruction)
            result.append("PAULI_CHANNEL_2",instruction.targets_copy(),[0,0,p/3,0,0,0,0,0,0,0,0,p/3,0,0,p/3])
            qubit_processed += [target.qubit_value for target in instruction.targets_copy()]
                
        # X^1/2
        elif instruction.name == 'SQRT_X':
            result.append(instruction)
            result.append("DEPOLARIZE1",[target.qubit_value for target in instruction.targets_copy()],p)
            qubit_processed += [target.qubit_value for target in instruction.targets_copy()]

        # add idle noise on unacted qubits
        elif instruction.name == 'TICK':
            for i in range(circuit.num_qubits):
                if i not in qubit_processed:
                    result.append("Z_ERROR",i,p)
            result.append(instruction)
            qubit_processed = []
            
        else:
            result.append(instruction)
            qubit_processed += [target.qubit_value for target in instruction.targets_copy()]

    circuit = result
    
    return circuit 
       
# -------------------------------
# simulation
# -------------------------------

# build circuit
circuit, detector_order = CreateCircuit(rounds)
circuit = AddNoise(circuit,5e-3)

# sample circuit
sampler = circuit.compile_detector_sampler()
batch_size = 1_000_000
detection_events, observables = sampler.sample(batch_size, separate_observables=True)

# decode samples
dem = circuit.detector_error_model(approximate_disjoint_errors = True)
decoder = BPOSD(dem)

# remove obvious BPOSD errors
H = decoder._matrices.check_matrix
priors = decoder._matrices.priors
Obs = decoder._matrices.observables_matrix

identical_pairs = []
H = H.toarray()
for i in range(H.shape[1]):
    for j in range(i + 1, H.shape[1]):
        if np.array_equal(H[:, i], H[:, j]):
            identical_pairs.append([i, j])

predicted_observables = np.zeros((batch_size,Obs.shape[0]))
for i in range(batch_size):
    corr = decoder._bposd.decode(detection_events[i])
    for pair in identical_pairs:
        if np.all(corr[pair] == 1):
            corr[pair] = 0
    predicted_observables[i] = (Obs@corr)%2

# -------------------------------
# calculate results
# -------------------------------

accepted = np.all(observables[:,:4] == predicted_observables[:,:4], axis = 1)
fail = (observables[:,4] != predicted_observables[:,4])

result_dict = {}

result_dict['batch_size'] = batch_size
result_dict['accepted'] = int(np.sum(accepted))
result_dict['fail'] = int(np.sum(fail))
result_dict['accepted&fail'] = int(np.sum(fail&accepted))
                
# -------------------------------
# save
# -------------------------------

with open(f"Results/result_{rounds}_{job_number}.json", "w") as file:
    json.dump(result_dict, file)

end = time.time()
print(f"Execution time: {end - start} seconds")


