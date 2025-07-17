import stim
import time
import numpy as np
import sys
import json

start = time.time()

# repetition code distance
d = int(sys.argv[1])
# number of rounds of X stabilizer measurements
r = int(sys.argv[2])
# same jobs in parallel
job_number = int(sys.argv[3])

# -------------------------------
# Stim circuit 
# -------------------------------

def CreateCircuit(d,r):

    circuit = stim.Circuit()

    for i in range(r):
        
        # preparation GHZ
        circuit.append('RX',int((d-3)/2))
        circuit.append('R',int((d-3)/2)+1)
        
        # preparation data
        if i == 0:
            circuit.append('RX',range(d,3*d-1,2))
            circuit.append('RX',range(3*d-1,5*d-2,2))
            circuit.append('RX',range(5*d-2,7*d-3,2))

        # preparation ancilla
        circuit.append('RX',range(d+1,3*d-1,2))
        circuit.append('RX',range(3*d,5*d-2,2))
        circuit.append('RX',range(5*d-1,7*d-3,2))

        circuit.append('TICK')

        # preparation GHZ second step
        if d!=3:
            circuit.append("R",int((d-3)/2)-1)
        circuit.append('CX',[int((d-3)/2),int((d-3)/2)+1])
        circuit.append('R',int((d-3)/2)+2)
        
        # first CNOT stabilizer
        for j in range(d-1):
            circuit.append('CX',[d+1+2*j,d+1+2*j-1])
            circuit.append('CX',[3*d+2*j,3*d+2*j-1])
            circuit.append('CX',[5*d-1+2*j,5*d-1+2*j-1])
            
        circuit.append('TICK')

        # preparation GHZ main loop
        for j in range(int((d-3)/2)-1):
            circuit.append('R',int((d-3)/2)-j-2)
            circuit.append('CX',[int((d-3)/2)+j+1,int((d-3)/2)+j+2])
            circuit.append('CX',[int((d-3)/2)-j,int((d-3)/2)-j-1])
            circuit.append('R',int((d-3)/2)+j+3)
            if j == 0:
                # second cnot stabilizer
                for j in range(d-1):
                    circuit.append('CX',[d+1+2*j,d+1+2*j+1])
                    circuit.append('CX',[3*d+2*j,3*d+2*j+1])
                    circuit.append('CX',[5*d-1+2*j,5*d-1+2*j+1])
            if j == 1:
                # measurement ancilla
                circuit.append('MX',range(d+1,3*d-1,2))
                circuit.append('MX',range(3*d,5*d-2,2))
                circuit.append('MX',range(5*d-1,7*d-3,2))
                # detector
                for j in range(d-1):
                    circuit.append('DETECTOR',stim.target_rec(-j-1))
                    circuit.append('DETECTOR',stim.target_rec(-(d-1)-j-1))
                    circuit.append('DETECTOR',stim.target_rec(-2*(d-1)-j-1))
                    
            circuit.append('TICK')

        # preparation GHZ penultimate step
        if d>3:
            circuit.append('CX',[d-3,d-2])
            circuit.append('CX',[1,0])
            circuit.append('R',d-1)
            if d==5:
                # second cnot stabilizer
                for j in range(d-1):
                    circuit.append('CX',[d+1+2*j,d+1+2*j+1])
                    circuit.append('CX',[3*d+2*j,3*d+2*j+1])
                    circuit.append('CX',[5*d-1+2*j,5*d-1+2*j+1])
            if d==7:
                # measurement ancilla
                circuit.append('MX',range(d+1,3*d-1,2))
                circuit.append('MX',range(3*d,5*d-2,2))
                circuit.append('MX',range(5*d-1,7*d-3,2))
                # detector
                for j in range(d-1):
                    circuit.append('DETECTOR',stim.target_rec(-j-1))
                    circuit.append('DETECTOR',stim.target_rec(-(d-1)-j-1))
                    circuit.append('DETECTOR',stim.target_rec(-2*(d-1)-j-1))
            circuit.append('TICK')
        
        # preparation GHZ last step
        circuit.append('CX',[d-2,d-1])
        circuit.append('CX',[0,d])
        if d==3:
            # second cnot stabilizer
            for j in range(d-1):
                circuit.append('CX',[d+1+2*j,d+1+2*j+1])
                circuit.append('CX',[3*d+2*j,3*d+2*j+1])
                circuit.append('CX',[5*d-1+2*j,5*d-1+2*j+1])
        if d==5:
            # measurement ancilla
            circuit.append('MX',range(d+1,3*d-1,2))
            circuit.append('MX',range(3*d,5*d-2,2))
            circuit.append('MX',range(5*d-1,7*d-3,2))
            # detector
            for j in range(d-1):
                circuit.append('DETECTOR',stim.target_rec(-j-1))
                circuit.append('DETECTOR',stim.target_rec(-(d-1)-j-1))
                circuit.append('DETECTOR',stim.target_rec(-2*(d-1)-j-1))
        circuit.append('TICK')
        
        # transversal CNOT (in place of Toffoli)
        for j in range(d):
            circuit.append('CX',[j,5*d-2+2*j])
            circuit.append('CX',[3*d-1+2*j,5*d-2+2*j])
            
        if d==3:
            # measurement ancilla
            circuit.append('MX',range(d+1,3*d-1,2))
            circuit.append('MX',range(3*d,5*d-2,2))
            circuit.append('MX',range(5*d-1,7*d-3,2))
            # detector
            for j in range(d-1):
                circuit.append('DETECTOR',stim.target_rec(-j-1))
                circuit.append('DETECTOR',stim.target_rec(-(d-1)-j-1))
                circuit.append('DETECTOR',stim.target_rec(-2*(d-1)-j-1))
            
        circuit.append('TICK')

        # measurement GHZ
        circuit.append('MX',range(d))
        # detector
        if i>0:
            circuit.append("DETECTOR",[stim.target_rec(-j-1) for j in range(d)] + [stim.target_rec(-d-3*(d-1)-j-1) for j in range(d)])
        # GHZ observable
        if i == 0:
            circuit.append("OBSERVABLE_INCLUDE",[stim.target_rec(-j-1) for j in range(d)],0)
        
        # measurement data
        if i==r-1:
            circuit.append('MX',range(d,3*d-1,2))
            circuit.append('MX',range(3*d-1,5*d-2,2))
            circuit.append('MX',range(5*d-2,7*d-3,2))
            # observables
            for j in range(d):
                circuit.append("OBSERVABLE_INCLUDE",[stim.target_rec(-3*d+j)],1+j)
            for j in range(d):
                circuit.append("OBSERVABLE_INCLUDE",[stim.target_rec(-2*d+j)],d+1+j)
            for j in range(d):
                circuit.append("OBSERVABLE_INCLUDE",[stim.target_rec(-d+j)],2*d+1+j)

        circuit.append('TICK')
        
    return circuit

# -------------------------------
# add noise
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
            # determine if Toffoli gate
            qubits = [qubit.value for qubit in instruction.targets_copy()]
            is_Toffoli = len(qubits) != len(set(qubits))
            
            # CNOT noise
            if not is_Toffoli:
                result.append("PAULI_CHANNEL_2",instruction.targets_copy(),[0,0,p/3,0,0,0,0,0,0,0,0,p/3,0,0,p/3])
            # Toffoli noise
            else:
                for i in range(0,len(qubits),4):
                    result.append("Z_ERROR",qubits[i],p/7)
                    result.append("Z_ERROR",qubits[i+2],p/7)
                    result.append("Z_ERROR",qubits[i+3],p/7)
                    result.append("CORRELATED_ERROR",[stim.target_z(qubits[i]), stim.target_z(qubits[i+2])],p/7)
                    result.append("CORRELATED_ERROR",[stim.target_z(qubits[i]), stim.target_z(qubits[i+3])],p/7)
                    result.append("CORRELATED_ERROR",[stim.target_z(qubits[i+2]), stim.target_z(qubits[i+3])],p/7)
                    result.append("CORRELATED_ERROR",[stim.target_z(qubits[i]), stim.target_z(qubits[i+2]), stim.target_z(qubits[i+3])],p/7)
                
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

circuit = AddNoise(CreateCircuit(d,r),5e-3)

# sample circuit
sampler = circuit.compile_detector_sampler()
batch_size = 10_000_000
detection_events, observables = sampler.sample(batch_size, separate_observables=True)
                           
# reject if detection event
accepted = np.all(~detection_events,axis = 1)
# fail if GHZ observable fail or more than d/2 errors 
fail = observables[:, 0] | np.any([np.sum(observables[:, i*d+1:(i+1)*d+1], axis=1) > d/2 for i in range(3)],axis=0)

# -------------------------------
# save results
# -------------------------------

result_dict = {}

result_dict['batch_size'] = batch_size
result_dict['accepted'] = int(np.sum(accepted))
result_dict['fail'] = int(np.sum(fail))
result_dict['accepted&fail'] = int(np.sum(fail&accepted))
                
# save in json
with open(f"out/result_{d}_{r}_{job_number}.json", "w") as file:
    json.dump(result_dict, file)

end = time.time()
print(f"Execution time: {end - start} seconds")


