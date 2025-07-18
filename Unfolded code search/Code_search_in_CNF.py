from z3 import *
import numpy as np
import itertools

# -------------------------------
# SAT variables 
# -------------------------------

# number of qubits that should not be used
nb_unused_qubits = 5

# code size (total number of qubits n*n)
n = 5

# X_L rotation associated to each qubit
Rotation = np.empty((n,n,5),dtype = object)
for i in range(n):
    for j in range(n):
        for k in range(5):
            x = z3.Bool('r_'+str(i)+str(j)+str(k))
            Rotation[i,j,k] = x
        
# connection with the 8 qubits around each qubit
Connection = np.empty((n,n,8),dtype = object)
for i in range(n):
    for j in range(n):
        for k in range(8):
            x = z3.Bool('c_'+str(i)+str(j)+str(k))
            Connection[i,j,k] = x
        
# nb qubits n**2
# max int BitVec 2**(M-1)-1
M = int(np.log2(n**2))+2
print('nb qubits ',n**2)
print('max int BitVec ',2**(M-1)-1)

# qubit ordering in the directed acyclic graph
Level = np.empty((n,n), dtype=object)
for i in range(n):
    for j in range(n):
        Level[i,j] = z3.BitVec('l'+str(i)+str(j), M)
        
# -------------------------------
# SAT constraints 
# -------------------------------

s = z3.Goal()

# X_L rotations of the first 5 qubits
for x in range(5):
    for i in range(n):
        for j in range(n):
            for k in range(5):
                if k!= x:
                    s.add(z3.Implies(Level[i, j] == x,Rotation[i,j][k] == False))
                else:
                    s.add(z3.Implies(Level[i, j] == x,Rotation[i,j][k] == True))

# levels are between 0 and n^2-1
for i in range(n):
    for j in range(n):
        s.add(Level[i,j] >= 0)
        s.add(Level[i,j] <= n*n-1)

# levels are different
for i in range(n):
    for j in range(n):
        for k in range(n):
            for l in range(n):
                if (i,j) != (k,l):
                    s.add(Level[i,j] != Level[k,l])

# no connection to the borders
for i in range(n):
    # up
    s.add(Connection[i,0,5] == False)
    s.add(Connection[i,0,6] == False)
    s.add(Connection[i,0,7] == False)
    
    # down
    s.add(Connection[i,n-1,1] == False)
    s.add(Connection[i,n-1,2] == False)
    s.add(Connection[i,n-1,3] == False)
    
for j in range(n):
    # left
    s.add(Connection[0,j,7] == False)
    s.add(Connection[0,j,0] == False)
    s.add(Connection[0,j,1] == False)
    
    # right
    s.add(Connection[n-1,j,3] == False)
    s.add(Connection[n-1,j,4] == False)
    s.add(Connection[n-1,j,5] == False)

# first five level qubits connected to max 4 qubits
# other qubits connected to max 3 qubits
for i in range(n):
    for j in range(n):
        arr = Connection[i,j].flatten()
        s.add(Implies(Level[i, j] <= 4,AtMost(*arr, 4)))
        s.add(Implies(Level[i, j] >= 5,AtMost(*arr, 3)))

# first five qubits no incoming connection
# other qubits max 3 incoming connections
for i in range(n):
    for j in range(n):
        # array of incoming connections
        arr = []
        if i != n-1:
            arr.append(Connection[i+1,j][0])
        if i != n-1 and j != 0:
            arr.append(Connection[i+1,j-1][1])
        if j != 0:
            arr.append(Connection[i,j-1][2])
        if i != 0 and j != 0:
            arr.append(Connection[i-1,j-1][3])
        if i != 0:
            arr.append(Connection[i-1,j][4])
        if i != 0 and j != n-1:
            arr.append(Connection[i-1,j+1][5])
        if j != n-1:
            arr.append(Connection[i,j+1][6])
        if i != n-1 and j != n-1:
            arr.append(Connection[i+1,j+1][7])
              
        arr = np.array(arr)
        s.add(Implies(Level[i, j] <= 4,AtMost(*arr, 0)))
        s.add(Implies(Level[i, j] >= 5,AtMost(*arr, 3)))

# connection only to qubit with level above
for i in range(n):
    for j in range(n):
        if i != 0:
            s.add(Implies(Connection[i,j][0],Level[i-1,j] > Level[i,j]))
        if i != 0 and j!= n-1:
            s.add(Implies(Connection[i,j][1],Level[i-1,j+1] > Level[i,j]))
        if j!= n-1:
            s.add(Implies(Connection[i,j][2],Level[i,j+1] > Level[i,j]))
        if i != n-1 and j != n-1:
            s.add(Implies(Connection[i,j][3],Level[i+1,j+1] > Level[i,j]))
        if i != n-1:
            s.add(Implies(Connection[i,j][4],Level[i+1,j] > Level[i,j]))
        if i != n-1 and j != 0:
            s.add(Implies(Connection[i,j][5],Level[i+1,j-1] > Level[i,j]))
        if j != 0:
            s.add(Implies(Connection[i,j][6],Level[i,j-1] > Level[i,j]))
        if i != 0 and j != 0:
            s.add(Implies(Connection[i,j][7],Level[i-1,j-1] > Level[i,j]))

# propagate XL rotations to qubit with higher levels
for i in range(n):
    for j in range(n):
        for x in range(5):
            # store rotations from incoming connections
            connection_terms = []
            
            if i!= n-1:
                connection_terms.append(And(Connection[i+1, j, 0], Rotation[i+1, j, x]))
            if i!= n-1 and j != 0:
                connection_terms.append(And(Connection[i+1, j-1, 1], Rotation[i+1, j-1, x]))
            if j != 0:
                connection_terms.append(And(Connection[i, j-1, 2], Rotation[i, j-1, x]))
            if j != 0 and i != 0:
                connection_terms.append(And(Connection[i-1, j-1, 3], Rotation[i-1, j-1, x]))
            if i != 0:
                connection_terms.append(And(Connection[i-1, j, 4], Rotation[i-1, j, x]))
            if i != 0 and j != n-1:
                connection_terms.append(And(Connection[i-1, j+1, 5], Rotation[i-1, j+1, x]))
            if j != n-1:
                connection_terms.append(And(Connection[i, j+1, 6], Rotation[i, j+1, x]))
            if i!= n-1 and j != n-1:
                connection_terms.append(And(Connection[i+1, j+1, 7], Rotation[i+1, j+1, x]))
                
            # xor all rotation terms
            while len(connection_terms) > 1:
                term1 = connection_terms.pop(0)
                term2 = connection_terms.pop(0)
                connection_terms.insert(0, Xor(term1, term2))
                            
            if connection_terms:
                s.add(Implies(Level[i,j]>=5,Rotation[i, j, x] == connection_terms[0]))

# incoming connection within a square
# so that stabilizer is in a square
def WithinASquare(arr):
    for i in range(4):
        for j in range(3):
            s.add(Not(And(arr[2*i],arr[(2*i+3+j)%8])))
        
    for i in range(4):
        for j in range(5):
            s.add(Not(And(arr[2*i+1],arr[(2*i+1+2+j)%8])))

for i in range(n):
    for j in range(n):
        arr = []
        if i != n-1:
            arr.append(Connection[i+1,j][0])
        else:
            arr.append(BoolVal(False))
        if i != n-1 and j != 0: 
            arr.append(Connection[i+1,j-1][1])
        else:
            arr.append(BoolVal(False))
        if j != 0:
            arr.append(Connection[i,j-1][2])
        else:
            arr.append(BoolVal(False))
        if i != 0 and j != 0:
            arr.append(Connection[i-1,j-1][3])
        else:
            arr.append(BoolVal(False))
        if i != 0:
            arr.append(Connection[i-1,j][4])
        else:
            arr.append(BoolVal(False))
        if i != 0 and j != n-1:
            arr.append(Connection[i-1,j+1][5])
        else:
            arr.append(BoolVal(False))
        if j != n-1:
            arr.append(Connection[i,j+1][6])
        else:
            arr.append(BoolVal(False))
        if i != n-1 and j != n-1:
            arr.append(Connection[i+1,j+1][7])
        else:
            arr.append(BoolVal(False))
                        
        WithinASquare(arr)

# no crossing diagonal connections
for i in range(n):
    for j in range(n):
        if i != 0:
            s.add(Implies(Connection[i,j,1],Not(Connection[i-1,j,3])))
        if j != n-1:
            s.add(Implies(Connection[i,j,1],Not(Connection[i,j+1,7])))
          
        if j != n-1:
            s.add(Implies(Connection[i,j,3],Not(Connection[i,j+1,5])))
        if i != n-1:
            s.add(Implies(Connection[i,j,3],Not(Connection[i+1,j,1])))
             
        if j != 0:
            s.add(Implies(Connection[i,j,5],Not(Connection[i,j-1,3])))
        if i != n-1:
            s.add(Implies(Connection[i,j,5],Not(Connection[i+1,j,7])))
            
        if i != 0:
            s.add(Implies(Connection[i,j,7],Not(Connection[i-1,j,5])))
        if j != 0:
            s.add(Implies(Connection[i,j,7],Not(Connection[i,j-1,1])))

# forbidden diagonal, vertical and horizontal connections in the same square
for i in range(n):
    for j in range(n):
        # first quadrant
        if i != 0 and j != n-1:
            s.add(Not(And(And(Connection[i-1,j,4],Connection[i,j+1,6]),Connection[i,j,1])))  
            s.add(Not(And(And(Connection[i-1,j,4],Connection[i,j+1,6]),Connection[i-1,j,3])))  
            s.add(Not(And(And(Connection[i-1,j,4],Connection[i,j+1,6]),Connection[i,j+1,7])))
            s.add(Not(And(And(Connection[i-1,j,4],Connection[i,j+1,6]),And(Connection[i-1,j,2],Connection[i,j+1,0]))))

        # second quadrant
        if i != n-1 and j != n-1:
            s.add(Not(And(And(Connection[i,j+1,6],Connection[i+1,j,0]),Connection[i,j,3]))) 
            s.add(Not(And(And(Connection[i,j+1,6],Connection[i+1,j,0]),Connection[i,j+1,5])))  
            s.add(Not(And(And(Connection[i,j+1,6],Connection[i+1,j,0]),Connection[i+1,j,1])))
            s.add(Not(And(And(Connection[i,j+1,6],Connection[i+1,j,0]),And(Connection[i,j+1,4],Connection[i+1,j,2]))))


        # third quadrant
        if i != n-1 and j != 0:
            s.add(Not(And(And(Connection[i+1,j,0],Connection[i,j-1,2]),Connection[i,j,5]))) 
            s.add(Not(And(And(Connection[i+1,j,0],Connection[i,j-1,2]),Connection[i,j-1,3])))  
            s.add(Not(And(And(Connection[i+1,j,0],Connection[i,j-1,2]),Connection[i+1,j,7])))  
            s.add(Not(And(And(Connection[i+1,j,0],Connection[i,j-1,2]),And(Connection[i+1,j,6],Connection[i,j-1,4]))))  

        # forth quadrant
        if i != 0 and j != 0:
            s.add(Not(And(And(Connection[i,j-1,2],Connection[i-1,j,4]),Connection[i,j,7])))  
            s.add(Not(And(And(Connection[i,j-1,2],Connection[i-1,j,4]),Connection[i,j-1,1])))  
            s.add(Not(And(And(Connection[i,j-1,2],Connection[i-1,j,4]),Connection[i-1,j,5])))
            s.add(Not(And(And(Connection[i,j-1,2],Connection[i-1,j,4]),And(Connection[i,j-1,0],Connection[i-1,j,6]))))  

# qubit with X_L_12345
rotations_12345 = BoolVal(False)
for i in range(n):
    for j in range(n):
        rotation_ij = BoolVal(True)
        for x in range(5):
            rotation_ij = And(rotation_ij,Rotation[i,j,x])
        rotations_12345 = Or(rotations_12345,rotation_ij)
    s.add(rotations_12345)
    
# qubit with all possible X_L_ijk
combinations = itertools.combinations(range(5), 3)
for comb in combinations:
    rotations_3 = z3.BoolVal(False)
    for i in range(n):
        for j in range(n):
            rotation_ij = z3.BoolVal(True)
            for k in range(5):
                if k in comb:
                    rotation_ij = z3.And(rotation_ij,Rotation[i,j,k])
                else:
                    rotation_ij = z3.And(rotation_ij,Not(Rotation[i,j,k]))
            rotations_3 = Or(rotations_3,rotation_ij)
    s.add(rotations_3 == True)

# nb_unused_qubits should not be used
Unused_qubits = np.empty((n,n),dtype = object)
for i in range(n):
    for j in range(n):
        Unused_qubits[i,j] = AtMost(*[Rotation[i, j, k] for k in range(5)],0)

s.add(AtLeast(*Unused_qubits.flatten(),nb_unused_qubits))

# no qubit with X_L_ij (to preserve distance of RM)
combinations = itertools.combinations(range(5), 2)
for comb in combinations:
    for i in range(n):
        for j in range(n):
            rotation_ij = z3.BoolVal(True)
            for k in range(5):
                if k in comb:
                    rotation_ij = z3.And(rotation_ij,Rotation[i,j,k])
                else:
                    rotation_ij = z3.And(rotation_ij,Not(Rotation[i,j,k]))
            s.add(Not(rotation_ij))
            
# no qubit with X_L_ijkl (to preserve distance of RM)
combinations = itertools.combinations(range(5), 4)
for comb in combinations:
    for i in range(n):
        for j in range(n):
            rotation_ij = z3.BoolVal(True)
            for k in range(5):
                if k in comb:
                    rotation_ij = z3.And(rotation_ij,Rotation[i,j,k])
                else:
                    rotation_ij = z3.And(rotation_ij,Not(Rotation[i,j,k]))
            s.add(Not(rotation_ij))


# -------------------------------
# convert to CNF 
# -------------------------------
tactic = z3.Then('simplify','propagate-values' ,'reduce-bv-size', 'bit-blast', 'pb2bv', 'tseitin-cnf')
s = tactic(s)[0]
dimacs = s.dimacs()

with open(f"./Code_search.cnf", "w") as output_f:
    output_f.write(dimacs)