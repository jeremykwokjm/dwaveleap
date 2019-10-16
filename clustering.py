import dwavebinarycsp
import dimod
import numpy as np

numElements=16;
variables=['b'+f'{i}'+'_'+f'{j}' for i in range(1, numElements+1) for j in range(1, numElements+1)]
posj=np.arange(1,numElements+1)

mik=np.random.randint(0,high=2,size=[numElements,numElements])
for i in range(1, numElements+1):
    mik[i-1,i-1]=1

def summation(b):
    total=0
    for i in b:
        total=total+i
    return (total==1)

def cspCheck(sample):
    for j in range(1, numElements+1):
        tempVariables=['b'+f'{i}'+'_'+f'{j}' for i in range(1, numElements+1)]
        if summation([sample[x] for x in tempVariables])==False:
            return False
    for i in range(1, numElements+1):
        tempVariables=['b'+f'{i}'+'_'+f'{j}' for j in range(1, numElements+1)]
        if summation([sample[x] for x in tempVariables])==False:
            return False
    return True

csp_m = np.zeros([len(variables),len(variables)])
csp_offset=0
for j in range(1, numElements+1):
    csp_offset+=1
    for i in range(1, numElements+1):
        csp_m[(i-1)*numElements+j-1,(i-1)*numElements+j-1]-=1
        for k in range(i+1, numElements+1):
            csp_m[(i-1)*numElements+j-1,(k-1)*numElements+j-1]+=2

for i in range(1, numElements+1):
    csp_offset+=1
    for j in range(1, numElements+1):
        csp_m[(i-1)*numElements+j-1,(i-1)*numElements+j-1]-=1
        for l in range(j+1, numElements+1):
            csp_m[(i-1)*numElements+j-1,(i-1)*numElements+l-1]+=2
    
objfn = np.zeros([len(variables),len(variables)])
for i in range(1, numElements+1):
    for k in range(1, numElements+1):
        for j in range(1, numElements+1):
            for l in range(1, numElements+1):
                objfn[(i-1)*numElements+j-1,(k-1)*numElements+l-1]=(posj[j-1]-posj[l-1])**2*mik[i-1,k-1]
w=900
A=w*csp_m+objfn

model = dimod.BinaryQuadraticModel.from_numpy_matrix(A, variable_order = variables,offset=w*csp_offset)

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dwave_qbsolv import QBSolv
system = EmbeddingComposite(DWaveSampler(solver='DW_2000Q_5'))
sampler = QBSolv()
response = sampler.sample(bqm=model, num_repeats=1, solver=system)

valid, invalid, data = 0, 0, []
repeat=0
for datum in response.samples(['sample', 'energy', 'num_occurrences']):
    if (cspCheck(datum.sample)):
        for m in range(datum.num_occurrences):
            values=[]
            for i in range(1, numElements+1):
                for j in range(1, numElements+1):
                    values=np.append(values,response.samples.first.sample['b'+f'{i}'+'_'+f'{j}'])
                    
            objvalue=np.matmul(np.matmul(values.reshape(1,-1),objfn),values)
            x=(datum.sample, datum.energy, objvalue[0])
            if x not in data:
                valid = valid+datum.num_occurrences
                data.append(x)
            else: repeat = repeat + datum.num_occurrences
    else:
        invalid = invalid+datum.num_occurrences
print(valid, repeat, invalid)
