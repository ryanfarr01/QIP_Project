import numpy as np
import qiskit as qs
import qiskit.circuit.library as ql

#https://arxiv.org/abs/2311.08555
def add_k_fourier(k,mod):
    blm = mod.bit_length() #Bit Len Mod 
    add_circuit = qs.QuantumCircuit(qs.QuantumRegister(blm+1,"add"))
    for i in range(blm,-1,-1):
        add_circuit.append(ql.PhaseGate(k*np.pi/(2**(blm-i))),qargs=[i])
    return add_circuit
def phase_adder(k,mod): #MAPPING: 0=WORK BIT, 1-10=ACTUAL MATH 
    blm = mod.bit_length() #Bit Len Mod 
    phase_circuit=qs.QuantumCircuit(qs.QuantumRegister(blm+1,"add"),qs.QuantumRegister(1,"adder"))
    phase_circuit.compose(add_k_fourier(k),inplace=True,qubits=range(0,blm+1))
    phase_circuit.compose(add_k_fourier(mod).reverse_ops().inverse(),inplace=True,qubits=range(0,blm+1))
    phase_circuit.append(ql.QFTGate(blm+1).inverse(),qargs=range(0,blm+1))
    phase_circuit.cx(0,blm+1,ctrl_state=1)
    phase_circuit.append(ql.QFTGate(blm+1),qargs=range(0,blm+1))
    phase_circuit.compose(add_k_fourier(mod).control(1,label="Add Mod"),inplace=True,qubits=[blm+1]+list(range(0,blm+1)))
    phase_circuit.compose(add_k_fourier(k).reverse_ops().inverse(),inplace=True,qubits=range(0,blm+1))
    phase_circuit.append(ql.QFTGate(blm+1).inverse(),qargs=range(0,blm+1))
    phase_circuit.append(ql.XGate().control(1,ctrl_state=0),qargs=[0,blm+1])
    phase_circuit.append(ql.QFTGate(blm+1),qargs=range(0,blm+1))
    phase_circuit.compose(add_k_fourier(k),inplace=True,qubits=range(0,blm+1))
    return phase_circuit
def mul_out_k_mod(k,mod):
    """Performs :math:`x times k` in the registers wires wires_aux"""
    blm = mod.bit_length() #Bit Len Mod 
    multiply_out = qs.QuantumCircuit(blm*2+2)
    qargs_list = [blm*2+1]
    qargs_list.extend(range(blm,blm*2))
    multiply_out.append(ql.QFTGate(blm+1),qargs=qargs_list)
    for idx in range(0,blm):
        codomain = list(range(blm,blm*2+2))
        new_list = [idx]
        new_list.extend(codomain)
        multiply_out.compose(phase_adder(k*(2**(blm-1-idx))%mod,mod).control(1),inplace=True,qubits=new_list)
    multiply_out.append(ql.QFTGate(blm+1).inverse(),qargs=qargs_list)
    return multiply_out

def modular_multiply(k,mod):
    blm = mod.bit_length() #Bit Len Mod 
    multiply_circuit = qs.QuantumCircuit(qs.QuantumRegister(blm,"multi"),qs.QuantumRegister(blm+1,"add"),qs.QuantumRegister(1,"adder"))
    multiply_circuit.compose(mul_out_k_mod(k,mod),inplace=True)
    for x_wire, aux_wire in zip(range(0,blm),range(blm,2*blm+1)):
        multiply_circuit.swap(x_wire, aux_wire)
    inv_k = pow(k, -1, mod)
    multiply_circuit.compose(mul_out_k_mod(inv_k,mod).reverse_ops().inverse(),inplace=True)
    return multiply_circuit

def modular_exponentiation(p:int,mod:int):
    blm = mod.bit_length() #Bit Len Mod 
    phase = qs.QuantumRegister(blm,"phase")
    multi = qs.QuantumRegister(blm,"multi")
    adder = qs.AncillaRegister(blm+1,"add")
    add_ancil = qs.AncillaRegister(1,"adder")
    shors_circuit = qs.QuantumCircuit(phase,multi,adder,add_ancil)
    shors_circuit.x(blm*2-1)
    for i,multiplier in enumerate(list(map(lambda x: ((p*x)%mod),range(1,blm+1)))):
        shors_circuit.compose(modular_multiply(multiplier,mod).control(1),qubits=[i]+list(range(blm,(3*blm+2))),inplace=True)
        shors_circuit.reset(range(2*blm,(3*blm)+2))
    shors_circuit.append(ql.QFTGate(blm).inverse(),range(0,blm))
    shors_circuit.measure_all()
    return shors_circuit