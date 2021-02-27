#!/usr/bin/env python3

"""
QAOA Noiseless + Noisy Simulations

(Based on Combinatorial optimization with QAOA, see: https://qiskit.org/textbook/ch-applications/qaoa.html)

author: dowusu
updated: Feb 26, 2021
"""

###############################################################################
# Useful imports...
###############################################################################
import sys

# Imports necessary graph and plotting tools
print("Importing plotting tools (from numpy, networkx, matplotlib, etc.)...")
import numpy as np
from random import random
from math import pi
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from time import time
print("Plotting tools imported.")

# Imports necessary qiskit tools to build and execute on IBMQ
print("Importing standard qiskit tools...")
from qiskit import Aer, IBMQ
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram, circuit_drawer
print("Qiskit tools imported.")


# Imports noise modeling tools
print("Importing qiskit noise tools...")
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
print("Qiskit noise tools imported.")

## Useful constants... (e.g., text separators for printing)
SHOTS = 10000           # number of execution repetitions, for sampling
EDGE_PROBABILITY = 0.5  # for building random connectivity graph (erdos-renyi)
SEP = "=" * 100
SEPS = {'enter': "=" * 10 + " ENTERING FUNCTION " + "=" * 10,
        'exit': "=" * 10 + " EXITING FUNCTION " + "=" * 10}
VERBOSE = 0 # 0 - no prints, 1 - basic prints, 2 - all prints
PLOT = 0 # 0 - no plotting, 1 - plotting

###############################################################################
# Useful functions...
###############################################################################

def show(descr=None):
    """
    Plots data and shows figure (maximized) for given description.
    """
    if descr:
        answer = input("Show '%s'? (y): " % descr)
        if answer != 'y':
            return
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.tight_layout()


def build_butterfly(num_qubits):
    """
    Builds butterfly graph representing qubit connectivity (see QAOA qiskit
      tutorial.
    """
    nodes = np.arange(num_qubits)
    edges = [(0, 1), (0, 2), (1,2), (3,2), (3,4), (4,2)]
    edge_weights = [1.0] * len(edges)
    weighted_edges = [edge + (edge_weights[index],) for index, edge in 
                      enumerate(edges)]
    butterfly = nx.Graph()
    butterfly.add_nodes_from(nodes)
    butterfly.add_weighted_edges_from(weighted_edges)
    return butterfly


def build_param_space(step):
    """
    Gets 2D grid of gamma and beta parameters.
    """
    gamma_axis, beta_axis = np.arange(0, pi, step), np.arange(0, pi, step)
    gamma_grid, beta_grid = np.meshgrid(gamma_axis, beta_axis) # why two grids?
    return gamma_grid, beta_grid


def get_optimal_butterfly():
    """
    Evaluates expectation, using grid search to find optimal parameters
      maximizing its value.
    """
    # Evaluates expectation via considering connectivity of butterfly graph
    step = 0.1
    gamma_grid, beta_grid = build_param_space(step)
    expectation = 3 - (np.sin(2 * beta_grid)**2 * np.sin(2 * gamma_grid)**2 \
                - 0.5 * np.sin(4 * beta_grid) * np.sin(4 * gamma_grid)) \
                * (1 + np.cos(4 * gamma_grid)**2)
    
    # Performs grid search (with numpy) on expectation, finding args-max
    max_expectation = np.max(expectation)
    optimum_indices = np.where(expectation == max_expectation)
    gamma_opt, beta_opt = [float(index * step) for index in optimum_indices]
    print("Optimal params (gamma, beta): %.02f, %.02f" % (gamma_opt, beta_opt))
    return gamma_opt, beta_opt, expectation


def plot_expectation_3D(expectation):
    """
    Plots graph of expectation value in 3D, as a function of gamma and
      beta parameters.
    """
    figure = plt.figure()
    axis = figure.gca(projection='3d')
    
    zmin, zmax = 1, 4
    COLORMAP = cm.coolwarm
    LINEWIDTH = 0
    ANTIALIAS = False  # should force opaque (irrelevant on jupyter...)
    
    axis.set_title("Expectation Value")
    axis.set_xlabel("Gamma")
    axis.set_ylabel("Beta")
    axis.set_zlabel("Expectation")
    axis.set_zlim(zmin, zmax)
    ###########################################################################
    # TODO: just plotting, not important...what do the following refer to?
    axis.zaxis.set_major_locator(LinearLocator(3))
    axis.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ########################################################################### 
    step = 0.1
    gamma_grid, beta_grid = build_param_space(step)
    surface = axis.plot_surface(gamma_grid, beta_grid, expectation,
                                cmap=COLORMAP, 
                                linewidth=LINEWIDTH,
                                antialiased=ANTIALIAS)
    if VERBOSE:
        show("cost v. parameters surface")
        max_expectation = np.max(expectation)
        print("Max expectation value: %.02f" % max_expectation)


def build_erdos_renyi(num_qubits, edge_probability):
    """
    Builds Erdos-Renyi graph (random graph) representing connectivity.
    """
    # Builds Erdos-Renyi graph (random graph)
    random_graph = nx.erdos_renyi_graph(num_qubits, edge_probability)
    nodes, edges = list(random_graph.nodes), list(random_graph.edges)
    
    # Updates edge weights for random graph
    edge_weights = [1.0] * len(edges)
    for index, edge in enumerate(edges):
        left, right = edge
        random_graph[left][right]['weight'] = edge_weights[index]
    return random_graph


def draw_nx_graph(graph):
    """
    Draw networkx graph with nodes and edges.
    """
    node_colors = ['y'] * len(graph.nodes)
    NODE_SIZE = 400
    TRANSPARENCY = 0
    default_axes = plt.axes(frameon=False)
    optimal_node_positioning = nx.spring_layout(graph)
    nx.draw_networkx(graph, node_color=node_colors, node_size=NODE_SIZE, 
                     alpha=1-TRANSPARENCY, ax=default_axes,
                     pos=optimal_node_positioning)


def build_circuit(nodes, edges, gamma_opt, beta_opt):
    """
    Builds quantum circuit to prepare trial state, using optimal parameters
    """
    # Builds empty circuit and initializes all qubits into superposition
    num_qubits = len(nodes)
    circuit = QuantumCircuit(num_qubits, num_qubits)
    circuit.h(nodes)

    # Applies circuit connectivity with Ising type gates
    circuit.barrier()
    for edge in edges:
        left_qubit, right_qubit = edge
        circuit.cp(-2 * gamma_opt, left_qubit, right_qubit) # cp replaces cu1
        circuit.p(gamma_opt, left_qubit)                    # p replaces u1
        circuit.p(gamma_opt, right_qubit)

    # Applies single qubit rotations to generate final state evolution 
    circuit.barrier()
    circuit.rx(2 * beta_opt, nodes)
    
    # Adds qubit measurements 
    circuit.measure_all()
    return circuit


def build_swap_circuit(nodes, edges, gamma_opt, beta_opt):
    """
    Builds quantum circuit using optimal paramaters, decomposing gates via 
      application of Fermionic SWAPS.
    """
    ## Decomposes gates (via application of Fermionic SWAPs) to improve success
    ##  probability.
    
    # Builds empty circuit and initializes all qubits (except the last, used as
    #  an "always on" control) into superposition.
    num_qubits = len(nodes)
    circuit = QuantumCircuit(num_qubits + 1, num_qubits + 1)
    #CONTROL_ALWAYS_ON = num_qubits
    #circuit.x(CONTROL_ALWAYS_ON)
    circuit.h(nodes)

    # Applies circuit connectivity with Ising type gates
    circuit.barrier()
    for edge in edges:
        left_qubit, right_qubit = edge

        # Decomposed cp(theta)
        circuit.cx(left_qubit, right_qubit)
        circuit.rz(-2 * gamma_opt, right_qubit)
        circuit.cx(right_qubit, left_qubit)
        circuit.cx(left_qubit, right_qubit)

        circuit.p(gamma_opt, left_qubit) 
        circuit.p(gamma_opt, right_qubit)       

    # Applies single qubit rotations to generate final state evolution 
    circuit.barrier()
    circuit.rx(2 * beta_opt, nodes)
    
    # Adds qubit measurements 
    circuit.measure_all()
    return circuit


# Defines cost function to compute cost given bitstring, to determine
# how good candidate bitstring is
def evaluate_cost(graph, bitstr):
    """
    Evaluates canonical cost function for combinatorial
    optimization, corresponding to MAXCUT.
    
    Inputs:
     graph: (networkx graph instance) 
     bitstr: (str) MAXCUT solution bitstring
    
    Returns cost value, else NaN if solution invalid (wrong length).
    """
    
    if len(bitstr) != len(graph.nodes):
        return np.nan
    
    cost = 0
    for edge in graph.edges():
        left_node, right_node = edge
        w = graph[left_node][right_node]['weight']
        cost += w * (int(bitstr[left_node]) * (1 - int(bitstr[right_node]))
                   + int(bitstr[right_node]) * (1 - int(bitstr[left_node])))
    return cost


def evaluate_data(counts, graph):
    """
    Performs evaluation of generated data.
    """ 
    # Reports sampled bitstring with maximum cost value
    samples = counts.keys()
    predicted_bitstr = {"bitstr": None, "cost": None}
    expected_cost = 0
    cost_values = {}
    num_qubits = len(graph.nodes)    

    for sample in samples:
        bitstr = sample[:num_qubits] # (?) sampled bitstring 2x qubits long...
        cost = evaluate_cost(graph, bitstr)
        current_max_cost = predicted_bitstr["cost"]
        if ((predicted_bitstr["bitstr"] == None) or 
            (cost > current_max_cost)):
            predicted_bitstr["bitstr"] = bitstr
            predicted_bitstr["cost"] = cost
    
        # Computes mean energy (?) and check if it agrees with prediction...
        probability = counts[sample] / SHOTS
        expected_cost += probability * cost
    
        # Stores costs in dictionary, for histogram plotting (to confirm
        # that costs concentrate around predicted mean)
        cost_label = str(round(cost))
        cost_values[cost_label] = cost_values.get(cost_label, 0) \
                                + counts[sample]

    return predicted_bitstr, expected_cost, cost_values


def add_noise(circuit, error_probability):
    """
    Adds depolarizing error channel with given error probability.
    """
    if VERBOSE:
        print(SEPS['enter'])
        print("Adding depolarizing error channel with %s error..."
              % error_probability)
    noise_model = NoiseModel()
    single_qubit_error = depolarizing_error(error_probability, 1)
    two_qubit_error = depolarizing_error(error_probability, 2)

    single_qubit_gates = ["p", "rx"]    # p <- u1
    two_qubit_gates = ["cp"]            #cp <- cu1
    noise_model.add_all_qubit_quantum_error(single_qubit_error,
                                            single_qubit_gates)
    noise_model.add_all_qubit_quantum_error(two_qubit_error,
                                            two_qubit_gates)
    #noise_model.add_quantum_error(error, ['p', 'u2', 'u3'], [0])
    #noise_model.add_nonlocal_quantum_error(error, ['p', 'u2', 'u3'], [0], [2]) 

    # Executes circuit with noisy QASM simulator
    backend = QasmSimulator(noise_model=noise_model)
    if VERBOSE:
        print("Executing circuit with noisy QASM...")
        print("Noisy backend: %s" % backend)
        print("Shots: %s" % SHOTS)
    noisy_result = execute(circuit, backend, shots=SHOTS).result()
    noisy_counts = noisy_result.get_counts()
    if VERBOSE:
        print(SEPS['exit'])
    return noise_model, noisy_counts


def estimate_gradient(qubit_graph, gamma, beta, error_probability):
    """
    Estimates gradient magnitudes (and expected costs) for evaluation.
    """
    nodes, edges = qubit_graph.nodes, qubit_graph.edges
    num_qubits = len(nodes)

    # For some parameter delta, estimates expected cost gradient.
    delta = 0.1
    for parameters_opt in [(gamma, beta), (gamma + delta, beta + delta)]:
        gamma_opt, beta_opt = parameters_opt
        circuit = build_swap_circuit(nodes, edges, gamma_opt, beta_opt)
        if error_probability > 0:
            ## bckd = add_noise() if error_probability else QASM_BACKEND
            depolarizing_noise, counts = add_noise(circuit,
                                                   error_probability)
        else:
            # Executes circuit with QASM simulator backend
            backend = Aer.get_backend("qasm_simulator")
            qaoa_result = execute(circuit, backend, shots=SHOTS).result()
            counts = qaoa_result.get_counts()
        predicted_bitstr, expected_cost, costs = evaluate_data(counts,
                                                               qubit_graph)

        # Remembers expected cost for current parameters...
        if parameters_opt == (gamma, beta):
            previous_cost = expected_cost

        if VERBOSE:
            print("Predicted bitstring: %s" %
                  predicted_bitstr["bitstr"])
            print("Cost: %f" % predicted_bitstr["cost"])
            print("Expected cost (energy): %s" % expected_cost)

    gradient = abs(expected_cost - previous_cost) / delta
    return gradient, previous_cost


def simulate(num_qubits_range, num_trials, error_probability):
    """
    Simulates circuit and evaluates data for some number of trials, returning
      expected costs.
    """
    # Over some number of trials, 
    expected_costs = {num_qubits:[] for num_qubits in num_qubits_range}
    gradient_magnitudes = {num_qubits:[] for num_qubits in num_qubits_range}
    for trial in range(num_trials):
        gamma, beta = random() * pi, random() * pi
        for num_qubits in num_qubits_range:
            qubit_graph = build_erdos_renyi(num_qubits, EDGE_PROBABILITY)
            gradient, expected_cost = estimate_gradient(qubit_graph, gamma,
                                                        beta, error_probability)
            expected_costs[num_qubits].append(expected_cost)
            gradient_magnitudes[num_qubits].append(gradient)
    print('=', end='', flush=True)
    variances = [np.var(expected_costs[num_qubits])
                 for num_qubits in num_qubits_range]
    gradient_magnitudes = [np.mean(gradient_magnitudes[num_qubits])
                           for num_qubits in num_qubits_range]
    return variances, gradient_magnitudes


#------------------------------------------------------------------------------#
def basic_simulate(nodes, edges, gamma_opt, beta_opt, noisy=False, swap=False):
    """
    Runs simulation on circuit given by set of qubits (nodes) and connectivity
     (edges), using given optimal parameters. 
    """
    # Builds quantum circuit to prepare trial state, using optimal parameters
    if swap:
        circuit = build_circuit(nodes, edges, gamma_opt, beta_opt)
    else:
        circuit = build_swap_circuit(nodes, edges, gamma_opt, beta_opt)
    circuit_image = circuit_drawer(circuit, output="latex")
    image_filename = "circuit.jpg" if not noisy else "noisy_circuit.jpg"
    circuit_image.save(image_filename)
    
    # Executes circuit with QASM simulator backend and plots results
    backend = Aer.get_backend("qasm_simulator")
    qaoa_result = execute(circuit, backend, shots=SHOTS).result()
    counts = qaoa_result.get_counts()
    if PLOT:
        plot_histogram(counts, bar_labels=False)
        show("noiseless bitstring probabilities")
    
    # Adds depolarizing error channel with probabiltiy 0.001
    if noisy:
        error_probability = 0.001
        noise_model, counts = add_noise(circuit, error_probability)

    # Evaluates data...
    predicted_bitstr, expected_cost, cost_values = evaluate_data(counts,
                                                                 qubit_graph)
    if VERBOSE:
        print("Predicted bitstring solution: %s" % predicted_bitstr["bitstr"])
        print("Cost: %f" % predicted_bitstr["cost"])
        print("Expected cost (energy): %s" % expected_cost)
    
    # Plots cost function distribution (over bitstrings sampled)
    if PLOT:
        plot_histogram(cost_values)
        show("noiseless cost distribution")
        if noisy:
            # Plots cost function distribution (over bitstrings sampled)
            show("noisy cost distirbution")
            plot_histogram(cost_values)

    return predicted_bitstr, expected_cost
#------------------------------------------------------------------------------#

if __name__ == "__main__":
    pass
