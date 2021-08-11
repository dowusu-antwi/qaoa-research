#!/usr/bin/env python3

"""
QAOA Simulation

For MAXCUT optimization on simulated quantum circuits, estimates cost function
 gradient magnitudes.
"""

import numpy as np
import networkx as nx
from random import random
from qiskit import QuantumCircuit, execute
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error


class RandomCircuit:
    """
    Random quantum circuit built using Erdos-Renyi model.
    """
    def __init__(self, num_qubits, gate_parameters):
        constants = self.set_circuit_constants()
        (EDGE_PROBABILITY,
         SHOTS,
         PARAMETER_STEP,
         ONE_QUBIT_GATE_BASIS,
         TWO_QUBIT_GATE_BASIS) = constants

        self.edge_probability = EDGE_PROBABILITY
        self.execution_shots = SHOTS
        self.parameter_step = PARAMETER_STEP

        # Saves collection of native gates, along with the corresponding gate
        #  dimensions, for 1-qubit gates and 2-qubit gates.
        gate_bases = (ONE_QUBIT_GATE_BASIS, TWO_QUBIT_GATE_BASIS)
        self.gate_bases = gate_bases
        self.gate_dimensions = range(1, len(gate_bases) + 1)
        
        self.num_qubits = num_qubits
        self.gate_parameters = gate_parameters
        connectivity = self.create_connectivity(num_qubits, EDGE_PROBABILITY)
        self.connectivity = connectivity
        self.circuit = self.build_circuit(gate_parameters)


    def set_circuit_constants(self):
        """
        Generates constants and labels used by random circuit.
        """
        EDGE_PROBABILITY = 0.5
        SHOTS = 1000
        PARAMETER_STEP = 0.01
        ONE_QUBIT_GATE_BASIS = ["rx", "rz"]    # ["p", "rx"]
        TWO_QUBIT_GATE_BASIS = ["cx"]          # ["cp"]
        return (EDGE_PROBABILITY,
                SHOTS,
                PARAMETER_STEP,
                ONE_QUBIT_GATE_BASIS,
                TWO_QUBIT_GATE_BASIS)


    def create_connectivity(self, num_qubits, edge_probability):
        """
        Builds Erdos-Renyi graph (random graph) representing circuit
         connectivity.
        """
        graph = nx.erdos_renyi_graph(num_qubits, edge_probability)
        for edge in graph.edges:
            graph[edge[0]][edge[1]]["weight"] = 1.0
        return graph


    def build_circuit(self, gate_parameters):
        """
        Given connectivity, build QAOA circuit.
        """
        num_qubits = self.num_qubits
        connectivity = self.connectivity

        gamma, beta = gate_parameters
        qubits = connectivity.nodes
        edges = connectivity.edges

        num_inputs = num_qubits
        num_outputs = num_qubits
        circuit = QuantumCircuit(num_inputs, num_outputs)

        # Prepares QAOA circuit, initializing qubits into superposition, adding 
        #  circuit connectivity with Ising-type interactions, and applying a
        #  mixing layer with single qubit X-rotations.
        circuit.h(qubits)
        for edge in edges:
            self.ising_interaction(edge, circuit, gamma)
        circuit.rx(2 * beta, qubits)
        circuit.measure_all()

        return circuit


    def ising_interaction(self, edge, circuit, gamma):
        """
        Applies an Ising-type interaction to given edge, connecting qubits.
        """
        left_qubit, right_qubit = edge
        circuit.cp(-2 * gamma, left_qubit, right_qubit)
        circuit.p(gamma, left_qubit)
        circuit.p(gamma, right_qubit)


    def fold(self, folding_factor, circuit=None):
        """
        Applies unitary folding, given factor by which to scale noise.
        """
        #TODO: add unitary folding...
        pass


    def set_backend(self, backend):
        """
        Given backend parameter, sets circuit execution backend.
        """
        self.backend = backend


    def execute(self, circuit=None):
        """
        Executes quantum circuit on some chosen backend, obtaining sampled
         bitstring counts.
        """
        # Sets circuit execution parameters, using RandomCircuit circuit
        #  variable if no circuit parameter is given.
        circuit = circuit if circuit else self.circuit
        backend = self.backend
        execution_shots = self.execution_shots

        job = execute(circuit, backend, shots=execution_shots)
        result = job.result()
        counts = result.get_counts()
        return counts


    def evaluate_cost(self, bitstring):
        """
        Evaluates canonical cost function for combinatorial optimization
         corresponding to MAXCUT.
        """
        connectivity = self.connectivity
        cost = 0
        edges = connectivity.edges()
        for edge in edges:
            left_qubit, right_qubit = edge
            left_qubit_value = int(bitstring[left_qubit])
            right_qubit_value = int(bitstring[right_qubit])
            additive_term = (left_qubit_value * (1 - right_qubit_value)
                           + right_qubit_value * (1 - left_qubit_value))
            edge_weight = connectivity[left_qubit][right_qubit]["weight"]
            cost += edge_weight * additive_term
        return cost


    def get_expected_cost(self, counts):
        """
        Computes expected cost of bitstrings sampled from circuit execution.
        """
        # Calculates expected cost, weighted sum of costs with weights given by 
        #  bitstring frequency / probability.
        execution_shots = self.execution_shots
        expected_cost = 0
        for sampled_bitstring, count in counts.items():
            frequency = count / execution_shots
            expected_cost += frequency * self.evaluate_cost(sampled_bitstring)
        return expected_cost


    def estimate_gradient(self, folding_factor=None):
        """
        Compute cost at given point in parameter space and at two points each
         slightly offset along one of the parameter axes, respectively, to
         estimate cost function gradient vector (np.array(shape=(2, 1))).
        """
        circuit = self.fold(folding_factor) if folding_factor else None
        counts = self.execute(circuit)
        expected_cost = self.get_expected_cost(counts)
        parameter_step = self.parameter_step
        gamma, beta = self.gate_parameters

        # Computes expected cost at each position shifted along the parameter
        #  axes, using expected cost differences to estimate gradient vector.
        gamma_shift_params = (gamma + parameter_step, beta)
        circuit_gamma_shift = self.build_circuit(gamma_shift_params)
        circuit_gamma_shift = (self.fold(folding_factor, circuit_gamma_shift)
                               if folding_factor else circuit_gamma_shift)
        gamma_shift_counts = self.execute(circuit_gamma_shift)
        expected_cost_gamma_shift = self.get_expected_cost(gamma_shift_counts)

        beta_shift_params = (gamma, beta + parameter_step)
        circuit_beta_shift = self.build_circuit(beta_shift_params)
        circuit_beta_shift = (self.fold(folding_factor, circuit_beta_shift)
                              if folding_factor else circuit_beta_shift)
        beta_shift_counts = self.execute(circuit_beta_shift)
        expected_cost_beta_shift = self.get_expected_cost(beta_shift_counts)
 
        gradient = np.zeros(shape=(2,1))
        gamma_shift_difference = expected_cost - expected_cost_gamma_shift
        beta_shift_difference = expected_cost - expected_cost_beta_shift
        gradient[0] = abs(gamma_shift_difference) / parameter_step
        gradient[1] = abs(beta_shift_difference) / parameter_step
        return gradient


class Data:
    """
    Stores raw data and includes methods to extract processed data.
    """
    def __init__(self):
        constants = self.set_simulation_constants()
        (NUM_TRIALS,
         CIRCUIT_SIZES,
         MAX_GATE_PARAMS,
         ERROR_RATES,
         MAX_FOLD,
         NOISE_LVLS) = constants
        self.num_trials = NUM_TRIALS
        self.circuit_sizes = CIRCUIT_SIZES
        self.max_gate_params = MAX_GATE_PARAMS
        self.error_rates = ERROR_RATES
        self.max_fold = MAX_FOLD
        self.noise_levels = NOISE_LVLS
        raw_data, data_entry_ranges = self.initialize_data_array()
        self.raw_data = raw_data
        self.trial_averaged_data = None

        # Tracks data entries in which values have been stored.
        self.accessed_data_entries = set()

        # Ordered list of data entry ranges necessary to position a given data
        #  entry value (see self.store_value).
        self.data_entry_ranges = data_entry_ranges


    def set_simulation_constants(self):
        """
        Generates constants and labels used in simulation.
        """
        NUM_TRIALS = 1
        CIRCUIT_SIZES = range(4, 11)
        MAX_GAMMA, MAX_BETA = np.pi, np.pi
        MAX_GATE_PARAMS = MAX_GAMMA, MAX_BETA
        ERROR_RATES = ["0%", "3%", "5%", "10%", "15%"]
        MAX_FOLD = 4
    
        # For each error rate, we want to include a label for the "synthetic"
        #  noise levels generated by adding more noise (i.e., via unitary
        #  folding). We do this even for an error rate of 0%; in general, we
        #  would not perform ZNE on an already noise-free result, but we include
        #  it as a sanity check.
        NOISE_LEVELS = ERROR_RATES[:]
        folding_factors = range(2, MAX_FOLD + 1)
        for error_rate in ERROR_RATES:
            folded_noise_levels = [error_rate + " fold x" + str(folding_factor)
                                   for folding_factor in folding_factors]
            NOISE_LEVELS.extend(folded_noise_levels)
    
        return (NUM_TRIALS, CIRCUIT_SIZES, MAX_GATE_PARAMS, ERROR_RATES,
                MAX_FOLD, NOISE_LEVELS)


    def initialize_data_array(self):
        """
        Builds an empty array for storing raw data.
        """
        num_pages = self.num_trials
        num_rows = len(self.noise_levels)
        num_columns = len(self.circuit_sizes) 
        raw_data = np.zeros(shape=(num_pages, num_rows, num_columns))
        data_entry_ranges = (range(self.num_trials),
                             self.noise_levels,
                             self.circuit_sizes)
        return raw_data, data_entry_ranges


    def store_value(self, value, num_trial, noise_level, circuit_size):
        """
        Stores new data entry value given position indicated by trial number,
         noise level, and circuit size.
        """
        raw_data = self.raw_data
        data_entry_ranges = self.data_entry_ranges
        accessed_data_entries = self.accessed_data_entries

        # Orders trial number, noise level, and circuit size labels according to
        #  the order in which corresponding ranges appear in saved variable.
        position_labels = (num_trial, noise_level, circuit_size)
        position = tuple(data_entry_ranges[idx].index(position_labels[idx])
                         for idx in range(len(position_labels)))
        raw_data[position] = value
        accessed_data_entries.add(position_labels)
        


    def get_value(self, num_trial, noise_level, circuit_size):
        """
        Gets data entry value given position indicated by trial number, noise
         level, and circuit size.
        """
        raw_data = self.raw_data
        data_entry_ranges = self.data_entry_ranges
        accessed_data_entries = self.accessed_data_entries

        # Orders trial number, noise level, and circuit size labels according to
        #  the order in which corresponding ranges appear in saved variable.
        position_labels = (num_trial, noise_level, circuit_size)
        if position_labels in accessed_data_entries:
            position = tuple(data_entry_ranges[idx].index(position_labels[idx])
                             for idx in range(len(position_labels)))
            value = raw_data[position]
            return value


    def extract_graph_data(self, noise_levels_filter):
        """
        Gets input / output data for matplotlib plotting, given a set of
         filters.

        Inputs:
         noise_levels_filter: list of noise_levels indicating whether to extract
                              input / output data for given noise level.
        """
        raw_data = self.raw_data
        trial_averaged_data = self.trial_averaged_data
        data_entry_ranges = self.data_entry_ranges
        num_trials_range, noise_levels, circuit_sizes = data_entry_ranges

        # Averages raw data across trials and saves result.
        if not trial_averaged_data:
            trial_axis = data_entry_ranges.index(num_trials_range)
            trial_averaged_data = np.mean(raw_data, axis=trial_axis)
            self.trial_averaged_data = trial_averaged_data

        graph_data = []
        for noise_level_index, noise_level in enumerate(noise_levels):
            if noise_level in noise_levels_filter:
                output_data = trial_averaged_data[noise_level_index]
                graph_data.append((circuit_sizes, output_data))
        return graph_data


class Trial:
    """
    Builds and executes necessary circuit elements for a single simulation
     trial.
    """
    def __init__(self, circuit_size, max_gate_params, noise_levels):
        gate_parameters = self.initialize_gate_parameters(max_gate_params)
        self.gate_parameters = gate_parameters
        self.random_circuit = RandomCircuit(circuit_size, gate_parameters)
        self.circuit_size = circuit_size
        self.noise_levels = noise_levels


    def initialize_gate_parameters(self, max_gate_params):
        """
        Initialize new random gate parameters, gamma and beta.
        """
        MAX_GAMMA, MAX_BETA = max_gate_params
        return random() * MAX_GAMMA, random() * MAX_BETA


    def build_backend(self, noise_level):
        """
        Creates a (potentially noisy) backend.
        """
        # Processes noise level label to extract data for building noise model.
        MAX_NO_FOLD_LABEL_LENGTH = 3
        fold = len(noise_level) > MAX_NO_FOLD_LABEL_LENGTH
        error_rate = int(noise_level.split('%')[0]) / 100
        folding_scale_factor = (int(noise_level[::-1].split('x')[0])
                                if fold else None)

        ## Prints processed noise level data...(FOR DEBUGGING)
        #print("fold?: %s\t error rate: %s\t scale factor: %s" %
        #      (("Yes" if fold else "No"),
        #       error_rate,
        #       (folding_scale_factor if fold else "None")))

        #TODO: return folding scale factor...

        if error_rate == 0:
            return QasmSimulator()

        random_circuit = self.random_circuit
        gate_dimensions = random_circuit.gate_dimensions
        gate_bases = random_circuit.gate_bases

        simulation_noise_model = NoiseModel()
        for gate_dimension, gate_basis in zip(gate_dimensions, gate_bases):
            gate_error = depolarizing_error(error_rate, gate_dimension)
            simulation_noise_model.add_all_qubit_quantum_error(gate_error,
                                                               gate_basis)
        return QasmSimulator(noise_model = simulation_noise_model)


    def run(self, trial, data):
        """
        Iterates over given noise levels and runs QAOA simulation given
         circuit size and noise level.
        """
        noise_levels = self.noise_levels
        random_circuit = self.random_circuit
        circuit_size = self.circuit_size
        for noise_level in noise_levels:
            # Skips data entries in which values have already been stored.
            if data.get_value(trial, noise_level, circuit_size):
                continue
            backend = self.build_backend(noise_level)
            random_circuit.set_backend(backend)
            gradient = random_circuit.estimate_gradient()
            gradient_magnitude = np.linalg.norm(gradient)
            print("circuit size: %s,\t trial num: %s, \t noise level: %s"
                   % (circuit_size, trial, noise_level))
            data.store_value(gradient_magnitude,
                             trial,
                             noise_level,
                             circuit_size)


def simulate(data, fold=True):
    """
    Runs QAOA simulation for a range of circuit sizes (i.e., circuit width:
     number of qubits), updating given data structure.

    Inputs:
     fold (boolean): indicates whether or not to simulate noise levels that
                     include unitary folding for noise scaling.
    """
    # Iterate over range of circuit sizes, building Erdos-Renyi graph for each
    #  size.
    num_trials = data.num_trials
    circuit_sizes = data.circuit_sizes
    max_gate_params = data.max_gate_params
    noise_levels = (data.noise_levels if fold else
                    [noise_level for noise_level in data.noise_levels
                     if "fold" not in noise_level])

    for circuit_size in circuit_sizes:
        # Per circuit size, iterates over a given number of trials.
        trial_object = Trial(circuit_size, max_gate_params, noise_levels)
        for trial in range(num_trials):
            # Per trial, iterates over a given number of error rates and runs
            #  QAOA simulation given circuit size and error rate.
            trial_object.run(trial, data)


def main():
    """
    Simulates MAXCUT optimization (i.e., computing cost function values by
     simulating execution of quantum circuits).
    """
    # Generates empty dataset large enough to store all possible data. Access
    #  data elements by index, where each field (circuit width, trial, noise
    #  level) is mapped to an index (i.e., data[circuit width][trial][noise
    #  level]).
    data = Data()
    simulate(data)
    return data


if __name__ == "__main__":
    data = main()
