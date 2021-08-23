#!/usr/bin/env python3

"""
QAOA Simulation

For MAXCUT optimization on simulated quantum circuits, estimates cost function
 gradient magnitudes.
"""

import numpy as np
import networkx as nx
from random import random
from qiskit import QuantumCircuit, execute, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
from mitiq.zne.scaling import fold_gates_from_right
from mitiq.zne.inference import PolyFactory

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
        self.backend = None
        self.folding_scale_factor = None


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


    def build_circuit(self, gate_parameters, folding_scale_factor=None):
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


    def fold(self, folding_scale_factor, circuit=None):
        """
        Applies unitary folding, given factor by which to scale noise.
        """
        circuit = self.circuit if not circuit else circuit
        if not folding_scale_factor or folding_scale_factor == 1:
            return circuit
        return fold_gates_from_right(circuit, folding_scale_factor)


    def set_backend(self, backend, folding_scale_factor):
        """
        Given backend parameters, sets circuit execution backend.
        """
        self.backend = backend
        self.folding_scale_factor = folding_scale_factor


    def execute(self, circuit=None):
        """
        Executes quantum circuit on some chosen backend, obtaining sampled
         bitstring counts.
        """
        # Sets circuit execution parameters, using RandomCircuit circuit
        #  variable if no circuit parameter is given.
        circuit = self.circuit if not circuit else circuit
        backend = self.backend
        execution_shots = self.execution_shots

        # Decomposes circuit into a fixed basis that works with mitiq folding
        #  methods. We do this even for non-folding cases so that we are always
        #  executing using the same basis gates.
        one_qubit_basis, two_qubit_basis = self.gate_bases
        basis = one_qubit_basis + two_qubit_basis
        circuit = transpile(circuit, backend, basis_gates=basis) 

        # Applies unitary folding, conditioned on whether the given backend
        #  noise level includes it.
        folding_scale_factor = self.folding_scale_factor
        circuit = self.fold(folding_scale_factor, circuit)

        # Executes circuit on given backend and extracts resulting bitstring
        #  counts.
        job = execute(circuit, backend,
                      optimization_level=0,    # stops collapse of folding
                      shots=execution_shots)
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


    def estimate_gradient(self, circuit=None):
        """
        Compute cost at given point in parameter space and at two points each
         slightly offset along one of the parameter axes, respectively, to
         estimate cost function gradient vector (np.array(shape=(2, 1))).
        """
        circuit = self.circuit if not circuit else circuit
        counts = self.execute(circuit)
        expected_cost = self.get_expected_cost(counts)
        parameter_step = self.parameter_step
        gamma, beta = self.gate_parameters

        # Computes expected cost at each position shifted along the parameter
        #  axes, using expected cost differences to estimate gradient vector.
        gamma_shift_params = (gamma + parameter_step, beta)
        circuit_gamma_shift = self.build_circuit(gamma_shift_params)
        gamma_shift_counts = self.execute(circuit_gamma_shift)
        expected_cost_gamma_shift = self.get_expected_cost(gamma_shift_counts)

        beta_shift_params = (gamma, beta + parameter_step)
        circuit_beta_shift = self.build_circuit(beta_shift_params)
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
         EXTRAPOLATION_ORDER) = constants
        noise_levels, folding_factors = self.get_noise_levels(ERROR_RATES,
                                                              MAX_FOLD)
        self.num_trials = NUM_TRIALS
        self.circuit_sizes = CIRCUIT_SIZES
        self.max_gate_parameters = MAX_GATE_PARAMS
        self.error_rates = ERROR_RATES
        self.max_fold = MAX_FOLD
        self.extrapolation_order = EXTRAPOLATION_ORDER
        self.folding_factors = folding_factors
        self.noise_levels = noise_levels
        self.extrapolator = self.build_extrapolator(folding_factors,
                                                    EXTRAPOLATION_ORDER)
        self.raw_data = self.initialize_data_array()
        self.trial_averaged_data = None

        # Tracks data entries in which values have been stored.
        self.accessed_data_entries = set()

        # Ordered list of data entry ranges necessary to position a given data
        #  entry value (see self.store_value).
        dimension_ranges = self.initialize_dimension_ranges()
        self.dimension_ranges = dimension_ranges

        # Initializes empty array for storing circuit QASM strings.
        qasm_data = self.initialize_data_array()
        self.qasm_data = qasm_data.astype("object")


    def set_simulation_constants(self):
        """
        Generates constants and labels used in simulation.
        """
        NUM_TRIALS = 2
        CIRCUIT_SIZES = range(4, 11)
        MAX_GAMMA, MAX_BETA = np.pi, np.pi
        MAX_GATE_PARAMS = MAX_GAMMA, MAX_BETA
        ERROR_RATES = ["0%", "3%", "5%", "10%", "15%"]
        MAX_FOLD = 4
        EXTRAPOLATION_ORDER = 2
    
        return (NUM_TRIALS, CIRCUIT_SIZES, MAX_GATE_PARAMS, ERROR_RATES,
                MAX_FOLD, EXTRAPOLATION_ORDER)


    def get_noise_levels(self, error_rates, max_fold):
        """
        Generates noise level labels, error rates together with "synthetic"
         noise levels from folding and extrapolating to zero noise.
        """ 
        # For each error rate, we want to include a label for the "synthetic"
        #  noise levels generated by adding more noise (i.e., via unitary
        #  folding). We do this even for an error rate of 0%; in general, we
        #  would not perform ZNE on an already noise-free result, but we include
        #  it as a sanity check.
        noise_levels = error_rates[:]
        folding_factors = range(2, max_fold + 1)
        for error_rate in error_rates:
            folded_noise_levels = [error_rate + " fold x" + str(folding_factor)
                                   for folding_factor in folding_factors]
            noise_levels.extend(folded_noise_levels)
            noise_levels.append(error_rate + " zne")
        return noise_levels, folding_factors


    def build_extrapolator(self, folding_factors, extrapolation_order):
        """
        Builds extrapolator object to perform zero-noise extrapolation.
        """
        UNIT_SCALE = 1.0 # required for polynomial extrapolation
        scale_factors = [UNIT_SCALE]
        scale_factors.extend(folding_factors)
        extrapolator = PolyFactory(scale_factors,
                                   order=extrapolation_order)
        return extrapolator


    def initialize_data_array(self):
        """
        Builds an empty array for storing raw data.
        """
        num_pages = self.num_trials
        num_rows = len(self.noise_levels)
        num_columns = len(self.circuit_sizes) 
        raw_data = np.zeros(shape=(num_pages, num_rows, num_columns))
        return raw_data


    def initialize_dimension_ranges(self):
        """
        Creates list of data entry ranges in particular order necessary to
         position a given data entry value (see store_value method).
        """
        dimension_ranges = (range(self.num_trials),
                            self.noise_levels,
                            self.circuit_sizes)
        return dimension_ranges


    def get_position(self, trial, noise_level, circuit_size):
        """
        Calculates position in numpy array corresponding to given trial number, 
         noise level, and circuit size.
        """
        # Orders trial number, noise level, and circuit size labels according to
        #  the order in which corresponding ranges appear in saved variable.
        dimension_ranges = self.dimension_ranges
        data_entry_label = (trial, noise_level, circuit_size)
        position = tuple(dimension_ranges[idx].index(data_entry_label[idx])
                         for idx in range(len(data_entry_label)))
        return position


    def store_value(self, value, trial, noise_level, circuit_size):
        """
        Stores new data entry value given position indicated by trial number,
         noise level, and circuit size.
        """
        raw_data = self.raw_data
        
        data_entry_label = (trial, noise_level, circuit_size)
        position = self.get_position(*data_entry_label)
        raw_data[position] = value
        accessed_data_entries = self.accessed_data_entries
        accessed_data_entries.add(data_entry_label)
        
        qasm = self.get_circuit_qasm(trial, noise_level, circuit_size)
        str_data_entry = ("circuit size: %s,\t trial num: %s, \t noise level: "\
                          "%s, \t value: %s,\t qasm: %s\n"\
                          % (circuit_size, trial, noise_level, value, qasm))
        


    def get_value(self, trial, noise_level, circuit_size):
        """
        Gets data entry value given position indicated by trial number, noise
         level, and circuit size.
        """
        raw_data = self.raw_data
        data_entry_label = (trial, noise_level, circuit_size)
        accessed_data_entries = self.accessed_data_entries
        if data_entry_label in accessed_data_entries:
            position = self.get_position(*data_entry_label)
            value = raw_data[position]
            return value


    def save_circuit_qasm(self, trial, noise_level, circuit_size, circuit):
        """
        Converts circuit to QASM and saves it to position indicated by trial
         number, noise level, and circuit size.

        Note: this saves QASM of raw circuit *before* folding.
        """
        qasm = circuit.qasm()
        position = self.get_position(trial, noise_level, circuit_size)
        qasm_data = self.qasm_data
        qasm_data[position] = qasm


    def get_circuit_qasm(self, trial, noise_level, circuit_size):
        """
        Returns circuit QASM for position indicated by trial number,
         noise level, and circuit size.
        """
        position = self.get_position(trial, noise_level, circuit_size)
        qasm_data = self.qasm_data
        return qasm_data[position]


    def extract_graph_data(self, noise_levels_filter=None):
        """
        Gets input / output data for matplotlib plotting, given a set of
         filters.

        Inputs:
         noise_levels_filter: list of noise_levels indicating whether to extract
                              input / output data for given noise level.
        """
        raw_data = self.raw_data
        trial_averaged_data = self.trial_averaged_data
        dimension_ranges = self.dimension_ranges
        num_trials_range, noise_levels, circuit_sizes = dimension_ranges

        # Averages raw data across trials and saves result.
        if not trial_averaged_data:
            trial_axis = dimension_ranges.index(num_trials_range)
            trial_averaged_data = np.mean(raw_data, axis=trial_axis)
            self.trial_averaged_data = trial_averaged_data

        graph_data = dict()
        if noise_levels_filter == None:
            noise_levels_filter = self.noise_levels
        for noise_level_index, noise_level in enumerate(noise_levels):
            if noise_level in noise_levels_filter:
                output_data = trial_averaged_data[noise_level_index]
                graph_data[noise_level] = (circuit_sizes, output_data)
        return graph_data


    def extrapolate_zero_noise(self, trial, noise_level, circuit_size):
        """
        Extrapolates computed gradient magnitudes for folded noise levels to
         zero-noise limit.
        
        (by np.polyfit under the hood; see https://mitiq.readthedocs.io)
        """
        extrapolator = self.extrapolator
        folding_factors = self.folding_factors
        extrapolation_order = self.extrapolation_order

        noise = noise_level.split(" zne")[0]
        noise_labels = [noise]
        noise_labels.extend([noise + " fold x" + str(folding_factor)
                             for folding_factor in folding_factors])
        gradient_magnitudes = [self.get_value(trial,
                                              noise_label,
                                              circuit_size)
                               for noise_label in noise_labels]

        UNIT_SCALE = 1.0 # required for polynomial extrapolation
        scale_factors = [UNIT_SCALE]
        scale_factors.extend(folding_factors)
        return extrapolator.extrapolate(scale_factors,
                                        gradient_magnitudes,
                                        order=extrapolation_order)


class Simulator:
    """
    Builds and executes necessary circuit elements for a QAOA circuit simulator,
     given a fixed circuit size.
    """
    def __init__(self, simulation_option):
        # Generates empty dataset large enough to store all possible data.
        #  Access data elements by index, where each field (circuit width,
        #  trial, noise level) is mapped to an index (i.e.,
        #  data[circuit width][trial][noise level]).
        self.data = Data()
        self.max_gate_parameters = self.data.max_gate_parameters
        self.steps = self.build_steps(simulation_option)
        self.trial = None
        self.circuit_size = None
        self.random_circuit = None


    def initialize_gate_parameters(self):
        """
        Initialize new random gate parameters, gamma and beta.
        """
        MAX_GAMMA, MAX_BETA = self.max_gate_parameters
        return random() * MAX_GAMMA, random() * MAX_BETA


    def build_backend(self, noise_level, random_circuit):
        """
        Creates a (potentially noisy) backend.
        """
        # Processes noise level label to extract data for building noise model.
        fold = "fold" in noise_level
        error_rate = int(noise_level.split('%')[0]) / 100
        folding_scale_factor = (int(noise_level[::-1].split('x')[0])
                                if fold else None)

        ## Prints processed noise level data...(FOR DEBUGGING)
        #print("fold?: %s\t error rate: %s\t scale factor: %s" %
        #      (("Yes" if fold else "No"),
        #       error_rate,
        #       (folding_scale_factor if fold else "None")))

        if error_rate == 0:
            return QasmSimulator(), folding_scale_factor

        simulation_noise_model = NoiseModel()
        gate_dimensions = random_circuit.gate_dimensions
        gate_bases = random_circuit.gate_bases
        for gate_dimension, gate_basis in zip(gate_dimensions, gate_bases):
            gate_error = depolarizing_error(error_rate, gate_dimension)
            simulation_noise_model.add_all_qubit_quantum_error(gate_error,
                                                               gate_basis)
        return (QasmSimulator(noise_model = simulation_noise_model),
                folding_scale_factor)


    def get_next_step(self):
        """
        Gets next circuit size, trial, and noise level to simulate.
        """
        steps = self.steps
        return steps.pop(0) if len(steps) > 0 else None


    def build_steps(self, simulation_option):
        """
        Builds list of steps to traverse when running simulation.
        """
        data = self.data
        trials, noise_levels, circuit_sizes = data.dimension_ranges

        # These options are for simulating with no noise scaling or only those
        #  noise levels that include noise scaling, respectively.
        if simulation_option == 1:
            noise_level_filter = (lambda noise_lvl:
                                  not (("fold" in noise_lvl) or
                                       ("zne" in noise_lvl)))
        elif simulation_option == 2:
            noise_level_filter = (lambda noise_lvl:
                                  "fold" in noise_lvl or "zne" in noise_lvl)

        noise_levels = [noise_level for noise_level in data.noise_levels
                        if noise_level_filter(noise_level)]

        return [(trial, noise_level, circuit_size)
                for circuit_size in circuit_sizes
                for trial in trials
                for noise_level in noise_levels]
        

    def get_random_circuit(self, trial, circuit_size):
        """
        Gets current random circuit, updating if necessary.
        """
        # Changes circuit if trial changes or circuit size changes, getting new
        #  parameters and building new circuit.
        current_trial = self.trial
        current_circuit_size = self.circuit_size
        update_circuit = ((trial != current_trial) or
                          (circuit_size != current_circuit_size))
        if update_circuit:
            gate_parameters = self.initialize_gate_parameters()
            random_circuit = RandomCircuit(circuit_size, gate_parameters)
            self.trial = trial
            self.circuit_size = circuit_size
            self.random_circuit = random_circuit
        return self.random_circuit


    def execute_trial_circuit(self, trial, noise_level, circuit_size):
        """
        Executes circuit and computes magnitude of estimated cost function
         gradient.
        """
        random_circuit = self.get_random_circuit(trial, circuit_size)
        data = self.data
        circuit = random_circuit.circuit
        data.save_circuit_qasm(trial, noise_level, circuit_size, circuit)
        backend, folding_scale_factor = self.build_backend(noise_level,
                                                           random_circuit)
        random_circuit.set_backend(backend, folding_scale_factor)
        gradient = random_circuit.estimate_gradient()
        gradient_magnitude = np.linalg.norm(gradient)
        return gradient_magnitude


    def run(self, trial, noise_level, circuit_size):
        """
        Runs QAOA simulation given trial, circuit size, and noise level.

        Note: per circuit size, each trial uses a different random circuit, i.e.
         with a new set of gate parameters - but given a single circuit for many
         trials, value still varies, likely due to randomness of execution (?)
        """
        # Skips data entries in which values have already been stored.
        data = self.data 
        if data.get_value(trial, noise_level, circuit_size):
            return
        print("circuit size: %s,\t trial num: %s, \t noise level: %s"
               % (circuit_size, trial, noise_level))
        if "zne" in noise_level:
            extrapolation = data.extrapolate_zero_noise(trial,
                                                        noise_level,                                                                    circuit_size)
            value = extrapolation
        else:
            gradient_magnitude = self.execute_trial_circuit(trial,
                                                            noise_level,
                                                            circuit_size)
            value = gradient_magnitude
        data.store_value(value,
                         trial,
                         noise_level,
                         circuit_size)


    def reset(self, option):
        """
        Resets simulation given new option.
        """
        #TODO: re-opens file to write data to
        self.steps = self.build_steps(option)


    def clean(self):
        """
        Closes file saving simulation data.
        """
        #TODO: closes file with written data


def simulate(simulator=None, option=0):
    """
    Runs QAOA simulation for a range of circuit sizes (i.e., circuit width:
     number of qubits), updating given data structure.

    Inputs:
     option (int): indicates whether to simulate all noise levels (0), only
                   those that don't include any noise scaling (1), or only those
                   that include unitary folding for noise scaling (2).
    """
    if simulator == None:
        simulator = Simulator(option)
    else:
        simulator.reset(option)
    
    next_step = simulator.get_next_step()
    while next_step:
        # Per circuit size, iterates over a given number of trials. Per trial,
        #  iterates over a given number of error rates and runs QAOA simulation
        #  given circuit size and error rate.
        trial, noise_level, circuit_size = next_step
        simulator.run(trial, noise_level, circuit_size)
        next_step = simulator.get_next_step()
    simulator.clean()
    return simulator


def main():
    """
    Simulates MAXCUT optimization (i.e., computing cost function values by
     simulating execution of quantum circuits).
    """
    # Chooses one of three simulation options:
    #  0) all noise levels,
    #  1) noise levels without noise scaling,
    #  2) noise levels with noise scaling.
    simulator = simulate(option=1)
    data = simulator.data
    return data


if __name__ == "__main__":
    data = main()
