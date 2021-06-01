#!/usr/bin/env python3

from qaoa_sim import *

def fill_data_space(data_key, simulation_conditions, input_ranges, input_data,
                    compute):
    """
    Fills data space with mean and/or variance computation, depending on
     simulation conditions.

    Inputs:
     data_key: a key to the dataset dictionary,
     simulation_conditions: conditions determining which data to compute,
     input_ranges: ranges of input over which to define empty data dictionary,
                   mapping inputs to an empty list of outputs,
     input_data: data to perform computation on,
     compute: function to perform on input data stream (i.e., mean or variance).

    Returns a list of newly computed data, or None.
    """
    # Note: last condition in if statement is so that exp_cost_variances doesn't
    #  try to compute variance of empty data... messy, but works for now.
    nibp_condition = simulation_conditions["nibp"]
    fermionic_condition = simulation_conditions["fermionic"]
    zne_condition = simulation_conditions["zne"]
    rounds_condition = simulation_conditions["rounds"]
    
    if ((not (nibp_condition
           or fermionic_condition
           or zne_condition
           or rounds_condition))
         or (compute == np.var and zne_condition)):
        return None

    NIBP_KEYS = {"standard"}
    FERMIONIC_KEYS = {"fermionic"}
    ZNE_KEYS = {"standard", "no fold", "double fold", "triple fold", "zne"}
    ROUNDS_KEYS = {"standard-rounds", "variances", "zne-rounds"}

    num_qubits_range, num_rounds_range = input_ranges

    if ((data_key in NIBP_KEYS and nibp_condition) or
        (data_key in FERMIONIC_KEYS and fermionic_condition) or
        (data_key in ZNE_KEYS and zne_condition)):
        input_range = num_qubits_range
        output_data = [compute(input_data[data_key][step])
                       for step in input_range]
        return output_data

    if (data_key in ROUNDS_KEYS and rounds_condition):
        # TODO: this is to avoid the 'mean of empty slice' error
        if data_key == "zne-rounds":
            return
        input_range = num_rounds_range
        output_data = [compute(input_data[data_key][step])
                       for step in input_range]
        return output_data


def get_plot_data(input_ranges, data, simulation_conditions, compute):
    """
    Extracts plotting data from expected cost and gradient magnitude data
     structures.
    """
    # Computes variance of expected costs and mean of gradient magnitudes over
    #  trials of parameter initialization.
    plot_data = {key: fill_data_space(key, simulation_conditions, input_ranges,
                                      data, compute)
                 for key in data}
    return plot_data


def save_data_to_string():
    """
    Given plot data / dictionary data structure, save data to string for easy
     replotting, with extra info about when generated and how long it took.
    """
    pass


def run_simulation(num_trials, sim_conditions, preloaded=False):
    """
    Runs main QAOA simulation with various amounts of noise, returning
     data structure containing all relevant data.
    """
    if num_trials > 1000:
        print("Number of trials exceeds 1000. Running 10 trials...")
        num_trials = 10

    # This iterates over each depolarizing error probability to generate cost
    #  gradients for increasing circuit size (number of qubits). Per circuit
    #  size,  gradient magnitude is averaged over random initializations of
    #  circuit parameters, i.e., gamma + beta.
    if VERBOSE:
        print("Iterating over depolarizing error probabilities...")

    num_qubits_range = range(4,10)
    num_rounds_range = range(1, 21) 
    input_ranges = (num_qubits_range, num_rounds_range)

    # Returns num_qubits_range, in case that datasets are pre-loaded.
    if preloaded:
        return input_ranges

    error_probabilities = [0, 0.03, 0.05, 0.1, 0.15]
    start = time()
    datasets = {"noise levels": error_probabilities,
                "raw expected costs": {},
                "raw gradient magnitudes": {},
                "exp cost variances": {},
                "avg grad magnitudes": {},
                "raw rounds data": {},
                "rounds magnitudes": {},
                "rounds variances": {}}
   
    # For 'rounds' case, builds qubit graph to be fixed over all noise levels.
    fixed_qubit_graph = build_erdos_renyi(FIXED_WIDTH, 1.0) # all edges
    draw_nx_graph(fixed_qubit_graph)
 
    for idx, noise_level in enumerate(error_probabilities):

        if VERBOSE:
            print("For error probability %s, iterating over qubit number..."
                  % noise_level)

        simulation_data = simulate(input_ranges,
                                   num_trials,
                                   noise_level,
                                   sim_conditions,
                                   fixed_qubit_graph)
        expected_cost_data, grad_magnitude_data, rounds_data = simulation_data
        variances = get_plot_data(input_ranges, expected_cost_data,
                                  sim_conditions, np.var)
        avg_grad_magnitudes = get_plot_data(input_ranges, grad_magnitude_data,
                                            sim_conditions, np.mean)
        rounds_avg_grad_magnitudes = get_plot_data(input_ranges,
                                                   rounds_data,
                                                   sim_conditions,
                                                   np.mean)
        rounds_variances = get_plot_data(input_ranges,
                                         rounds_data,
                                         sim_conditions,
                                         np.var)
        datasets["raw expected costs"][noise_level] = expected_cost_data
        datasets["raw gradient magnitudes"][noise_level] = grad_magnitude_data
        datasets["raw rounds data"][noise_level] = rounds_data
        datasets["exp cost variances"][noise_level] = variances
        datasets["avg grad magnitudes"][noise_level] = avg_grad_magnitudes
        datasets["rounds magnitudes"][noise_level] = rounds_avg_grad_magnitudes
        datasets["rounds variances"][noise_level] = rounds_variances

    print("") # moves to next line, after progress bar printing...
    runtime_sec = time() - start
    print("runtime (sec, min): %s, %s" % (runtime_sec, runtime_sec / 60))
    datasets["runtime"] = runtime_sec

    return datasets, num_qubits_range, num_rounds_range

if __name__ == "__main__":
    pass
