#!/usr/bin/env python3

from qaoa_sim import *

def main():
    """
    Runs main QAOA simultation with various amounts of noise.
    """
    # Prepares data structures.
    error_probabilities = [0, 0.03, 0.05, 0.1, 0.15]
    #cost_variances_by_error = {}
    gradient_magnitudes_by_error = {}
    #variances = {error: [] for error in error_probabilities} # cost variances
    #variance_figure = plt.figure()
    gradient_figure = plt.figure()
    num_qubits_range = range(4,10)
    num_trials = 5
    
    # This iterates over each depolarizing error probability to generate a cost
    #  outputs for increasing circuit size (number of qubits). Per circuit size,
    #  cost value is averaged over random initializations of circuit parameters,
    #  i.e., gamma + beta.
    if VERBOSE:
        print("Iterating over depolarizing error probabilities...")
    start = time()
    for error_probability in error_probabilities:
        if VERBOSE:
            print("For error probability %s, iterating over qubit number..."
                  % error_probability)
    
        cost_variances, gradient_magnitudes = simulate(num_qubits_range,
                                                       num_trials,
                                                       error_probability)
        #plt.figure(variance_figure.number)
        #plt.plot(num_qubits_range, cost_variances, label=error_probability,
        #         marker='o', markersize=12)
        plt.figure(gradient_figure.number)
        plt.plot(num_qubits_range, gradient_magnitudes, label=error_probability,
                 marker='o', markersize=12)
        #cost_variances_by_error[error_probability] = cost_variances
        gradient_magnitudes_by_error[error_probability] = gradient_magnitudes
    
    print("") # moves to next line, after progress bar printing...
    runtime_sec = time() - start
    print("runtime (sec, min): %s, %s" % (runtime_sec, runtime_sec / 60))
    
    # Plots variance and gradient magnitudes over number of qubits.
    #plt.figure(variance_figure.number)
    #plt.title("Variance of Cost v. Circuit Size (# of Qubits)")
    #plt.legend(title="depolarization probability", loc='upper left')
    #plt.xlabel("number of qubits")
    #plt.ylabel("expected cost variance")
    #show()
    #plt.savefig("cost-versus-numqubits-%smin.png" % round(runtime_sec / 60))
    
    plt.figure(gradient_figure.number)
    plt.title("Gradient of Cost v. Circuit Size (# of Qubits)")
    plt.legend(title="depolarization probability", loc='upper left')
    plt.xlabel("number of qubits")
    plt.ylabel("gradient magnitude")
    plt.show() #show()
    #plt.savefig("gradient-versus-numqubits-%smin.png" % round(runtime_sec / 60))

if __name__ == "__main__":
    main()
