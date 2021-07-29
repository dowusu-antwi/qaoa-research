#!/usr/bin/env python3

import sys
from datetime import datetime
from random import randint
from matplotlib.lines import Line2D
import matplotlib
from textwrap import wrap
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
#matplotlib.rcParams['font.weight'] = 'bold'
#matplotlib.rcParams['text.usetex'] = True

from simulation import *

PLOT_COLORS = ['b', "orange", "green", 'r', "purple"]
ZERO_NOISE = 0

plt.rcParams.update({'font.size': 24})

def plot_cost_variances_demo(datasets, num_qubits_range, sim_condition):
    """
    Generates plot demonstrating vanishing variances (i.e., which indicates
     vanishing gradients, NIBPs).

    Inputs:
     datasets (dictionary): contains sorted simulation data,
     num_qubits_range (range object): input range for graphing,
     sim_condition: simulation condition corresponding to variance plot.

    Return a matplotlib figure instance, or False if no data available to plot.
    """
    nibp_sim, fermionic_sim = sim_condition
    if not (nibp_sim or fermionic_sim):
        print("No simulation data available.")
        return False

    figure = plt.figure(figsize=(8, 4.5))
    noise_levels = datasets["noise levels"]
    exp_cost_variances_per_noise = datasets["exp cost variances"]
    plot_lines = []

    for index, noise_level in enumerate(noise_levels):
        exp_cost_variances_data = exp_cost_variances_per_noise[noise_level]
        exp_cost_variances = exp_cost_variances_data["standard"]
        plot_objects = plt.plot(num_qubits_range, exp_cost_variances,
                               label=noise_level, marker='o', markersize=12,
                               color=PLOT_COLORS[index])
        plot_lines.append(plot_objects[0])
        if fermionic_sim:
            fermionic_exp_cost_variances = exp_cost_variances_data["fermionic"]
            plt.plot(num_qubits_range, fermionic_exp_cost_variances,
                     label=noise_level, marker='o', markersize=12,
                     color=PLOT_COLORS[index], ls='--')

    # Adds informative labeling (title, legend, axes labels) to plot.
    plt.title("Variance of Expected Cost v. Circuit Width (# of Qubits)")
    legend = plt.legend(plot_lines, noise_levels,
                        title="noise level (depolarization probability)",
                        loc='upper right')
    if fermionic_sim:
        plt.gca().add_artist(legend) # keep this legend instance
        legend_marks = [Line2D(range(4), range(4), color='k', lw=1, marker='o'),
                        Line2D(range(4), range(4), color='k', lw=1, marker='x',
                               ls='--')]
        plt.legend(legend_marks, ["ising type", "fermionic SWAP"],
                   title="connectivity", loc='lower right')
    plt.xlabel("number of qubits")
    plt.ylabel("variance of expected cost")
    return figure

def plot_nibps_demo(datasets, num_qubits_range, sim_condition):
    """
    Generates plot demonstrating vanishing gradients (NIBPs).

    Inputs:
     datasets (dictionary): contains sorted simulation data,
     num_qubits_range (range object): input range for graphing.

    Return a matplotlib figure instance, or False if no data available to plot.
    """
    nibp_sim, fermionic_sim = sim_condition
    if not (nibp_sim or fermionic_sim):
        print("No simulation data available.")
        return False

    # Generates figure and extracts plot data from dataset.
    figure = plt.figure(figsize=(8, 4.5))
    noise_levels = datasets["noise levels"]
    avg_grad_magnitudes_per_noise = datasets["avg grad magnitudes"]
    plot_lines = []

    # Per noise level, plots gradient magnitudes v. circuit width, averaged over
    #  trials in each of which gate parameters are randomly initialized.
    for index, noise_level in enumerate(noise_levels):
        avg_grad_magnitudes_data = avg_grad_magnitudes_per_noise[noise_level]
        avg_grad_magnitudes = avg_grad_magnitudes_data["standard"]
        plot_objects = plt.plot(num_qubits_range, avg_grad_magnitudes,
                                label=noise_level, marker='o', markersize=12,
                                color=PLOT_COLORS[index])
        plot_lines.append(plot_objects[0])
        if fermionic_sim:
            fermionic_avg_grad_mags = avg_grad_magnitudes_data["fermionic"]
            plt.plot(num_qubits_range, fermionic_avg_grad_mags,
                     label=noise_level, marker='o', markersize=12,
                      color=PLOT_COLORS[index], ls='--')

    # Adds informative labeling (title, legend, axes labels) to plot.
    plt.title("Gradient of Cost v. Circuit Size (# of Qubits)")
    legend = plt.legend(plot_lines, noise_levels,
                        title="noise level (depolarization probability)",
                        loc='upper right')
    if fermionic_sim:
        plt.gca().add_artist(legend) # keep this legend instance
        legend_marks = [Line2D(range(4), range(4), color='k', lw=1, marker='o'),
                        Line2D(range(4), range(4), color='k', lw=1, marker='x',
                               ls='--')]
        plt.legend(legend_marks, ["ising type", "fermionic SWAP"],
                   title="connectivity", loc='lower right')
    plt.xlabel("number of qubits")
    plt.ylabel("gradient magnitude")
    return figure


def plot_zne_demo(datasets, num_qubits_range, sim_condition, noise_level=None):
    """
    Generates plot demonstrating ZNE of vanshing gradients (i.e., combating
     NIBPs).

    Inputs:
     datasets (dictionary): contains sorted simulation data,
     num_qubits_range (range object): input range for graphing.

    Return a matplotlib figure instance, or False if no data available to plot.
    """
    if not sim_condition:
        print("No simulation data available.")
        return False    

    # Generates plot data to demonstrate zero noise extrapolation, comparing
    #  effects of noise, noise scaling (unitary folding), and extrapolation on
    #  gradient magnitude.
    
    # Generates figure and extracts plot data from dataset.
    figure = plt.figure(figsize=(8, 4.5))
    noise_levels = datasets["noise levels"]
    avg_grad_magnitudes_per_noise = datasets["avg grad magnitudes"]
    plot_lines = []

    # Per noise level, plots gradient magnitudes v. circuit width, averaged over
    #  trials in each of which gate parameters are randomly initialized.

    # 1. Plot standard at zero-noise,
    # 2. Plot no fold at some nonzero noise,
    # 3. Plot zne at the same nonzero noise. 


    # Select, at random, som nonzero noise level for ZNE to use.
    if noise_level==None:
        noise_level = noise_levels[randint(1, len(noise_levels) - 1)]
    print(f"RANDOMLY SELECTED NOISE LEVEL: {noise_level}")
    avg_grad_magnitudes_data = avg_grad_magnitudes_per_noise[noise_level]

    # Selects a random noise level to demonstrate ZNE
    noise_free = avg_grad_magnitudes_data["standard"]
    no_fold_noisy = avg_grad_magnitudes_data["no fold"]
    double_fold, triple_fold, zne = (avg_grad_magnitudes_data["double fold"],
                                     avg_grad_magnitudes_data["triple fold"],
                                     avg_grad_magnitudes_data["zne"])

    grad_magnitude_types = ["noise free", "no fold noisy", "double fold noisy",
                            "triple fold noisy", "zne"] 
    grad_magnitudes = {"noise free": noise_free,
                       "no fold noisy": no_fold_noisy,
                       "double fold noisy": double_fold,
                       "triple fold noisy": triple_fold,
                       "zne": zne}

    MARKERS = ['o', 'o', 'o', 'o', 'x']
    LINESTYLES = ['-', '-', '--', '--', '-']
    plot_lines = []
    for index, data_key in enumerate(grad_magnitude_types):
        # TODO: make sure plot colors doesn't fail, maybe generate new colors
        data = grad_magnitudes[data_key]
        plot_objects = plt.plot(num_qubits_range, data,
                                marker=MARKERS[index], markersize=12,
                                color=PLOT_COLORS[index],
                                ls=LINESTYLES[index])
        plot_lines.append(plot_objects[0])
        print(f"Plotted {data_key} in color {PLOT_COLORS[index]}")

    # Adds informative labeling (title, legend, axes labels) to plot.
    title = "zero noise extrapolation of gradient magnitudes"
    plt.title(title + f" (noise level = {noise_level})")
    legend = plt.legend(plot_lines, grad_magnitude_types,
                        title="grad magnitude type",
                        loc='upper right',
                        handlelength=3)
    plt.xlabel("number of qubits")
    plt.ylabel("gradient magnitude")
    return figure


def plot_zne_demo_full(datasets, num_qubits_range, sim_condition,
                       noise_level=None):
    """
    Generates plot demonstrating ZNE of vanshing gradients (i.e., combating
     NIBPs), but all on a single plot.

    Inputs:
     datasets (dictionary): contains sorted simulation data,
     num_qubits_range (range object): input range for graphing.

    Return a matplotlib figure instance, or False if no data available to plot.
    """
    if not sim_condition:
        print("No simulation data available.")
        return False    

    # Generates plot data to demonstrate zero noise extrapolation, comparing
    #  effects of noise, noise scaling (unitary folding), and extrapolation on
    #  gradient magnitude.
    
    # Generates figure and extracts plot data from dataset.
    figure, axis = plt.subplots(2, 2, figsize=(8, 4.5))
    noise_levels = datasets["noise levels"]
    avg_grad_magnitudes_per_noise = datasets["avg grad magnitudes"]
    plot_lines = []

    # Per noise level, plots gradient magnitudes v. circuit width, averaged over
    #  trials in each of which gate parameters are randomly initialized.

    # 1. Plot standard at zero-noise,
    # 2. Plot no fold at some nonzero noise,
    # 3. Plot zne at the same nonzero noise. 

    for noise_index, noise_level in enumerate(noise_levels[1:]):
        a, b = noise_index // 2, noise_index % 2
        avg_grad_magnitudes_data = avg_grad_magnitudes_per_noise[noise_level]

        # Selects a random noise level to demonstrate ZNE
        noise_free = avg_grad_magnitudes_data["standard"]
        no_fold_noisy = avg_grad_magnitudes_data["no fold"]
        double_fold, triple_fold = (avg_grad_magnitudes_data["double fold"],
                                    avg_grad_magnitudes_data["triple fold"])
        zne = avg_grad_magnitudes_data["zne"]

        grad_magnitude_types = ["noise free", "no fold noisy",
                                "double fold noisy", "triple fold noisy", "zne"]
        grad_magnitudes = {"noise free": noise_free,
                           "no fold noisy": no_fold_noisy,
                           "double fold noisy": double_fold,
                           "triple fold noisy": triple_fold,
                           "zne": zne}

        MARKERS = ['o', 'o', 'o', 'o', 'x']
        LINESTYLES = ['-', '-', '--', '--', '-']
        plot_lines = []
        for index, data_key in enumerate(grad_magnitude_types):
            # TODO: make sure plot colors doesn't fail, maybe generate new...
            data = grad_magnitudes[data_key]
            plot_objects = axis[a, b].plot(num_qubits_range, data,
                                           marker=MARKERS[index], markersize=12,
                                           color=PLOT_COLORS[index],
                                           ls=LINESTYLES[index])
            plot_lines.append(plot_objects[0])
            print(f"Plotted {data_key} in color {PLOT_COLORS[index]}")

        # Adds informative labeling (title, legend, axes labels) to plot.
        zne_title = f"Depolarization Probability = {int(noise_level * 100)}%"
        axis[a, b].set(title=zne_title,
                       xlabel="Circuit Width (Node Count)",
                       ylabel=r"Average |$\partial C_{\gamma\beta}$|")

    # Adds global figure titles and labels
    figure.suptitle("Zero Noise Extrapolation of Gradient Magnitudes (MAXCUT)",
                    fontweight='bold')
    PLOT_KEY = ["True Noise-Free", "Noisy", "Noise-Scaled (x2)",
                "Noise-Scaled (x3)", "ZNE"]
    figure.legend(plot_lines, ['\n' + '\n'.join(wrap(label, 10)) + '\n'
                               for label in PLOT_KEY],
                                   loc='center right',
                                   handlelength=3)
    plt.subplots_adjust(hspace=0.5, right=0.8, left=0.07)

    recovery_error = np.mean([abs(avg_grad_magnitudes_per_noise[0.03]['standard'][i] - avg_grad_magnitudes_per_noise[0.03]['zne'][i]) for i in range(6)])
    datasets['recovery error'] = recovery_error

    return figure


def plot_folding(datasets, num_qubits_range, noise_level=None):
    """
    Plots gradient magnitude results for unitary folding, per noise level.
    """
    figure, axis = plt.subplots(2, 2)
    #all_figure = plt.figure()

    MARKERSIZE = 7
    MEW = 4 #marker edge width

    NUM_QUBITS = 5
    n_qubits_index = num_qubits_range.index(NUM_QUBITS)
    if noise_level == None:
        noise_levels = datasets["noise levels"][1:]
    legend_elements = []
    for index, noise_level in enumerate(noise_levels):
        # Extracts data for current subplot.
        grad_mags = datasets["avg grad magnitudes"][noise_level]
        inputs = ["zne", "no fold", "double fold", "triple fold"]
        scale_factors = range(len(inputs))
        outputs = [grad_mags[inputs[scale_factor_index]][n_qubits_index] 
                   for scale_factor_index in range(len(inputs))]

        # Per axis subplot, adds plot data for folding, extrapolation, and
        #  true noise-free gradient magnitudes.
        a, b = index // 2, index % 2
        plot_object = axis[a, b].plot(scale_factors[1:], outputs[1:],
                                      marker='o', markersize=MARKERSIZE,
                                      color='k')
        legend_elements.extend(plot_object)
        plot_object = axis[a, b].plot(scale_factors[0:2], outputs[0:2],
                                      marker='o', markersize=MARKERSIZE,
                                      ls='--', color='b')
        legend_elements.extend(plot_object)

        # Sets titles and labels for axis subplot, growing list of legend
        #  marker elements (actually, legend markers hard-coded below instead..)
        fold_title = f"Depolarization Probability = {int(noise_level * 100)}%"
        axis[a, b].set_title(fold_title)
        axis[a, b].set(xlabel="Noise Scaling Factor",
                       ylabel=r"$|\partial C_{\gamma\beta}|$")
        axis[a, b].set_xticks(scale_factors)
        plot_object = axis[a, b].plot([0],
                                      [grad_mags["standard"][n_qubits_index]],
                                      marker='x', markersize=MARKERSIZE,
                                      lw=7, color='r', mew=MEW / 2)
        legend_elements.extend(plot_object)
        #plt.figure(all_figure.number)
        #plt.plot(scale_factors, outputs)
        #plt.figure(figure.number)

    # Adds a global plot legend, with hard-coded legend markers.
    legend_marks = [Line2D(range(1), range(1), color='k', lw=2, marker='o',
                           ls="None"),
                    Line2D(range(1), range(1), color='b', lw=2, marker='o',
                           ls="None"),
                    Line2D(range(1), range(1), color='r', lw=2, marker='x',
                           ls="None", mew=MEW)]
    LEGEND_KEYS = ["Unitary Folding", "Extrapolation", "True Noise-Free"]
    figure.legend(legend_marks, ['\n' + '\n'.join(wrap(label, 13)) + '\n'
                               for label in LEGEND_KEYS],
                  loc='center right',
                  handlelength=3,
                  markerscale=2)
    plt.subplots_adjust(hspace=0.6, wspace=0.3 ,right=0.8, left=0.06)

    # Plots maximum ZNE gradient magnitude
    #avg_grad_mags = datasets["avg grad magnitudes"]
    #plt.plot([0],
    #         [max([avg_grad_mags[noise]["zne"][n_qubits_index] 
    #               for noise in noise_levels])],
    #         marker='o')
    #plt.legend(datasets["noise levels"][1:], title="noise level")
    #plt.xticks(scale_factors)
    #plt.title("unitary folding")

    figure.suptitle(f"ZNE per Noise Level ({NUM_QUBITS} Node MAXCUT)",
                    fontweight="bold")
    #figure.tight_layout()
    return figure, axis#, all_figure


def plot_rounds_demo(datasets, num_rounds_range, rounds_condition):
    """
    """
    if rounds_condition == False:
        return
    noise_levels = datasets["noise levels"]
    avg_grad_magnitudes_per_noise = datasets["rounds magnitudes"]
   
    # Gets random noise level (for now...)
    ZERO_NOISE = noise_levels[0]
    NOISE_LEVEL = noise_levels[randint(1, len(noise_levels) - 1)]
    print(f"RANDOMLY SELECTED NOISE LEVEL: {NOISE_LEVEL}")
    print(f"ZERO NOISE LEVEL: {ZERO_NOISE}")

    # Plots noise free and noisy results
    plt.figure()
    noise_free_grad_magnitudes = avg_grad_magnitudes_per_noise[ZERO_NOISE]
    avg_grad_magnitudes_data = avg_grad_magnitudes_per_noise[NOISE_LEVEL]
    noise_free_per_round = noise_free_grad_magnitudes["standard-rounds"]
    noisy_per_round = avg_grad_magnitudes_data["standard-rounds"]
    plt.plot(num_rounds_range, noise_free_per_round, marker='o', color='k')
    plt.plot(num_rounds_range, noisy_per_round, marker='o', color='red')

    # Updates labels
    plt.xlabel(r"Number of QAOA Rounds (i.e., Circuit Depth $P$)")
    plt.xticks([int(num) for num in num_rounds_range])
    plt.ylabel(r"Average |$\partial C_{\gamma\beta}$|")
    plt.title("Average Gradient Magnitudes v. Circuit Depth")
    labels = ["Noise Free",
              f"Noisy (Depolarization Probability = {int(NOISE_LEVEL * 100)}%)"]
    plt.legend(labels) 

def main(datasets, num_qubits_range, num_rounds_range, simulation_conditions):
    """
    Plots extracted data from simulation.
    """

    # Extracts simulation conditions
    nibp_condition = simulation_conditions["nibp"]
    fermionic_condition = simulation_conditions["fermionic"]
    zne_condition = simulation_conditions["zne"]
    rounds_condition = simulation_conditions["rounds"]
    
    cost_variances_demo = plot_cost_variances_demo(datasets, num_qubits_range,
                                                   (nibp_condition,
                                                    fermionic_condition))
    nibps_demo = plot_nibps_demo(datasets, num_qubits_range,
                                 (nibp_condition, fermionic_condition))
    zne_demo = plot_zne_demo_full(datasets, num_qubits_range, zne_condition)
    rounds_demo = plot_rounds_demo(datasets, num_rounds_range, rounds_condition)
 
    if not nibps_demo:
        print("No NIBP plot generated.")
    else:
        plt.figure(nibps_demo.number)
        figure = show("live") # note: plt.show() shows ALL plotted figures

        # Saves plot to images subfolder (using datetime string formatting).
        runtime_sec = datasets["runtime"]
        filename = datetime.today().strftime("%Y-%m-%d-%H:%M") + "-grad-v-num" \
                   + "-runtime-%s-min" % round(runtime_sec / 60)
        filepath = "images/" + filename + ".pdf"
        if figure:
            figure.savefig(filepath, dpi=240)
        else:
            plt.savefig(filepath, dpi=240)

    if not cost_variances_demo:
        print("No cost variances plot generated.")
    else:
        plt.figure(cost_variances_demo.number)
        figure = show("live") # note: plt.show() shows ALL plotted figures

        # Saves plot to images subfolder (using datetime string formatting).
        runtime_sec = datasets["runtime"]
        filename = datetime.today().strftime("%Y-%m-%d-%H:%M") \
                   + "-cost-variances" \
                   + "-runtime-%s-min" % round(runtime_sec / 60)
        filepath = "images/" + filename + ".pdf"
        if figure:
            figure.savefig(filepath, dpi=240)
        else:
            plt.savefig(filepath, dpi=240) 

    if zne_demo:
        plt.figure(zne_demo.number)
        figure = show("live")
        runtime_sec = datasets["runtime"]
        filename = datetime.today().strftime("%Y-%m-%d-%H:%M") + "-zne" \
                   + "-runtime-%s-min" % round(runtime_sec / 60)
        filepath = "images/" + filename + ".pdf"
        figure.savefig(filepath, dpi=240)
    else:
        print("No ZNE plot generated.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("---> ERROR: Input number of trials...")
    else:
        sim_conditions = {"nibp": False, "fermionic": False, "zne": True,
                          "rounds": False}
        num_trials = int(sys.argv[1])
        if len(sys.argv) > 2:
            extract_data = sys.argv[2]
        else:
            extract_data = None 
        if extract_data is not None:
            print("Attempting to extract data...")
            try:
                datasets = load(open("datasets.p", "rb"))
                print("Data successfully extracted.")
                simulation_data = run_simulation(num_trials, sim_conditions,
                                                 preloaded=True)
                num_qubits_range, num_rounds_range = simulation_data
            except:
                print("Data unavailable. Running simulation...")
                simulation_data = run_simulation(num_trials, sim_conditions)
                datasets, num_qubits_range, num_rounds_range = simulation_data
                print("Dumping data to a pickle file...")
                dump(datasets, open("datasets.p", "wb"))
                print("Data dumped.")
        else:
            simulation_data = run_simulation(num_trials, sim_conditions)
            datasets, num_qubits_range, num_rounds_range = simulation_data 
            print("Dumping data to a pickle file...")
            dump(datasets, open("datasets.p", "wb"))
            print("Data dumped.")
        main(datasets, num_qubits_range, num_rounds_range, sim_conditions)
