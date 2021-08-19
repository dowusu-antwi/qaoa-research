#!/usr/bin/env python3

from main import *

from qiskit import QuantumCircuit
from qiskit.visualization.matplotlib import MatplotlibDrawer as Drawer
from qiskit.visualization.utils import _get_layered_instructions as UtilOutput

from PyQt5 import QtWidgets, QtCore
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas

plt.switch_backend("Qt5Agg")


class SimulatorInterface(Simulator):
    """
    Interfaces simulator with visualizer, reimplementing its execution.
    """
    def __init__(self, noise_level_option):
        data = Data()
        self.max_gate_parameters = data.max_gate_parameters
        self.data = data
        steps = self.build_steps(noise_level_option)
        self.steps = steps
 

    def build_steps(self, noise_level_option):
        """
        Builds list of steps to traverse when running simulation.
        """
        data = self.data
        trials, noise_levels, circuit_sizes = data.dimension_ranges

        # These options are for simulating with no noise scaling or only those
        #  noise levels that include noise scaling, respectively.
        if noise_level_option == 1:
            noise_levels = [noise_level for noise_level in data.noise_levels
                            if "fold" not in noise_level]
        elif noise_level_option == 2:
            noise_levels = [noise_level for noise_level in data.noise_levels
                            if "fold" in noise_level]

        return [(trial, noise_level, circuit_size)
                for circuit_size in circuit_sizes
                for trial in trials
                for noise_level in noise_levels]


    def get_next_step(self):
        """
        Gets next circuit size, trial, and noise level to simulate.
        """
        steps = self.steps
        if len(steps) > 0:
            return steps.pop(0)
  

    def build_trial(self, trial, noise_level, circuit_size):
        """
        Builds environment for executing a trial of QAOA simulation.
        """
        data = self.data
        if data.get_value(trial, noise_level, circuit_size):
            return
        print("circuit size: %s,\t trial num: %s, \t noise level: %s"
               % (circuit_size, trial, noise_level))
        gate_parameters = self.initialize_gate_parameters()
        random_circuit = RandomCircuit(circuit_size, gate_parameters)
        return gate_parameters, random_circuit 


    def run(self, trial, noise_level, circuit_size, random_circuit):
        """
        Runs QAOA simulation given circuit size, trial, and noise level.
        """
        data = self.data
        gate_dimensions = random_circuit.gate_dimensions
        gate_bases = random_circuit.gate_bases
        backend, folding_scale_factor = self.build_backend(noise_level,
                                                           gate_dimensions,
                                                           gate_bases)
        random_circuit.set_backend(backend, folding_scale_factor)
        data.save_circuit_qasm(trial, noise_level, circuit_size,
                               random_circuit)
        gradient = random_circuit.estimate_gradient()
        gradient_magnitude = np.linalg.norm(gradient)
        data.store_value(gradient_magnitude,
                         trial,
                         noise_level,
                         circuit_size)


class Visualizer(QtWidgets.QWidget):
    """
    Visualizes simulation.
    """
    def __init__(self, simulator):
        """
        Initializes necessary attributes and methods.
        """
        self.application_manager = QtWidgets.QApplication([])
        super().__init__()
        self.resize(1500, 900)
        self.simulator = simulator
        self.trial_label = None
        self.noise_level_label = None
        self.circuit_size_label = None
        layout = self.build()
        self.layout = layout
        self.circuit_image = None
        self.circuit_image_axes = None
        self.parameter_map = None
        timer = self.setup_timer()
        self.timer = timer


    def resize(self, width, height):
        """
        Change size of console window.
        """
        self.setGeometry(0, 0, width, height)


    def setup_timer(self):
        """
        Sets up timer for updating simulation, with milliseconds for timer
         count.
        """
        # Simulation steps are slow as is, so they'll take as long as they need
        #  to before moving on to the next step...
        timer = QtCore.QTimer(self)
        TIMESTEP = 1                        # in milliseconds
        timer.start(TIMESTEP)
        timer.timeout.connect(self.update)
        return timer


    def build(self):
        """
        Builds simulation visualizer.
        """
        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)
        
        title = QtWidgets.QLabel("Simulation Visualizer")
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title, *(0, 0, 1, 6))

        toggle = QtWidgets.QPushButton("Pause")
        toggle.clicked.connect(self.toggle)
        layout.addWidget(toggle, *(1, 0, 1, 2))

        trial_label_title = QtWidgets.QLabel("Trial Number:")
        noise_level_label_title = QtWidgets.QLabel("Noise Level:")
        circuit_size_label_title = QtWidgets.QLabel("Circuit Size:")
        trial_label_title.setAlignment(QtCore.Qt.AlignCenter)
        noise_level_label_title.setAlignment(QtCore.Qt.AlignCenter)
        circuit_size_label_title.setAlignment(QtCore.Qt.AlignCenter)
        trial_label = QtWidgets.QLabel("")
        noise_level_label = QtWidgets.QLabel("")
        circuit_size_label = QtWidgets.QLabel("")
        trial_label.setAlignment(QtCore.Qt.AlignCenter)
        noise_level_label.setAlignment(QtCore.Qt.AlignCenter)
        circuit_size_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(trial_label_title, *(2, 0, 1, 1))
        layout.addWidget(trial_label, *(2, 1, 1, 1))
        layout.addWidget(noise_level_label_title, *(3, 0, 1, 1))
        layout.addWidget(noise_level_label, *(3, 1, 1, 1))
        layout.addWidget(circuit_size_label_title, *(4, 0, 1, 1))
        layout.addWidget(circuit_size_label, *(4, 1, 1, 1))

        self.trial_label = trial_label
        self.noise_level_label = noise_level_label
        self.circuit_size_label = circuit_size_label

        return layout


    def embed_matplotlib_objects(self, circuit_image, parameter_map):
        """
        Embeds matplotlib objects into PyQt toplevel Widget.
        """
        layout = self.layout
        circuit_image.tight_layout()
        circuit_image_canvas = Canvas(circuit_image)
        layout.addWidget(circuit_image_canvas, *(1, 2, 4, 4))
        parameter_map_canvas = Canvas(parameter_map)
        layout.addWidget(parameter_map_canvas, *(5, 0, 4, 6))
        self.circuit_image = circuit_image
        self.circuit_image_axes = circuit_image.get_axes()[0]
        self.parameter_map = parameter_map


    def toggle(self):
        """
        Toggles (i.e., pauses or resumes) simulation run.
        """


    def update_labels(self, trial, noise_level, circuit_size):
        """
        Updates GUI labels.
        """
        trial_label = self.trial_label
        noise_level_label = self.noise_level_label
        circuit_size_label = self.circuit_size_label
        trial_label.setText(str(trial))
        noise_level_label.setText(str(noise_level))
        circuit_size_label.setText(str(circuit_size))


    def update_figures(self, gate_parameters, random_circuit):
        """
        Updates circuit image and plot of gate parameters.
        """
        circuit_image = self.circuit_image
        circuit_image_axes = self.circuit_image_axes
        parameter_map = self.parameter_map
        layout = self.layout
        circuit = random_circuit.circuit

        plt.figure(parameter_map.number)
        beta_parameter, gamma_parameter = gate_parameters
        plt.scatter([beta_parameter], [gamma_parameter],
                    marker='x', color='r', s=250, linewidths=5)
        parameter_map.canvas.draw()

        circuit_image_axes.clear()
        circuit_image.canvas.draw()
        qubits, clbits, nodes = UtilOutput(circuit,
                                           reverse_bits=False,
                                           justify=None,
                                           idle_wires=True)
        drawing = Drawer(qubits, clbits, nodes, ax=circuit_image_axes,
                         qregs=circuit.qregs, cregs=circuit.cregs)
        drawing.draw()
        circuit_image.tight_layout()
        circuit_image.canvas.draw()


    def update(self):
        """
        Takes a simulation step, updating labels, circuit, and parameter map.
        """
        simulator = self.simulator
        next_step = simulator.get_next_step()
        if not next_step:
            timer = self.timer
            timer.stop()
        else:
            #TODO: add circuit and plot updating...
            trial, noise_level, circuit_size = next_step
            self.update_labels(trial, noise_level, circuit_size)
            (gate_parameters,
             random_circuit) = simulator.build_trial(trial,
                                                     noise_level,
                                                     circuit_size)
            self.update_figures(gate_parameters, random_circuit)
            simulator.run(trial, noise_level, circuit_size, random_circuit)


    def run(self):
        """
        Executes visualization.
        """
        self.show()
        self.application_manager.exec_()


def build_parameter_map(max_gate_parameters):
    """
    Gets sample parameter map.
    """
    max_beta_parameter, max_gamma_parameter = max_gate_parameters
    figure = plt.figure()
    plt.scatter([0], [0], marker='x', color='k', s=250, linewidths=5)
    plt.title("parameter map")
    plt.xlim(0, max_beta_parameter)
    plt.ylim(0, max_gamma_parameter)
    figure = plt.gcf()
    return figure


def build_circuit_image():
    """
    Gets sample circuit matplotlib.
    """
    figure = plt.figure()
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.h(1)
    circuit.measure_all()
    return circuit.draw(output="mpl")


def main():
    """
    Creates visualizer and runs simulation.
    """
    NOISE_LEVEL_OPTION = 1
    simulator = SimulatorInterface(NOISE_LEVEL_OPTION)
    visualizer = Visualizer(simulator)
    circuit_image = build_circuit_image()
    max_gate_parameters = simulator.max_gate_parameters
    parameter_map = build_parameter_map(max_gate_parameters)
    visualizer.embed_matplotlib_objects(circuit_image, parameter_map)
    visualizer.run()
    return visualizer


if __name__ == "__main__":
    visualizer = main()
