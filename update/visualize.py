#!/usr/bin/env python3

from qiskit import QuantumCircuit

from PyQt5 import QtWidgets, QtCore
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas

plt.switch_backend("Qt5Agg")

class Visualizer(QtWidgets.QWidget):
    """
    Visualizes simulation.
    """
    def __init__(self):
        """
        Initializes necessary attributes and methods.
        """
        self.application_manager = QtWidgets.QApplication([])
        super().__init__()
        self.resize(1500, 900)
        timer = self.setup_timer()
        self.timer = timer
        layout = self.build()
        self.layout = layout


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

        trial_label = QtWidgets.QLabel("Trial Number:")
        noise_level_label = QtWidgets.QLabel("Noise Level:")
        circuit_size_label = QtWidgets.QLabel("Circuit Size:")
        trial_label.setAlignment(QtCore.Qt.AlignCenter)
        noise_level_label.setAlignment(QtCore.Qt.AlignCenter)
        circuit_size_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(trial_label, *(2, 0, 1, 1))
        layout.addWidget(noise_level_label, *(3, 0, 1, 1))
        layout.addWidget(circuit_size_label, *(4, 0, 1, 1))

        return layout


    def embed_matplotlib_objects(self, circuit, parameter_map):
        """
        Embeds matplotlib objects into PyQt toplevel Widget.
        """
        layout = self.layout
        circuit_canvas = Canvas(circuit)
        layout.addWidget(circuit_canvas, *(1, 2, 4, 4))
        parameter_map_canvas = Canvas(parameter_map)
        layout.addWidget(parameter_map_canvas, *(5, 0, 4, 6))


    def toggle(self):
        """
        Toggles (i.e., pauses or resumes) simulation run.
        """


    def update(self):
        """
        Takes a simulation step.
        """


    def run(self):
        """
        Executes visualization.
        """
        self.show()
        self.application_manager.exec_()


def build_parameter_map():
    """
    Gets sample parameter map.
    """
    figure = plt.figure()
    plt.scatter([5], [5], marker='x', color='r', s=250, linewidths=5)
    figure = plt.gcf()
    return figure


def build_circuit():
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
    visualizer = Visualizer()
    circuit = build_circuit()
    parameter_map = build_parameter_map()
    visualizer.embed_matplotlib_objects(circuit, parameter_map)
    visualizer.run()

if __name__ == "__main__":
    main()
