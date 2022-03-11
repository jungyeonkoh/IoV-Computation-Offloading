import sys
import csv
import random
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout
)

class Simulator(QWidget):
    def __init__(self):
        super().__init__()
        self.SimulatorUI()

    def SimulatorUI(self):
        # task
        self.lb1 = QLabel("Number of Vehicles")
        self.lb2 = QLabel("Length of Timestep")
        self.input1 = QLineEdit(self)
        self.input2 = QLineEdit(self)
        self.btnSimulateTask = QPushButton("Simulate Tasks")
        self.btnSimulateTask.pressed.connect(self.simulate_tasks)

        vbox1 = QVBoxLayout()
        vbox1.addWidget(self.lb1)
        vbox1.addWidget(self.input1)
        vbox1.addWidget(self.lb2)
        vbox1.addWidget(self.input2)
        vbox1.addWidget(self.btnSimulateTask)

        self.setLayout(vbox1)
        self.setWindowTitle("My Simulator")
        self.move(400, 100)
        self.resize(200, 200)
        self.show()

    def simulate_tasks(self):
        self.vehicles = int(self.input1.text())
        self.timestep = int(self.input2.text())

        f = open("simulated_tasks.csv", "w", newline="")
        wr = csv.writer(f)
        wr.writerow(["Timestamp", "Vehicle ID", "Threshold", "Input", "Computation", "Energy Weight"])

        for i in range(self.vehicles):
            for j in range(self.timestep):
                data_per_task = []
                data_per_task.append(j + 1) #timestamp
                data_per_task.append(i + 1) #vehicle_id
                data_per_task.append(np.random.normal(5, 1)) #threshold
                data_per_task.append(random.randint(1, 100)) #input
                data_per_task.append(random.randint(1,100)) #computation
                data_per_task.append(random.random()) #weight(0~1)

                wr.writerow(data_per_task)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = Simulator()
    sys.exit(app.exec_())
