import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QGridLayout, QLabel, QLineEdit, QPlainTextEdit, QDesktopWidget, QMessageBox, QWidget
from PyQt5.QtCore import Qt
import numpy as np
import pulp

def game_to_lp(payoff_matrix):
    A = -payoff_matrix.T
    b = np.ones(A.shape[0])
    c = np.zeros(A.shape[1])
    return c, A, b

def solve_zero_sum_game(payoff_matrix):
    min_val = np.min(payoff_matrix)
    constant = 0
    if min_val <= 0:
        constant = 1 - min_val
        payoff_matrix += constant

    num_strategies = payoff_matrix.shape[1]
    prob = pulp.LpProblem("Two-person Zero-Sum Game", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x{i}", lowBound=0) for i in range(num_strategies)]

    prob += pulp.lpSum(x), "Objective function"
    for row in range(payoff_matrix.shape[0]):
        prob += pulp.lpSum(payoff_matrix[row, i] * x[i] for i in range(num_strategies)) <= 1

    prob.solve()

    if prob.status != 1:
        raise ValueError("Failed to solve the linear programming problem. Error: " + pulp.LpStatus[prob.status])

    value = 1 / pulp.value(prob.objective) - constant
    strategy = [pulp.value(var) / pulp.value(prob.objective) for var in x]

    return strategy, value

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("Two-person Zero-Sum Game Solver")
        self.setFixedSize(800, 600)

        self.generalLayout = QVBoxLayout()
        self._centralWidget = QWidget(self)
        self.setCentralWidget(self._centralWidget)
        self._centralWidget.setLayout(self.generalLayout)

        self._createMatrixSizeInterface()
        self._createCopyright()

    def _createMatrixSizeInterface(self):
        sizeLayout = QGridLayout()

        self.label_rows = QLabel("Number of rows:")
        self.entry_rows = QLineEdit()
        sizeLayout.addWidget(self.label_rows, 0, 0)
        sizeLayout.addWidget(self.entry_rows, 0, 1)

        self.label_columns = QLabel("Number of columns:")
        self.entry_columns = QLineEdit()
        sizeLayout.addWidget(self.label_columns, 1, 0)
        sizeLayout.addWidget(self.entry_columns, 1, 1)

        self.submit_dimensions_button = QPushButton("Submit")
        sizeLayout.addWidget(self.submit_dimensions_button, 2, 0, 1, 2)

        self.submit_dimensions_button.clicked.connect(self.submit_dimensions)

        self.generalLayout.addLayout(sizeLayout)

    def submit_dimensions(self):
        if not self.entry_rows.text() or not self.entry_columns.text():
            QMessageBox.warning(self, "Input Error", "Please fill in both fields.")
            return

        try:
            rows = int(self.entry_rows.text())
            columns = int(self.entry_columns.text())
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter valid integers in both fields.")
            return

        self.matrix_entries = []

        matrixLayout = QGridLayout()

        for row in range(rows):
            matrix_row = []
            for column in range(columns):
                entry = QLineEdit()
                matrixLayout.addWidget(entry, row, column)
                matrix_row.append(entry)
            self.matrix_entries.append(matrix_row)

        self.solve_button = QPushButton("Solve")
        matrixLayout.addWidget(self.solve_button, rows, 0, 1, columns)

        self.solve_button.clicked.connect(self.solve)

        self.generalLayout.addLayout(matrixLayout)


    def solve(self):
        rows = len(self.matrix_entries)
        columns = len(self.matrix_entries[0])

        payoff_matrix = np.empty((rows, columns))

        for row in range(rows):
            for column in range(columns):
                payoff_matrix[row, column] = float(self.matrix_entries[row][column].text())

        strategy, value = solve_zero_sum_game(payoff_matrix)
        strategy_output = "\n".join([f"Strategy {i + 1}: {s:.4f}" for i, s in enumerate(strategy)])
        QMessageBox.information(self, "Optimal strategy and value",
                                f"Optimal strategy for Player 1:\n{strategy_output}\n\nValue of the game: {value:.4f}")

    def _createCopyright(self):
        copyrightLayout = QVBoxLayout()

        copyrightLabel = QLabel("Made by: Halfhero212\nGithub: https://github.com/Halfhero212")
        copyrightLabel.setAlignment(Qt.AlignCenter)
        
        copyrightLayout.addWidget(copyrightLabel)
        self.generalLayout.addLayout(copyrightLayout)

def main():
    q_app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(q_app.exec())

if __name__ == "__main__":
    main()
