import sys
from PyQt5.QtWidgets import QApplication
from ui.workout_screen import WorkoutScreen


def main():
    app = QApplication(sys.argv)
    window = WorkoutScreen()
    window.setWindowTitle("Fitness Assistant")

    # Open window in maximized mode (with minimize, maximize, close buttons)
    window.showMaximized()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
