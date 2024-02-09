# This Python file uses the following encoding: utf-8
import os
from pathlib import Path
import sys
from model import Parse, Treiner_Tester, LeNet5, CifarNet

from PySide2.QtWidgets import QApplication, QWidget
from PySide2.QtCore import QFile
from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import QPushButton


class Widget(QWidget):
    def __init__(self):
        super(Widget, self).__init__()
        self.load_ui()

    def load_ui(self):
        loader = QUiLoader()
        path = os.fspath(Path(__file__).resolve().parent / "form.ui")
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        loader.load(ui_file, self)
        ui_file.close()
        self.learn_button = self.findChild(QPushButton, "learnButton")
        if self.learn_button:
            self.learn_button.clicked.connect(self.on_learn_click)

    def on_learn_click(self):
        parse_instance = Parse("MNIST")
        parse_instance.ParseData()

        accuracies = {}
        losses = {}

      # tanh
        trainer_tester_instance = Treiner_Tester(LeNet5(activation='tanh', pooling='avg',
                                                        conv_size=5,use_batch_norm=False, padd=2,
                                                        in_chan=1), 30, parse_instance)
        accuracies['tanh'], losses['tanh'] = \
         trainer_tester_instance.Train()



if __name__ == "__main__":
    app = QApplication([])
    widget = Widget()
    widget.show()
    sys.exit(app.exec_())
