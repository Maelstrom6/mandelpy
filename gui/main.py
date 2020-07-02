from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
from mandelpy import Settings, create_image
from gui.validators import *
from PIL import Image
import sys
import time
from gui.generated_ui import Ui_MainWindow


class MainGUI(Ui_MainWindow, QtWidgets.QMainWindow):

    def __init__(self):
        super(MainGUI, self).__init__()
        self.setupUi(self)
        self.img = Image.open("../images/showcase/cave.png")
        self.update_image()
        self.connect_buttons()
        self.typeComboBox.addItems(["mand", "buddha", "julia", "julia_buddha", "orbit"])
        self.colorSchemeComboBox.addItems(["0", "1", "2", "3", "4", "5"])
        self.orbitTypeComboBox.addItems(["0", "1", "2", "3", "4"])


    def connect_buttons(self):
        self.refreshButton.clicked.connect(self.preview)
        self.saveButton.clicked.connect(self.save)

    def preview(self):
        self.clear_error_messages()
        QApplication.processEvents()
        settings = self.get_settings()
        if settings is not None:
            self.img = create_image(settings, progress_bar=self.previewProgressBar)
            self.img.thumbnail((800, 800))
            self.update_image()
        # i = 0
        # while self.previewProgressBar.value() < 100:
        #     self.previewProgressBar.setValue(i)
        #     i += 10
        #     time.sleep(1)

    def update_image(self):
        # print("A")
        # img = ImageQt.ImageQt(self.img)
        # print("B")
        # img = QtGui.QPixmap.fromImage(img)
        # print("C")
        # self.label.setPixmap(img)
        bytes_image = self.img.tobytes("raw", "RGB")
        image = QtGui.QImage(bytes_image, self.img.size[0], self.img.size[1],
                             QtGui.QImage.Format_RGB888)
        pix_map = QtGui.QPixmap.fromImage(image)
        self.label.setPixmap(pix_map)
        self.label.adjustSize()

    def get_settings(self):
        d = self.get_dict()
        if d is None:
            return None
        settings = Settings(**d)
        return settings

    def get_dict(self):
        d = dict()
        x = self.validate(self.label_2, validate_float, self.realFocusLineEdit.text())
        y = self.validate(self.label_3, validate_float, self.imaginaryFocusLineEdit.text())
        ratio = self.validate(self.label_15, validate_float, self.aspectRatioLineEdit.text(),
                              positive=True)
        scale = self.validate(self.label_4, validate_float, self.scaleLineEdit.text(),
                              positive=True)

        d["left"] = x - scale
        d["right"] = x + scale
        d["top"] = y + scale/ratio
        d["bottom"] = y - scale/ratio

        d["width"] = 1000
        d["height"] = int(1000/ratio)

        d["max_iter"] = self.validate(self.label_5, validate_int, self.maxIterationsLineEdit.text(),
                                      positive=True)
        d["threshold"] = self.validate(self.label_6, validate_int, self.thresholdLineEdit.text(),
                                       positive=True)
        d["tipe"] = self.typeComboBox.currentText()
        d["z0"] = self.validate(self.label_8, validate_complex, self.z0LineEdit.text())
        d["fn"] = self.validate(self.label_9, validate_function, self.fnLineEdit.text(), ["z", "c"])
        d["transform"] = self.validate(self.label_10, validate_function,
                                       self.transformLineEdit.text(), ["z"])
        d["inv_transform"] = self.validate(self.label_11, validate_function,
                                           self.inverseTransformLineEdit.text(), ["z"])
        d["mirror_x"] = self.mirrorXAxisCheckBox.isChecked()
        d["mirror_y"] = self.mirrorYAxisCheckBox.isChecked()

        d["color_scheme"] = self.colorSchemeComboBox.currentIndex()
        d["orbit_id"] = self.orbitTypeComboBox.currentIndex()

        d["block_size"] = self.validate(self.label_14, validate_block_size,
                                        self.computeSizeLineEdit.text())
        for k, v in d.items():
            if v is None:
                return None
        return d

    def clear_error_messages(self):
        self.label_2.setText("")
        self.label_3.setText("")
        self.label_4.setText("")
        self.label_5.setText("")
        self.label_6.setText("")
        self.label_7.setText("")
        self.label_8.setText("")
        self.label_9.setText("")
        self.label_10.setText("")
        self.label_11.setText("")
        self.label_12.setText("")
        self.label_13.setText("")
        self.label_14.setText("")

    def save(self):
        pass

    @staticmethod
    def validate(error_lbl, validator, *args, **kwargs):
        try:
            result = validator(*args, **kwargs)
        except ValueError as err:
            result = None
            error_lbl.setText(str(err))
            error_lbl.adjustSize()

        return result


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = MainGUI()
    ui.show()
    sys.exit(app.exec_())
