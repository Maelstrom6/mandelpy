from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QApplication, QFileDialog
from mandelpy import Settings, presets, post_processing, Generator
from mandelpy.validators import *
from PIL import Image
import sys
from gui.generated_ui import Ui_MainWindow
from ast import literal_eval
import traceback
from numba.cuda.cudadrv.driver import CudaAPIError, LinkerError
import time
from PyQt5.QtCore import QThread, pyqtSignal


def excepthook(type_, value, traceback_):
    traceback.print_exception(type_, value, traceback_)
    QtCore.qFatal('')


sys.excepthook = excepthook


class GenerateImageThread(QThread):
    progress = pyqtSignal(int)
    output = pyqtSignal(Image.Image)

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.running = True

    def __del__(self):
        self.wait()

    def run(self):
        with Generator(self.settings) as g:
            blocks = g.blocks
            for i, block in enumerate(blocks):
                if self.running is False:
                    return
                self.progress.emit(int(90 * (i + 1) / len(blocks)))
                try:
                    g.run_block(block)
                except CudaAPIError:  # blocks too big
                    self.output.emit(Image.open("./cuda_error.jpg"))
                    return
                except LinkerError:  # tried x**n which does not exist for n!=2
                    self.output.emit(Image.open("./cuda_error.jpg"))
                    return
            self.progress.emit(90)
            img = g.finished_img()
            self.output.emit(img)


class MainGUI(Ui_MainWindow, QtWidgets.QMainWindow):
    stop_signal = pyqtSignal(int)

    def __init__(self):
        super(MainGUI, self).__init__()
        self.setupUi(self)
        self.img: Image.Image = Image.open("./startup.png")
        self.adj_img = self.img.copy()
        self.update_image()
        self.connect_buttons()
        self.connect_actions()
        self.typeComboBox.addItems(["mand", "buddha", "julia", "julia_buddha", "orbit"])
        self.colorSchemeComboBox.addItems(["0", "1", "2", "3", "4", "5"])
        self.orbitTypeComboBox.addItems(["0", "1", "2", "3", "4"])
        self.presetNameComboBox.addItems(presets.keys())
        self.generate_image_thread = None
        self.stop_flag = QtWidgets.QProgressBar()
        self.stop_flag.setValue(0)

        self.settings_mapper = {
            "width": self.computedWidthLineEdit,
            "height": self.computedHeightLineEdit,
            "max_iter": self.maxIterationsLineEdit,
            "threshold": self.thresholdLineEdit,
            "tipe": self.typeComboBox,
            "z0": self.z0LineEdit,
            "fn_str": self.fnLineEdit,
            "transform_str": self.transformLineEdit,
            "inv_transform_str": self.inverseTransformLineEdit,
            "mirror_x": self.mirrorXAxisCheckBox,
            "mirror_y": self.mirrorYAxisCheckBox,
            "color_scheme": self.colorSchemeComboBox,
            "orbit_id": self.orbitTypeComboBox,
            "block_size": self.computeSizeLineEdit,
            "remove_h": self.removeCentreHorizontalCheckBox,
            "remove_v": self.removeCentreVerticalCheckBox,
            "hue": self.hueSlider,
            "saturation": self.saturationSlider,
            "brightness": self.brightnessSlider,
            "quality": self.qualitySlider
        }

        # need this otherwise if you clicked too fast on startup, it would crash
        time.sleep(1)

    def connect_buttons(self):
        self.refreshButton.clicked.connect(self.preview)
        self.saveButton.clicked.connect(self.save)
        self.resetHSBButton.clicked.connect(self.reset_hsb)
        self.loadPresetButton.clicked.connect(self.load_preset)
        self.cancelButton.clicked.connect(self.cancel_preview)
        self.refreshAdjButton.clicked.connect(self.perform_adjustments)

    def connect_actions(self):
        self.actionNew.triggered.connect(self.new)
        self.actionOpen.triggered.connect(self.eemport)
        self.actionImport.triggered.connect(self.eemport)
        self.actionExport.triggered.connect(self.export)
        self.actionSave.triggered.connect(self.save)
        self.actionSave_As.triggered.connect(self.save)

    def preview(self):
        self.clear_error_messages()
        self.stop_flag.setValue(0)
        QApplication.processEvents()
        settings = self.get_settings()
        if settings is None:
            return

        # I know this is a terrible way of handling threads but I don't know better and it works
        if self.generate_image_thread:
            self.cancel_preview()
            self.generate_image_thread.quit()
            self.generate_image_thread.wait()

        self.generate_image_thread = GenerateImageThread(settings)
        self.generate_image_thread.progress.connect(self.update_progress)
        self.generate_image_thread.output.connect(self.finish_preview)
        self.generate_image_thread.start()

        # if you don't want threading
        # self.img = create_image(settings, progress_bar=self.previewProgressBar)
        # self.perform_adjustments()

    def finish_preview(self, img: Image.Image):
        self.previewProgressBar.setValue(100)
        self.img = img
        self.perform_adjustments()

    def perform_adjustments(self):
        self.adj_img = self.img.copy()

        if self.removeCentreHorizontalCheckBox.isChecked():
            self.adj_img = post_processing.remove_centre_horizontal_pixels(self.img)

        if self.removeCentreVerticalCheckBox.isChecked():
            self.adj_img = post_processing.remove_centre_vertical_pixels(self.adj_img)

        self.adj_img = post_processing.enhance(self.adj_img,
                                               self.hueSlider.value() / 100,
                                               self.saturationSlider.value() / 100,
                                               self.brightnessSlider.value() / 100)

        self.update_image()

    def update_progress(self, value):
        self.previewProgressBar.setValue(value)

    def cancel_preview(self):
        self.previewProgressBar.setValue(0)
        if self.generate_image_thread:
            self.generate_image_thread.running = False

    def update_image(self):
        img = self.adj_img.copy()
        img.thumbnail((800, 800))
        pix_map = img.toqpixmap()
        self.label.setPixmap(pix_map)
        self.label.adjustSize()

    def get_settings(self):
        d = self.get_dict(settings_only=True)
        if d is None:
            return None
        d["fn"] = d.pop("fn_str")
        d["transform"] = d.pop("transform_str")
        d["inv_transform"] = d.pop("inv_transform_str")
        settings = Settings(**d)
        return settings

    def get_dict(self, settings_only=False):
        d = dict()
        x = self.validate(self.label_2, validate_float, self.realFocusLineEdit.text())
        y = self.validate(self.label_3, validate_float, self.imaginaryFocusLineEdit.text())
        ratio = self.validate(self.label_15, validate_float, self.aspectRatioLineEdit.text(),
                              positive=True)
        scale = self.validate(self.label_4, validate_float, self.scaleLineEdit.text(),
                              positive=True)

        d["left"] = x - scale
        d["right"] = x + scale
        d["top"] = y + scale / ratio
        d["bottom"] = y - scale / ratio

        width = self.validate(self.label_16, validate_int, self.computedWidthLineEdit.text(),
                              positive=True)
        height = self.validate(self.label_17, validate_int, self.computedHeightLineEdit.text(),
                               positive=True)

        d["width"] = width
        d["height"] = height

        d["max_iter"] = self.validate(self.label_5, validate_int, self.maxIterationsLineEdit.text(),
                                      positive=True)
        d["threshold"] = self.validate(self.label_6, validate_int, self.thresholdLineEdit.text(),
                                       positive=True)
        d["tipe"] = self.typeComboBox.currentText()
        d["z0"] = self.validate(self.label_8, validate_complex, self.z0LineEdit.text())

        d["fn_str"] = self.validate(self.label_9, validate_function,
                                    self.fnLineEdit.text(), ["z", "c"])
        d["fn_str"] = self.fnLineEdit.text()
        d["transform_str"] = self.validate(self.label_10, validate_function,
                                           self.transformLineEdit.text(), ["z"])
        d["transform_str"] = self.transformLineEdit.text()
        d["inv_transform_str"] = self.validate(self.label_11, validate_function,
                                               self.inverseTransformLineEdit.text(), ["z"])
        d["inv_transform_str"] = self.inverseTransformLineEdit.text()

        d["mirror_x"] = self.mirrorXAxisCheckBox.isChecked()
        d["mirror_y"] = self.mirrorYAxisCheckBox.isChecked()

        d["color_scheme"] = self.colorSchemeComboBox.currentIndex()
        d["orbit_id"] = self.orbitTypeComboBox.currentIndex()

        d["block_size"] = self.validate(self.label_14, validate_block_size,
                                        self.computeSizeLineEdit.text())

        if not settings_only:
            d["remove_h"] = self.removeCentreHorizontalCheckBox.isChecked()
            d["remove_v"] = self.removeCentreVerticalCheckBox.isChecked()
            d["hue"] = self.hueSlider.value()
            d["saturation"] = self.saturationSlider.value()
            d["brightness"] = self.brightnessSlider.value()
            d["quality"] = self.qualitySlider.value()

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
        width = self.validate(self.label_18, validate_int, self.widthLineEdit.text(),
                              positive=True)
        height = self.validate(self.label_19, validate_int, self.heightLineEdit.text(),
                               positive=True)

        default = QUrl(f"file:{self.fileNameLineEdit.text()}")
        file_name = QFileDialog.getSaveFileUrl(self.centralwidget,
                                               caption="Save image as PNG or JPEG.",
                                               filter="PNG file (*.png);;JPEG file (*.jpg)",
                                               directory=default,
                                               supportedSchemes=["png", "jpg"])
        file_name = QUrl.toLocalFile(file_name[0])

        if file_name != "":
            img = self.img.copy()
            img.thumbnail((width, height))
            img.save(file_name, quality=self.qualitySlider.value())

    def reset_hsb(self):
        self.hueSlider.setValue(100)
        self.saturationSlider.setValue(100)
        self.brightnessSlider.setValue(100)

    def load_preset(self):
        preset = presets[self.presetNameComboBox.currentText()]
        json_preset = preset.to_dict()
        self.load_dict(json_preset)

    def load_dict(self, d: dict):
        left = d.pop("left")
        right = d.pop("right")
        top = d.pop("top")
        bottom = d.pop("bottom")
        scale = (right - left) / 2
        ratio = scale / ((top - bottom) / 2)
        x = left + scale
        y = bottom + scale / ratio
        self.realFocusLineEdit.setText(str(x))
        self.imaginaryFocusLineEdit.setText(str(y))
        self.scaleLineEdit.setText(str(scale))
        self.aspectRatioLineEdit.setText(str(ratio))

        for k, v in d.items():
            item = self.settings_mapper[k]
            if isinstance(item, QtWidgets.QLineEdit):
                item.setText(str(v))
            elif isinstance(item, QtWidgets.QComboBox):
                index = item.findText(str(v), QtCore.Qt.MatchFixedString)
                item.setCurrentIndex(index)
            elif isinstance(item, QtWidgets.QCheckBox):
                item.setChecked(v)
            elif isinstance(item, QtWidgets.QSlider):
                item.setValue(v)

        self.mainTabWidget.setCurrentIndex(0)

    def export(self):
        d = self.get_dict()
        default = QUrl(f"file:project.txt")
        file_name = QFileDialog.getSaveFileUrl(self.centralwidget,
                                               caption="Save image settings as text.",
                                               filter="Text file (*.txt)",
                                               directory=default,
                                               supportedSchemes=["txt"])
        file_name = QUrl.toLocalFile(file_name[0])
        if file_name != "":
            with open(file_name, "w") as f:
                f.write(str(d))

    def eemport(self):
        file_name = QFileDialog.getOpenFileUrl(self.centralwidget,
                                               caption="Open image settings as text.",
                                               filter="Text file (*.txt)",
                                               supportedSchemes=["txt"])
        file_name = QUrl.toLocalFile(file_name[0])
        if file_name != "":
            with open(file_name, "r") as f:
                d = literal_eval(f.read())
            self.load_dict(d)

    def new(self):
        s = Settings()
        self.load_dict(s.to_dict())

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
