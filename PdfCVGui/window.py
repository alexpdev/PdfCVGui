import math
import os
import fitz
import cv2
import numpy as np
from PIL.ImageQt import ImageQt
from PIL import Image
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
import subprocess
import tempfile
import parsel


THRESH_TYPES = {
    "THRESH_BINARY": cv2.THRESH_BINARY,
    "THRESH_BINARY_INV": cv2.THRESH_BINARY_INV,
    "THRESH_TRUNC": cv2.THRESH_TRUNC,
    "THRESH_TOZERO": cv2.THRESH_TOZERO,
    "THRESH_TOZERO_INV": cv2.THRESH_TOZERO_INV,
    "THRESH_MASK": cv2.THRESH_MASK,
    "THRESH_OTSU": cv2.THRESH_OTSU,
    "THRESH_TRIANGLE": cv2.THRESH_TRIANGLE,
}




def cvToImage(img):
    arr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(arr).convert('RGB')
    qimage = ImageQt(image)
    return qimage


class Window(QMainWindow):
    """Window object."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self.central = QWidget(parent=self)
        self.layout = QVBoxLayout(self.central)
        self.toolbar = QToolBar(self)
        self.list_widget = QListWidget(self)
        self.scroll_area = QScrollArea(self)
        self.contours_button = QPushButton("Contours")
        self.scroll_widget = QWidget()
        self.leftside = QWidget()
        vlayout = QVBoxLayout(self.leftside)
        self.open_button = QPushButton("Open")
        vlayout.addWidget(self.open_button)
        vlayout.addWidget(self.list_widget)
        self.leftside.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred))
        self.vlayout = QVBoxLayout()
        self.hlayout = QHBoxLayout()
        self.vlayout.addWidget(self.toolbar)
        self.vlayout.addWidget(self.scroll_area)
        self.hlayout.addWidget(self.leftside)
        self.hlayout.addLayout(self.vlayout)
        self.layout.addLayout(self.hlayout)
        self.scroll_area.setWidget(self.scroll_widget)
        self.scroll_area.setWidgetResizable(True)
        self.grid = QGridLayout(self.scroll_widget)
        self.row = self.col = 0
        self.setCentralWidget(self.central)
        self.setObjectName('MainWindow')
        self.open_button.clicked.connect(self.open_new_pdf)
        self.add_toolbar_buttons()
        self.set_default_pdf()
        self.list_widget.currentItemChanged.connect(self.highlight_label)
        self.resize(900,800)
        self.list_widget.sizePolicy().setHorizontalPolicy(QSizePolicy.Policy.Fixed)
        self.list_widget.installEventFilter(self)

    def extract_text(self, img):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmpfile = tmp.name
            cv2.imwrite(tmpfile, img)
            hocr = subprocess.check_output(f"tesseract {tmpfile} stdout --psm 11 -l eng hocr", stderr=subprocess.STDOUT, shell=True)
            self.hocr = hocr.decode("utf-8")
        selector = parsel.Selector(self.hocr)
        return selector

    def ocr_img(self):
        item = self.get_selected_item()
        if not item: return
        img = item.image
        response = self.extract_text(img)
        nimg = img.copy()
        for elem in response.xpath("//span[@class='ocrx_word']"):
            bbox = elem.xpath("./@title").get()
            bbox = bbox[4:bbox.index(";")]
            rect = [int(i) for i in bbox.split()]
            nimg[rect[1]:rect[3], rect[0]:rect[2]] = 255
        child = QListWidgetItem()
        child.setText(item.text() + "-ocred")
        child.image = nimg
        self.add_image_to_grid(nimg, child)
        self.list_widget.addItem(child)

    def eventFilter(self, widget, event):
        if event.type() == QEvent.Type.KeyPress:
            if event.keyCombination().key()._name_ == "Key_Delete":
                self.delete_seleced_items()
        return False

    def delete_seleced_items(self):
        items = self.list_widget.selectedItems()
        for item in items:
            index = self.list_widget.indexFromItem(item)
            item = self.list_widget.takeItem(index.row())
            label = item.label
            self.grid.removeWidget(label)
            label.deleteLater()
        empty = []
        for row in range(self.grid.rowCount()):
            for column in range(self.grid.columnCount()):
                item = self.grid.itemAtPosition(row, column)
                if not item or not item.widget():
                    empty.append((row, column))
                else:
                    if len(empty) > 0:
                        r,c = empty.pop(0)
                        for i in range(self.grid.count()):
                            if self.grid.itemAt(i) == item:
                                j = self.grid.takeAt(i)
                                self.grid.addWidget(j.widget(), r, c)
                                break
        if len(empty) > 0:
            r,c = empty.pop(0)
            self.row = r
            self.col = c

    def set_default_pdf(self):
        root = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(root, "pdf", "T2.pdf")
        self.open_new_pdf(path)

    def clear_label_selection(self):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            label = item.label
            label.setStyleSheet("border: solid transparent 0px;")

    def highlight_label(self, item):
        self.clear_label_selection()
        if item:
            label = item.label
            label.setStyleSheet("border: 3px solid cyan;")

    def get_selected_item(self):
        items = self.list_widget.selectedItems()
        if not items:
            return None
        return items[0]

    def convert_grayscale(self):
        item = self.get_selected_item()
        if not item: return
        img = item.image
        newimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        child = QListWidgetItem()
        child.setText(item.text() + "-grayscaled")
        child.image = newimg
        self.add_image_to_grid(newimg, child)
        self.list_widget.addItem(child)

    def add_image_to_grid(self, img, item):
        label = QLabel()
        val = int(img.shape[1] * .50)
        image = cvToImage(img).scaledToWidth(val)
        pixmap = QPixmap.fromImage(image)
        label.setPixmap(pixmap)
        if self.col < 2:
            self.grid.addWidget(label, self.row, self.col)
            self.col += 1
        else:
            self.grid.addWidget(label, self.row, self.col)
            self.row += 1
            self.col = 0
        item.label = label

    def blur_image(self):
        item = self.get_selected_item()
        if not item: return
        img = item.image
        blur_value = self.blur_spinbox.value()
        if blur_value % 2 == 0 and blur_value > 0:
            blur_value -= 1
        newimg = cv2.GaussianBlur(img, (blur_value, blur_value), 0)
        child = QListWidgetItem()
        child.setText(item.text() + f"-blurred-{blur_value}")
        child.image = newimg
        self.add_image_to_grid(newimg, child)
        self.list_widget.addItem(child)

    def add_toolbar_buttons(self):
        self.gray_button = QToolButton(self)
        self.gray_button.setText("Grayscal")
        self.gray_button.clicked.connect(self.convert_grayscale)
        self.toolbar.addWidget(self.gray_button)
        self.blur_groupbox = QGroupBox()
        self.blur_layout = QHBoxLayout(self.blur_groupbox)
        self.blur_spinbox = QSpinBox()
        self.blur_spinbox.setRange(0,25)
        self.blur_button = QToolButton(self)
        self.blur_button.setText("Blur")
        self.blur_button.clicked.connect(self.blur_image)
        self.blur_layout.addWidget(self.blur_button)
        self.blur_layout.addWidget(self.blur_spinbox)
        self.toolbar.addWidget(self.blur_groupbox)
        self.thresh_groupbox = QGroupBox()
        self.thresh_layout = QHBoxLayout(self.thresh_groupbox)
        self.thresh_spinbox = QSpinBox()
        self.thresh_spinbox.setRange(-25,25)
        self.thresh_button = QToolButton()
        self.thresh_button.setText("Adap. Thresh.")
        self.thresh_button.clicked.connect(self.apply_threshold)
        self.thresh_layout.addWidget(self.thresh_button)
        self.thresh_layout.addWidget(self.thresh_spinbox)
        self.toolbar.addWidget(self.thresh_groupbox)
        self.dilate_groupbox = QGroupBox()
        self.dilate_layout = QHBoxLayout(self.dilate_groupbox)
        self.dilate_spinbox = QSpinBox()
        self.dilate_spinbox.setRange(-25,25)
        self.dilate_button = QToolButton()
        self.dilate_button.setText("Dilate")
        self.dilate_button.clicked.connect(self.dilate_image)
        self.dilate_layout.addWidget(self.dilate_button)
        self.dilate_layout.addWidget(self.dilate_spinbox)
        self.toolbar.addWidget(self.dilate_groupbox)
        self.erode_button = QToolButton(self)
        self.erode_button.setText("erode")
        self.erode_button.clicked.connect(self.erode)
        self.toolbar.addWidget(self.erode_button)
        self.thresh_std_groupbox = QGroupBox()
        self.thresh_std_layout = QHBoxLayout(self.thresh_std_groupbox)
        self.thresh_std_spinbox = QSpinBox()
        self.thresh_std_button = QToolButton()
        self.thresh_std_combo = QComboBox()
        for i in THRESH_TYPES:
            self.thresh_std_combo.addItem(i)
        self.thresh_std_spinbox.setRange(-255,255)
        self.thresh_std_button.setText("Thresh. Std.")
        self.thresh_std_layout.addWidget(self.thresh_std_button)
        self.thresh_std_layout.addWidget(self.thresh_std_spinbox)
        self.thresh_std_layout.addWidget(self.thresh_std_combo)
        self.thresh_std_button.clicked.connect(self.thresh_std)
        self.toolbar.addWidget(self.thresh_std_groupbox)
        self.extract_button = QToolButton()
        self.extract_button.setText("OCR")
        self.extract_button.clicked.connect(self.ocr_img)
        self.toolbar.addWidget(self.extract_button)

    def thresh_std(self):
        value = self.thresh_std_spinbox.value()
        typ = self.thresh_std_combo.currentText()
        thresh_type = THRESH_TYPES[typ]
        item = self.get_selected_item()
        if not item: return
        img = item.image
        _, thresh = cv2.threshold(img, value, 255, thresh_type)
        child = QListWidgetItem()
        child.setText(item.text() + f"-thresh.std")
        child.image = thresh
        self.add_image_to_grid(thresh, child)
        self.list_widget.addItem(child)

    def dilate_image(self):
        item = self.get_selected_item()
        if not item: return
        img = item.image
        value = self.dilate_spinbox.value()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (value,value))
        dilate = cv2.dilate(img, kernel, iterations=3)
        child = QListWidgetItem()
        child.setText(item.text() + f"-dilate{value}")
        child.image = dilate
        self.add_image_to_grid(dilate, child)
        self.list_widget.addItem(child)

    def apply_threshold(self):
        item = self.get_selected_item()
        if not item: return
        img = item.image
        value = self.thresh_spinbox.value()
        thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,value)
        child = QListWidgetItem()
        child.setText(item.text() + f"-thresh.adap{value}")
        child.image = thresh
        self.add_image_to_grid(thresh, child)
        self.list_widget.addItem(child)

    def erode(self):
        item = self.get_selected_item()
        if not item: return
        img = item.image
        height, width = img.shape
        kh, kw = int(10), int(width * .05)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kh))
        hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
        self.contours = []
        for tag, kernel in [("v",vertical_kernel), ("h", hori_kernel)]:
            eroded = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=3)
            child = QListWidgetItem()
            child.setText(item.text() + f"{tag}-erode")
            child.image = eroded
            self.contours.append(eroded)
            self.add_image_to_grid(eroded, child)
            self.list_widget.addItem(child)

    def open_new_pdf(self, filepath=None):
        if filepath is None:
            filepath = QFileDialog.getOpenFileName()
            if filepath and filepath[0]:
                filename = os.path.basename(filepath[0])
                filepath = filepath[0]
        else:
            filename = os.path.basename(filepath)
        doc = fitz.open(filepath)
        mat = fitz.Matrix(2, 2.1)
        for page in doc.pages():
            pix = page.get_pixmap(matrix=mat)
            img = np.frombuffer(buffer=pix.samples, dtype=np.uint8).reshape(
                (pix.height, pix.width, 3)
            )
            listitem = QListWidgetItem()
            listitem.setText(filename)
            listitem.image = img
            self.list_widget.addItem(listitem)
            self.add_image_to_grid(img, listitem)





if __name__ == "__main__":
    app = QApplication([])
    window = Window()
    window.show()
    app.exec()
