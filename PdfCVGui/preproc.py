import os
import fitz
import cv2
import numpy as np
import xxhash
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
from PdfCVGui.img import extract_tables, extract_cells, table_image
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


class Memo:
    def __init__(self, func):
        self.func = func
        self.cache = {}
        self.limit = 10

    def __call__(self, img, width=None, height=None):
        digest = xxhash.xxh32_digest(img.tostring())
        if (digest, width, height) in self.cache:
            return self.cache[digest, width, height]
        else:
            result = self.func(img, width=width, height=height)
            self.cache[(digest, width, height)] = result
            return result

@Memo
def cvToImage(img, width=None, height=None) -> QImage:
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB, img)
    image = QImage(
        img.tobytes(), img.shape[1], img.shape[0],
        img.shape[1]*3, QImage.Format.Format_BGR888
    )
    if width:
        return image.scaledToWidth(
            width, Qt.TransformationMode.SmoothTransformation)
    if height:
        return image.scaledToHeight(
            height, Qt.TransformationMode.SmoothTransformation)
    return image


class ScrollArea(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._widget = QWidget()
        self.layout = QVBoxLayout(self._widget)
        self.setWidget(self._widget)
        self.setWidgetResizable(True)
        self.setViewportMargins(3, 3, 3, 3)
        self.hlayouts = []
        self.labels = []

    def resizeEvent(self, event):
        self.update_background()
        super().resizeEvent(event)

    def paintEvent(self, event):
        self.update_background()
        super().paintEvent(event)

    def update_background(self):
        rect = self.viewport().rect()
        self._widget.setFixedWidth(self.width() - 8)
        width = rect.width() - 6
        for label in self.labels:
            image = cvToImage(label.arr, width=width//2)
            label.setPixmap(QPixmap.fromImage(image))

    def remove_label(self, label):
        index = self.labels.index(label)
        self.labels.pop(index)
        45

    def add_label(self, label):
        if len(self.hlayouts) > 0:
            last_layout = self.hlayouts[-1]
            if last_layout.count() <= 1:
                last_layout.addWidget(label)
                self.labels.append(label)
                return
            for i in range(last_layout.count()):
                item = last_layout.itemAt(i)
                if not item or not item.widget():
                    last_layout.insertWidget(label, i)
                    self.labels.append(label)
                    return
        hlayout = QHBoxLayout()
        self.hlayouts.append(hlayout)
        self.layout.addLayout(hlayout)
        hlayout.addWidget(label)
        self.labels.append(label)
        return




class PreProcTab(QWidget):
    """Window object."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self.splitter = QSplitter()
        self.leftside = QWidget()
        self.rightside = QWidget()
        self.splitter.addWidget(self.leftside)
        self.splitter.addWidget(self.rightside)
        self.splitter.setStretchFactor(0,3)
        self.splitter.setStretchFactor(1,4)
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.splitter)
        self.toolbar = QToolBar(self)
        self.toolbar2 = QToolBar(self)
        self.list_widget = QListWidget(self)
        self.scroll_area = ScrollArea(self)
        self.contours_button = QPushButton("Contours")
        vlayout = QVBoxLayout(self.leftside)
        self.open_button = QPushButton("Open")
        vlayout.addWidget(self.open_button)
        vlayout.addWidget(self.list_widget)
        self.vlayout = QVBoxLayout(self.rightside)
        self.hlayout = QHBoxLayout()
        self.vlayout.addWidget(self.toolbar)
        self.vlayout.addWidget(self.toolbar2)
        self.vlayout.addWidget(self.scroll_area)
        self.row = self.col = 0
        self.setObjectName('ImageProc')
        self.open_button.clicked.connect(self.open_new_pdf)
        self.add_toolbar_buttons()
        self.list_widget.currentItemChanged.connect(self.highlight_label)
        self.list_widget.sizePolicy().setHorizontalPolicy(QSizePolicy.Policy.Fixed)
        self.list_widget.installEventFilter(self)
        self.setToolbar2()

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
            nimg[rect[1]+1:rect[3]-1, rect[0]+1:rect[2]-1] = 255
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
        items = self.get_selected_items()
        positions = []
        for item in items:
            self.scroll_area.remove_label(item.image)
            i = self.list_widget.indexFromItem(item)
            positions.append(i)
        for j in sorted(positions, reverse=True):
            self.list_widget.takeItem(j)

    def highlight_label(self, item):
        for i in range(self.list_widget.count()):
            list_item = self.list_widget.item(i)
            list_item.label.setStyleSheet("border: 3px solid #DDA;")
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
        image = cvToImage(img)
        pixmap = QPixmap.fromImage(image)
        label.setPixmap(pixmap)
        label.setScaledContents(True)
        item.label = label
        item.image = img
        label.item = item
        label.arr = img
        label.cache = image
        self.scroll_area.add_label(label)

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
        self.toolbar.addSeparator()
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
        self.toolbar.addSeparator()
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
        self.toolbar.addSeparator()
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
        self.toolbar.addSeparator()
        self.erode_button = QToolButton(self)
        self.erode_button.setText("erode")
        self.erode_button.clicked.connect(self.erode)
        self.toolbar.addWidget(self.erode_button)
        self.toolbar.addSeparator()
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
        self.toolbar.addSeparator()
        self.extract_button = QToolButton()
        self.extract_button.setText("OCR")
        self.extract_button.clicked.connect(self.ocr_img)
        self.toolbar.addWidget(self.extract_button)
        self.toolbar.addSeparator()
        self.thresh_std_layout.setSpacing(6)
        self.thresh_std_layout.setContentsMargins(3,1,3,1)
        self.dilate_layout.setSpacing(6)
        self.dilate_layout.setContentsMargins(3,1,3,1)
        self.blur_layout.setSpacing(6)
        self.blur_layout.setContentsMargins(3,1,3,1)
        self.thresh_layout.setSpacing(6)
        self.thresh_layout.setContentsMargins(3,1,3,1)


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

    def open_new_pdf(self):
        filepath = QFileDialog.getOpenFileName()
        if filepath and filepath[0]:
            filepath = filepath[0]
            self.load_page(filepath)

    def load_page(self, path, pageno=0, checked=False):
        filename = os.path.basename(path)
        doc = fitz.open(path)
        mat = fitz.Matrix(2.0, 2.0)
        for i, page in enumerate(doc.pages()):
            if checked and pageno != i:
                continue
            pix = page.get_pixmap(matrix=mat)
            img = np.frombuffer(buffer=pix.samples, dtype=np.uint8).reshape(
                (pix.height, pix.width, 3)
            )
            img = cv2.copyMakeBorder(img, 4,4,4,4, cv2.BORDER_CONSTANT, value=[255,255,255])
            listitem = QListWidgetItem()
            listitem.setText(filename + f"-p{i}")
            self.list_widget.addItem(listitem)
            self.add_image_to_grid(img, listitem)


    def setToolbar2(self):
        self.img_table_finder_btn = QToolButton(self)
        self.img_table_finder_btn.setText("Image Table Finder")
        self.img_cell_finder_btn = QToolButton(self)
        self.img_cell_finder_btn.setText("Image Cell Finder")
        self.toolbar2.addWidget(self.img_table_finder_btn)
        self.toolbar2.addWidget(self.img_cell_finder_btn)
        self.toolbar2.addSeparator()
        self.pd_table_finder_btn = QToolButton(self)
        self.pd_table_finder_btn.setText("DataFrame Table Finder")
        self.pd_cell_finder_btn = QToolButton(self)
        self.pd_cell_finder_btn.setText("DataFrame Cell Finder")
        self.toolbar2.addWidget(self.pd_table_finder_btn)
        self.toolbar2.addWidget(self.pd_cell_finder_btn)
        self.pd_table_finder_btn.clicked.connect(self.find_table_dataframes)
        self.img_table_finder_btn.clicked.connect(self.find_image_tables)
        self.pd_cell_finder_btn.clicked.connect(self.find_cells_dataframes)
        self.img_cell_finder_btn.clicked.connect(self.find_image_cells)

    def find_table_dataframes(self):
        item = self.get_selected_item()

    def find_image_tables(self):
        item = self.get_selected_item()
        bounding_rects = extract_tables(item.image)
        for i, rect in enumerate(bounding_rects):
            image = table_image(item.image, rect)
            child = QListWidgetItem()
            child.setText(item.text() + f"-extracted_table{i}")
            child.image = image
            self.add_image_to_grid(image, child)
            self.list_widget.addItem(child)

    def find_cells_dataframes(self):
        item = self.get_selected_item()

    def find_image_cells(self):
        item = self.get_selected_item()
        images = extract_cells(item.image)
        for i, image in enumerate(images):
            child = QListWidgetItem()
            child.setText(item.text() + f"-extract_cells{i}")
            child.image = image
            self.add_image_to_grid(image, child)
            self.list_widget.addItem(child)
