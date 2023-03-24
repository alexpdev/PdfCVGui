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
        self.scroll_widget = QWidget()
        self.leftside = QWidget()
        vlayout = QVBoxLayout(self.leftside)
        self.open_button = QPushButton("Open")
        vlayout.addWidget(self.open_button)
        vlayout.addWidget(self.list_widget)
        self.list_widget.setHeaderHidden(True)
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
        self.tree_widget.currentItemChanged.connect(self.highlight_label)
        self.resize(900,800)

    def clear_label_selection(self):
        for i in range(self.tree_widget.topLevelItemCount()):
            item = self.tree_widget.topLevelItem(i)
            def deselect_children(root):
                label = root.label
                label.setStyleSheet("border: solid transparent 0px;")
                for j in range(root.childCount()):
                    child = root.child(j)
                    deselect_children(child)
            deselect_children(item)

    def highlight_label(self, item):
        self.clear_label_selection()
        label = item.label
        label.setStyleSheet("border: 3px solid cyan;")



    def get_selected_item(self):
        items = self.tree_widget.selectedItems()
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
        image = cvToImage(img).scaledToWidth(400)
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
        self.thresh_spinbox.setRange(0,25)
        self.thresh_button = QToolButton()
        self.thresh_button.setText("Adap. Thresh.")
        self.thresh_button.clicked.connect(self.apply_threshold)
        self.thresh_layout.addWidget(self.thresh_button)
        self.thresh_layout.addWidget(self.thresh_spinbox)
        self.toolbar.addWidget(self.thresh_groupbox)
        self.dilate_groupbox = QGroupBox()
        self.dilate_layout = QHBoxLayout(self.dilate_groupbox)
        self.dilate_spinbox = QSpinBox()
        self.dilate_spinbox.setRange(0,25)
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

    def dilate_image(self):
        item = self.get_selected_item()
        if not item: return
        img = item.image
        value = self.dilate_spinbox.value()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (value,value))
        dilate = cv2.dilate(img, kernel, iterations=4)
        child = QListWidgetItem()
        child.setText(item.text() + f"-dilate")
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
        child.setText(item.text() + f"-thresh")
        child.image = thresh
        self.add_image_to_grid(thresh, child)
        self.list_widget.addItem(child)

    def erode(self):
        item = self.get_selected_item()
        if not item: return
        img = item.image
        height, width = img.shape
        vertical_kernel_height = math.ceil(height*0.3)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_height))
        horizontal_kernel_width = math.ceil(width*0.3)
        hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_width, 1))
        for tag, kernel in [("v",vertical_kernel), ("h", hori_kernel)]:
            eroded = cv2.erode(img, kernel, iterations=3)
            child = QListWidgetItem()
            child.setText(item.text() + f"{tag}-erode")
            child.image = eroded
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
        mat = fitz.Matrix(2.0, 2.0)
        for page in doc.pages():
            pix = page.get_pixmap(matrix=mat)
            img = np.frombuffer(buffer=pix.samples, dtype=np.uint8).reshape(
                (pix.height, pix.width, 3)
            )
            treeitem = QListWidgetItem()
            treeitem.setText(0, filename)
            treeitem.image = img
            self.tree_widget.addTopLevelItem(treeitem)
            self.add_image_to_grid(img, treeitem)







if __name__ == "__main__":
    app = QApplication([])
    window = Window()
    window.show()
    app.exec()
