from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
import fitz
import tempfile
import numpy as np
import pandas as pd
import parsel
import subprocess
import cv2



class GraphicDrawer(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setMouseTracking(True)



class ToolBar(QToolBar):



    def __init__(self):
        super().__init__()
        self.hocrbtn = QToolButton()
        self.hocrbtn.setText("hocr")
        self.addWidget(self.hocrbtn)
        self.hocrbtn.clicked.connect(self.collect_hocr)

    def collect_hocr(self):
        img = self.parent().image
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        with tempfile.NamedTemporaryFile("wb", delete=False, suffix=".png") as fdtemp:
            cv2.imwrite(fdtemp.name, grayscale)
            hocr = subprocess.check_output(
                f"tesseract {fdtemp.name} stdout --psm 11 -l eng hocr",
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            print(hocr)


class DrawTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.layout = QVBoxLayout(self)
        self.toolbar = ToolBar()
        self.layout.addWidget(self.toolbar)
        self.drawer = GraphicDrawer()
        self.layout.addWidget(self.drawer)
        self.image = None

    def loadPage(self, path, page, checked=True):
        doc = fitz.open(path)
        for i, fpage in enumerate(doc.pages()):
            if i == page:
                mat = fitz.Matrix(2.0,2.0)
                pixmap = fpage.get_pixmap(matrix=mat)
                img = np.frombuffer(buffer=pixmap.samples, dtype=np.uint8).reshape((pixmap.height, pixmap.width, 3))
                self.image = img
                self.scene = QGraphicsScene()
                self.scene.setSceneRect(0,0,img.shape[1], img.shape[0])
                item_image = QImage(img.tobytes(), img.shape[1], img.shape[0], img.shape[1]*3, QImage.Format.Format_BGR888)
                item = QGraphicsPixmapItem(QPixmap.fromImage(item_image))
                self.scene.addItem(item)
                item.setPos(QPoint(0,0))
                self.items = [item_image]
                self.drawer.setScene(self.scene)
