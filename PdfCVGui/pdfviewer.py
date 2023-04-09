from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtPdf import *

DEFAULT_COLOR = Qt.black
DEFAULT_THICKNESS = 2

class MyGraphicsView(QGraphicsView):

    currentPageChanged = Signal(int, int)
    totalPagesChanged = Signal(int)

    def __init__(self, scene):
        super().__init__(scene)
        self.path = None
        self.page_number = 1
        self.page_count = None
        self.document = None
        self.pixitem = None
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

    def load_pdf(self, pdf_filename):
        self.document = QPdfDocument()
        self.document.load(pdf_filename)
        self.path = pdf_filename
        self.page_number = 0
        self.page_count = self.document.pageCount()
        self.currentPageChanged.emit(self.page_number + 1, self.page_count)
        self.totalPagesChanged.emit(self.page_count)
        self.updateBackground()

    def next_page(self):
        if self.page_number < self.page_count - 1:
            self.page_number += 1
            self.currentPageChanged.emit(self.page_number + 1, self.page_count)
            self.updateBackground()

    def prev_page(self):
        if self.page_number > 0:
            self.page_number -= 1
            self.currentPageChanged.emit(self.page_number + 1, self.page_count)
            self.updateBackground()

    def updateBackground(self):
        if self.document is not None:
            if self.pixitem:
                self.scene().removeItem(self.pixitem)
            psize = self.document.pagePointSize(self.page_number)
            image = self.document.render(self.page_number, psize.toSize())
            pixmap = QPixmap.fromImage(image)
            self.pixitem = QGraphicsPixmapItem(pixmap)
            self.scene().addItem(self.pixitem)
            self.pixitem.setPos(QPoint(0,0))

    def resizeEvent(self, event):
        self.updateBackground()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Right or event.key() == Qt.Key_N:
            self.next_page()
        elif event.key() == Qt.Key_Left or event.key() == Qt.Key_P:
            self.prev_page()
