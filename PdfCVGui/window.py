from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

from PdfCVGui.viewer import ViewerTab
from PdfCVGui.preproc import PreProcTab
from PdfCVGui.drawer import DrawTab
from PdfCVGui.style import template_style


class Window(QMainWindow):
    """Window object."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self.layout = QVBoxLayout()
        self.central = QTabWidget(parent=self)
        self.preprocTab = PreProcTab()
        self.viewerTab = ViewerTab()
        self.drawTab = DrawTab()
        self.central.addTab(self.viewerTab, "Viewer")
        self.central.addTab(self.preprocTab, "PreProcess")
        self.central.addTab(self.drawTab, "Drawing")
        self.setCentralWidget(self.central)
        self.viewerTab.transferPage.connect(self.transferPdf)
        self.setObjectName('MainWindow')
        self.resize(1200,900)

    def transferPdf(self, path, number, checked):
        self.preprocTab.load_page(path, number, checked)
        self.drawTab.loadPage(path, number, checked)



def execute():
    app = QApplication([])
    app.setStyleSheet(template_style)
    window = Window()
    window.show()
    try:
        app.exec()
    except Exception as e:
        print(e)
