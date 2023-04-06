from pathlib import Path
import os
import logging
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
from PdfCVGui.pdfviewer import MyGraphicsView
from PdfCVGui.main import run_main

PARENT = os.path.dirname(__file__)
ROOT = os.path.dirname(PARENT)
RESULTS = os.path.join(ROOT, "results")
PDFS = os.path.join(ROOT, "pdfs")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ToolBar(QToolBar):

    def __init__(self):
        super().__init__()
        self.setObjectName("ViewerBar")
        self.backbtn = QAction()
        self.backbtn.setText("Back")
        self.nextbtn = QAction()
        self.nextbtn.setText("Next")
        self.transferbtn = QAction()
        self.transferbtn.setText("Transfer")
        self.singlePage = QToolButton()
        self.singlePage.setText("Single Page")
        self.label = QLabel("Page -/-")
        self.singlePage.setCheckable(True)
        self.singlePage.setChecked(True)
        self.addAction(self.transferbtn)
        self.addWidget(self.singlePage)
        self.addSeparator()

        self.addAction(self.backbtn)
        self.addWidget(self.label)
        self.addAction(self.nextbtn)
        self.setOrientation(Qt.Horizontal)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)

    def change_page(self, current, total):
        text = f"Page {current}/{total}"
        self.label.setText(text)

class ViewerTab(QWidget):
    """Window object."""

    showPdf = Signal(str)
    transferPage = Signal(str, int, bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self.layout         = QVBoxLayout(self)
        self.splitter       = QSplitter(Qt.Horizontal)


        self.hlay           = QHBoxLayout()
        self.file_btn       = QPushButton("Add File(s)")
        self.folder_btn     = QPushButton("Add Folder")
        self.widg           = QWidget()
        self.vlay           = QVBoxLayout(self.widg)
        self.files_widget   = QListWidget()
        self.process_button = QPushButton("Process")

        self.widg2          = QWidget()
        self.vlay2          = QVBoxLayout(self.widg2)
        self.toolbar        = ToolBar()
        self.pdfscene       = QGraphicsScene()
        self.pdfview        = MyGraphicsView(self.pdfscene)

        self.hlay.addWidget(self.file_btn)
        self.hlay.addWidget(self.folder_btn)
        self.layout.addWidget(self.splitter)
        self.vlay.addLayout(self.hlay)
        self.vlay.addWidget(self.files_widget)
        self.vlay.addWidget(self.process_button)
        self.vlay2.addWidget(self.toolbar)
        self.vlay2.addWidget(self.pdfview)
        self.splitter.addWidget(self.widg)
        self.splitter.addWidget(self.widg2)
        self.splitter.setStretchFactor(0,1)
        self.splitter.setStretchFactor(1,4)

        self.process_button.clicked.connect(self.process_all_files)
        self.files_widget.currentItemChanged.connect(self.show_pdf)
        self.file_btn.clicked.connect(self.get_pdf_files)
        self.folder_btn.clicked.connect(self.get_pdfs_in_folder)
        self.toolbar.nextbtn.triggered.connect(self.pdfview.next_page)
        self.toolbar.backbtn.triggered.connect(self.pdfview.prev_page)
        self.toolbar.transferbtn.triggered.connect(self.transfer_page)
        self.pdfview.currentPageChanged.connect(self.toolbar.change_page)
        self.showPdf.connect(self.pdfview.load_pdf)
        self.load_icons_images()
        self.load_default_folder()

    def load_default_folder(self):
        if os.path.exists(PDFS):
            self.load_folder(Path(PDFS))

    def process_all_files(self):
        if not os.path.exists(RESULTS):
            os.mkdir(RESULTS)
        for i in range(self.files_widget.count()):
            item = self.files_widget.item(i)
            path = Path(item.path)
            run_main(path)

    def transfer_page(self):
        page_no = self.pdfview.page_number
        path = self.pdfview.path
        checked = self.toolbar.singlePage.isChecked()
        self.transferPage.emit(path, page_no, checked)

    def get_pdf_files(self):
        path = QFileDialog.getOpenFileNames(self, "Choose File(s)", filter="PDF Files(*.pdf)")
        for i in path[0]:
            name = os.path.basename(i)
            item = QListWidgetItem(self.pdficon, name, self.files_widget, 0)
            item.path = i
            self.files_widget.addItem(item)

    def get_pdfs_in_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Choose Folder")
        if not path:
            return
        self.load_folder(path)

    def load_folder(self, path):
        for filename in os.listdir(path):
            if os.path.splitext(filename)[1] == ".pdf":
                fullpath = os.path.join(path, filename)
                item = QListWidgetItem(self.pdficon, filename, self.files_widget, 0)
                item.path = fullpath
                self.files_widget.addItem(item)

    def load_icons_images(self):
        self.pdficon = QIcon(os.path.join(ROOT, "assets", "pdf.png"))
        self.load_icons = []
        self.load_icon_index = 0
        self.loading = False
        self.loading_path = None
        self.timer = QTimer()
        imagepath = os.path.join(ROOT, "assets", "image")
        for base in os.listdir(imagepath):
            fullpath = os.path.join(imagepath, base)
            icon = QIcon(fullpath)
            self.load_icons.append(icon)

    def iter_icon(self):
        if not self.loading or not self.loading_path:
            return
        index = self.load_icon_index % len(self.load_icons)
        icon = self.load_icons[index]
        for i in range(self.pre_widget.count()):
            item = self.pre_widget.item(i)
            if item.text() != os.path.basename(self.loading_path):
                continue
            item.setIcon(icon)
            self.load_icon_index += 1
            self.timer.singleShot(400, self.iter_icon_pre)
            break
        return

    def show_pdf(self, item, prev):
        path = item.path
        self.showPdf.emit(path)

    def set_load_icon(self, path):
        self.loading = True
        self.loading_path = path
        self.iter_icon()

    def set_pdf_icon(self, path):
        self.loading = False
        self.loading_path = None
        self.load_icon_index = 0
        icon = QIcon(os.path.join(PARENT, "assets", "pdf.png"))
        base = os.path.basename(path)
        return icon, base

    def set_pdf_icon_pre(self, path):
        icon, base = self.set_pdf_icon(path)
        for i in range(self.pre_widget.count()):
            item = self.pdf_widget.item(i)
            if base == item.text():
                item.setIcon(icon)
                break

    def setThreadName(self, first, last):
        self.thread.chosen_name = first + last
