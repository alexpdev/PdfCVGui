import re
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
from queue import Queue
import fitz
import tempfile
import numpy as np
import pandas as pd
import parsel
import subprocess
import cv2

DEFAULT_COLOR = Qt.black
DEFAULT_THICKNESS = 2

class PdfPage(QGraphicsPixmapItem):

    def __init__(self):
        super().__init__(self)
        self.blockEdit = False
        self.ongoingEdit = False
        self.drawPoints = Queue()
        self.tempPoints = Queue()
        self.startPos = 0
        self.endPos = 0
        self.ongoingEdit = False
        self.blockEdit = False
        self.formPoints = []
        self.drawIndicators = []
        self.lastZoomFactor = -1
        self.isDraft = False
        self.penDraw = False
        self.avPressure = 1

    def paint(self, painter, option, widget):
        res = super().paint(painter, option, widget)
        if self.tempPoints.qsize() > 0:
            if self.scene().mode == self.scene().Mode.draw:
                try:
                    penSize = self.avPressure / self.tempPoints.qsize() * 1.6 * self.freeHandSize
                    tempList = list(zip(*list(self.tempPoints.queue)))
                except Exception as e:
                    print(e)
                painter.setPen(QPen(QColor(*self.freeHandColor), penSize))
                painter.drawPolyline(self.templist[0])

            if self.scene().mode == self.scene().Mode.line:
                painter.setPen(QPen(DEFAULT_COLOR, self.markerSize))
                painter.setRenderHint(QPainter.SmoothPixmapTransform)
                l = list(self.tempPoints.queue)
                j = [i[0] for i in l]
                painter.drawPolyline(j)
        return res

class PathItem(QGraphicsItem):
    def __init__(self, pos):
        super().__init__()
        self.path = QPainterPath(pos)
        self.color = DEFAULT_COLOR
        self.thickness = DEFAULT_THICKNESS

    def boundingRect(self):
        return self.path.boundingRect()

    def paint(self, painter, option, widget):
        painter.setPen(QPen(self.color, self.thickness))
        painter.drawPath(self.path)

    def lineTo(self, point):
        self.path.lineTo(point)
        self.update()

    def moveTo(self, point):
        self.path.moveTo(point)


class LineItem(QGraphicsLineItem):
    def __init__(self, start, end):
        super().__init__()
        self.start = start
        self.end = end
        self._line = QLineF(start, end)
        self.setLine(self._line)
        self.setPen(QPen(QColor(0,0,0), 3, Qt.PenStyle.SolidLine))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)

    def setStartPos(self, pos):
        self.start = pos
        self.updateLine()

    def setEndPos(self, pos):
        self.end = pos
        self.updateLine()

    def updateLine(self):
        if self._line.p1() != self.start:
            self._line.setP1(self.start)
        if self._line.p2() != self.end:
            self._line.setP2(self.end)
        self.setLine(self._line)


class Scene(QGraphicsScene):

    class Mode:
        select = 1
        draw = 2
        line = 3
        eraser = 4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image = None
        self.current = None
        self.dragging = None
        self.last = None
        self.ongoingEdit = False
        self.mode = self.Mode.select

    def mousePressEvent(self, event):
        try:
            pos = event.scenePos()
            for view in self.views():
                item = self.itemAt(pos, view.transform())
                imageItem = view.imageItem
                break
            if item and item != imageItem:
                return super().mousePressEvent(event)
            if self.selectedItems():
                self.clearSelection()
            if self.mode == self.Mode.draw:
                self.path = PathItem(pos)
                self.prev_path = self.addItem(self.path)
                return
            elif self.mode == self.Mode.line:
                self.line = LineItem(pos, pos)
                self.addItem(self.line)
                return
            elif self.mode == self.Mode.eraser:
                self.startEraser(event.scenePos())
                return
            super().mousePressEvent(event)
        except Exception as e:
            print(e)

    def mouseMoveEvent(self, event):
        try:
            if event.buttons() == Qt.LeftButton:
                pos = event.scenePos()
                if self.selectedItems():
                    return super().mouseMoveEvent(event)
                if self.mode == self.Mode.draw and self.path:
                    self.path.lineTo(pos)
                    if self.prev_path:
                        self.removeItem(self.prev_path)
                    self.prev_path = self.scene().addItem(self.path)
                    self.update()
                    return
                if self.mode == self.Mode.line and self.line:
                    self.line.setEndPos(pos)
                    return
                if self.mode == self.Mode.eraser:
                    self.updateEraserPoints(event.scenePos())
                    return
            super().mouseMoveEvent(event)
        except Exception as e:
            print(e)

    def mouseReleaseEvent(self, event):
        try:
            if self.dragging:
                super().mouseReleaseEvent(event)
                return
            if self.mode == self.Mode.eraser:
                self.stopEraser(event.scenePos())
            else:
                self.path = None
                self.line = None
                self.current = None
                self.dragged = None
            print(event.pos())
        except Exception as e:
            print(e)

    def removeItemAtPoints(self):
        for view in self.views():
            if view.imageItem:
                break
        for item in self.items():
            if item != view.imageItem:
                for point in self.eraserPoints:
                    if item.contains(point):
                        self.removeItem(item)

    def applyEraser(self):
        self.removeItemAtPoints()

    def stopEraser(self, qpos):
        self.ongoingEdit = False
        self.applyEraser()

    def updateEraserPoints(self, qpos):
        self.eraserPoints.append(qpos)

    def startEraser(self, qpos):
        self.ongoingEdit = True
        self.eraserPoints = []

class GraphicDrawer(QGraphicsView):

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent=parent)
        self._scene = scene
        self.image = None
        self.imageItem = None
        self.setRenderHint(QPainter.Antialiasing)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

    def setImage(self, image):
        self.image = image
        width = self.width() - 10
        scaled = self.image.scaledToWidth(width)
        pixmap = QPixmap.fromImage(scaled)
        item = QGraphicsPixmapItem(pixmap)
        self.scene().setSceneRect(0,0,pixmap.width(), pixmap.height())
        self.scene().addItem(item)
        item.setPos(QPoint(0,0))
        self.imageItem = item

    def resizeEvent(self, event):
        width = self.width() - 10
        if self.image is not None:
            scaled = self.image.scaledToWidth(width)
            pixmap = QPixmap.fromImage(scaled)
            self.scene().removeItem(self.imageItem)
            self.scene().setSceneRect(0,0,pixmap.width(),pixmap.height())
            item = QGraphicsPixmapItem(pixmap)
            self.scene().addItem(item)
            self.imageItem = item

class ToolBar(QToolBar):

    hocrd = Signal(list)

    def __init__(self):
        super().__init__()
        self.hocrbtn = QToolButton()
        self.hocrbtn.setText("hocr")
        self.addWidget(self.hocrbtn)
        self.hocrbtn.clicked.connect(self.collect_hocr)
        self.selectButton = QToolButton()
        self.selectButton.setText("select")
        self.addWidget(self.selectButton)
        self.lineButton = QToolButton()
        self.lineButton.setText("Line")
        self.addWidget(self.lineButton)
        self.drawButton = QToolButton()
        self.drawButton.setText("Draw")
        self.addWidget(self.drawButton)
        self.eraserButton = QToolButton()
        self.eraserButton.setText("Erase")
        self.addWidget(self.eraserButton)
        self.clearButton = QToolButton()
        self.clearButton.setText("Clear")
        self.addWidget(self.clearButton)


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
            with open("a.html", 'wb') as d:
                d.write(hocr)
            selector = parsel.Selector(hocr.decode('utf8'))
            words = []
            for elem in selector.xpath("//span[@class='ocrx_word']"):
                bbox = elem.xpath("./@title").get()
                matches = re.findall(r"bbox (\d+\s\d+\s\d+\s\d+);", bbox)
                for match in matches:
                    arr = [int(i) for i in match.split(" ")]
                    words.append(arr)
            lines = []
            for elem in selector.xpath("//span[@class='ocr_line']"):
                bbox = elem.xpath("./@title").get()
                matches = re.findall(r"bbox (\d+\s\d+\s\d+\s\d+);", bbox)
                for match in matches:
                    arr = [int(i) for i in match.split(" ")]
                    if arr not in words:
                        lines.append(arr)
            pars = []
            for elem in selector.xpath("//p[@class='ocr_par']"):
                bbox = elem.xpath("./@title").get()
                matches = re.findall(r"bbox (\d+\s\d+\s\d+\s\d+);", bbox)
                for match in matches:
                    arr = [int(i) for i in match.split(" ")]
                    if arr not in words and arr not in lines:
                        pars.append(arr)
            divs = []
            for elem in selector.xpath("//div[@class='ocr_carea']"):
                bbox = elem.xpath("./@title").get()
                matches = re.findall(r"bbox (\d+\s\d+\s\d+\s\d+);", bbox)
                for match in matches:
                    arr = [int(i) for i in match.split(" ")]
                    if arr not in words and arr not in lines and arr not in pars:
                        divs.append(arr)
            self.hocrd.emit([words, lines, pars, divs])






class DrawTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.layout = QVBoxLayout(self)
        self.toolbar = ToolBar()
        self.layout.addWidget(self.toolbar)
        self.scene = Scene()
        self.drawer = GraphicDrawer(self.scene)
        self.layout.addWidget(self.drawer)
        self.toolbar.lineButton.clicked.connect(self.activate_line_mode)
        self.toolbar.selectButton.clicked.connect(self.activate_select_mode)
        self.toolbar.drawButton.clicked.connect(self.activate_draw_mode)
        self.toolbar.eraserButton.clicked.connect(self.activate_eraser)
        self.toolbar.clearButton.clicked.connect(self.clear_items)
        self.toolbar.hocrd.connect(self.transmit_bboxes)
        self.image = None
        self.items = []

    def transmit_bboxes(self, bboxes):
        hfact = self.drawer.imageItem.boundingRect().height() / self.image.shape[0]
        wfact = self.drawer.imageItem.boundingRect().width() / self.image.shape[1]
        print(hfact, wfact)
        for cat in bboxes:
            for bbox in cat:
                groups = [[[0,1],[0,3]], [[2,1],[2,3]], [[0,1],[2,1]], [[0,3],[2,3]]]
                for group in groups:
                    g1, g2 = group
                    x1, y1 = g1
                    x2, y2 = g2
                    point1 = QPointF(bbox[x1] * hfact, bbox[y1]* hfact)
                    point2 = QPointF(bbox[x2] * hfact, bbox[y2]* hfact)
                    point1 = self.drawer.imageItem.mapToScene(point1)
                    point2 = self.drawer.imageItem.mapToScene(point2)
                    item = LineItem(point1, point2)
                    self.scene.addItem(item)
                    self.items.append(item)

    def clear_items(self):
        imageitem = self.drawer.imageItem
        for item in self.scene.items():
            if item != imageitem:
                self.scene.removeItem(item)

    def activate_eraser(self):
        self.drawer.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.scene.mode = self.scene.Mode.eraser

    def activate_line_mode(self):
        self.drawer.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.scene.mode = self.scene.Mode.line

    def activate_select_mode(self):
        self.drawer.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.scene.mode = self.scene.Mode.select

    def activate_draw_mode(self):
        self.drawer.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.scene.mode = self.scene.Mode.draw

    def loadPage(self, path, page, checked=True):
        doc = fitz.open(path)
        for i, fpage in enumerate(doc.pages()):
            if i == page:
                mat = fitz.Matrix(2.0,2.0)
                pixmap = fpage.get_pixmap(matrix=mat)
                img = np.frombuffer(buffer=pixmap.samples, dtype=np.uint8).reshape((pixmap.height, pixmap.width, 3))
                self.image = img
                item_image = QImage(img.tobytes(), img.shape[1], img.shape[0], img.shape[1]*3, QImage.Format.Format_BGR888)
                self.drawer.setImage(item_image)
