import string

theme = {
    "_1":  "#000",
    "_2":  "#FFF",
    "_3":  "#473422",
    "_4":  "#323232",
    "_5":  "#676767",
    "_6":  "#445361",
    "_7":  "#282299",
    "_8":  "#FF0000",
    "_9":  "#AA0000",
    "_10": "#550000",
    "_11": "#FFFF00",
    "_12": "#CACA00",
    "_13": "#777700",
    "_14": "#00FF00",
    "_15": "#00AA00",
    "_16": "#005500",
    "_17": "#1862C9",
    "_18": "#148CF2",
    "_19": "#14506E",
    "_20": "#19232D",
    "_21": "#F0F0F0",
    "_22": "#72414B"
}



style = """
QWidget {
    background-color: $_4;
    color: $_2;
}
QGraphicsView {
    background-color: $_2;
}
QTreeWidget,
QPushButton,
QListWidget {
    border: 1px solid $_7;
    background: $_5;
}
QToolButton {
  background-color: transparent;
  border: 1px solid transparent;
  border-radius: 4px;
  margin: 2px 0px 2px 0px;
  padding: 2px;
}
#ViewerBar {
    spacing: 12px;
}
QToolButton:checked {
  background-color: transparent;
  border: 1px solid $_17;
}
QToolButton:pressed {
  margin: 1px;
  background-color: transparent;
  border: 1px solid $_17;
}
QToolButton:disabled {
  border: none;
}
QToolButton:hover {
  border: 1px solid $_18;
}

QToolButton[popupMode="0"] {
  padding-right: 2px;
}


QToolButton[popupMode="1"]::menu-button {
  border: none;
}

QToolButton[popupMode="1"]::menu-button:hover {
  border: none;
  border-left: 1px solid $_18;
  border-radius: 0;
}

QToolButton[popupMode="2"] {
  padding-right: 2px;
}

QToolButton::menu-button {
  padding: 2px;
  border-radius: 4px;
  border: 1px solid $_4;
  width: 12px;
  outline: none;
}

QToolButton::menu-button:hover {
  border: 1px solid $_18;
}

QToolButton::menu-button:checked:hover {
  border: 1px solid $_18;
}
QToolButton::menu-indicator {
  height: 8px;
  width: 8px;
  top: 0;
  left: -2px;
}
QToolButton::menu-arrow {
  height: 8px;
  width: 8px;
}
QLabel {
    background-color: transparent;
}
QListWidget::selected,
QTreeWidget::selected {
    background-color: $_6;
}
QPushButton,
QToolButton {
    background-color: $_17;
    padding: 2px 5px;
}
QPushButton::hover,
QToolButton::hover {
    background-color: $_13;
}
QToolButton {
    margin-left: 2px;
    margin-right: 2px;
}
QTabWidget {
  padding: 2px;
  selection-background-color: $_4;
}

QTabWidget QWidget {
  border-radius: 4px;
}

QTabWidget::pane {
  border: 1px solid $_4;
  border-radius: 4px;
}

QTabWidget::pane:selected {
  background-color: $_4;
  border: 1px solid $_17;
}

QTabBar {
  qproperty-drawBase: 0;
  border-radius: 4px;
  margin: 3px;
  padding: 6px;
}

QTabBar::tab:top:!selected {
  border-bottom: 2px solid $_20;
  margin-top: 6px;
}

QTabBar::tab:top {
  background-color: $_4;
  color: $_21;
  margin-left: 3px;
  padding-left: 6px;
  padding-right: 6px;
  padding-top: 5px;
  padding-bottom: 5px;
  border-bottom: 3px solid $_4;
}

QTabBar::tab:top:selected {
  background-color: $_6;
  color: $_21;
  border-bottom: 3px solid $_17;
}

QTabBar::tab:top:!selected:hover {
  border: 1px solid $_18;
  border-bottom: 3px solid $_18;
  padding-left: 4px;
  padding-right: 4px;
}
QSplitter {
  background-color: $_4;
  spacing: 0px;
  padding: 0px;
  margin: 0px;
}

QSplitter::handle {
  background-color: $_4;
  border: 0px solid $_20;
  spacing: 0px;
  padding: 1px;
  margin: 0px;
}

QSplitter::handle:hover {
  background-color: $_5;
}

QSplitter::handle:horizontal {
  width: 5px;
}

QSplitter::handle:vertical {
  height: 5px;
}
QComboBox {
  border: 1px solid $_4;
  border-radius: 4px;
  selection-background-color: $_17;
  padding-left: 4px;
  min-height: 1.5em;
}

QComboBox QAbstractItemView {
  border: 1px solid $_4;
  border-radius: 0;
  background-color: $_20;
  selection-background-color: $_17;
}

QComboBox QAbstractItemView:hover {
  background-color: $_20;
  color: $_21;
}

QComboBox QAbstractItemView:selected {
  background: $_17;
  color: $_4;
}

QComboBox QAbstractItemView:alternate {
  background: $_20;
}

QComboBox:disabled {
  background-color: $_20;
  color: $_5;
}

QComboBox:hover {
  border: 1px solid $_18;
}

QComboBox:focus {
  border: 1px solid $_17;
}

QComboBox:on {
  selection-background-color: $_17;
}

QComboBox::indicator {
  border: none;
  border-radius: 0;
  background-color: transparent;
  selection-background-color: transparent;
  color: transparent;
  selection-color: transparent;
}

QComboBox::indicator:alternate {
  background: $_20;
}

QComboBox::item:alternate {
  background: $_20;
}

QComboBox::item:checked {
  font-weight: bold;
}
QScrollArea {
    border: 1px solid $_7;
}
QScrollArea QWidget {
    background-color: $_5;
}
QGraphicsView {
  background-color: $_21;
  border: 1px solid $_4;
  color: $_21;
  border-radius: 4px;
}
QGraphicsView:hover{
  border: 1px solid $_22;
}
QSpinBox,
QComboBox {
    border: 1px solid $_13;
    background-color: $_4;
    padding: 4px;
}
QSpinBox:hover,
QComboBox:hover {
    border: 1px solid $_11;
}
QGroupBox {
    border: 1px solid $_6;
    padding: 1px;
}
"""

template = string.Template(style)
template_style = template.substitute(theme)
