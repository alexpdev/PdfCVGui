import copy
import io
import math
import os
import re
import statistics
import subprocess
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import NamedTuple, OrderedDict

import cv2
import fitz
import numpy as np
import pandas as pd
import xlsxwriter
from bs4 import BeautifulSoup

class Validations:
    def __post_init__(self):
        for name, field in self.__dataclass_fields__.items():
            method = getattr(self, f"validate_{name}", None)
            setattr(self, name, method(getattr(self, name), field=field))


@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class TableCell:
    bbox: BBox
    value: str

    def __hash__(self):
        return hash(repr(self))


class CellPosition(NamedTuple):
    cell: TableCell
    row: int
    col: int


class TableObject:
    def bbox(
        self, margin: int = 0, height_margin: int = 0, width_margin: int = 0
    ) -> tuple:
        if margin != 0:
            bbox = (
                self.x1 - margin,
                self.y1 - margin,
                self.x2 + margin,
                self.y2 + margin,
            )
        else:
            bbox = (
                self.x1 - width_margin,
                self.y1 - height_margin,
                self.x2 + width_margin,
                self.y2 + height_margin,
            )

        return bbox

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def width(self) -> int:
        return self.x2 - self.x1


@dataclass
class Cell(TableObject):
    x1: int
    y1: int
    x2: int
    y2: int
    content: str = None

    @property
    def table_cell(self) -> TableCell:
        bbox = BBox(x1=self.x1, x2=self.x2, y1=self.y1, y2=self.y2)
        return TableCell(bbox=bbox, value=self.content)

    def __hash__(self):
        return hash(repr(self))


class Row(TableObject):
    def __init__(self, cells):
        if cells is None:
            raise ValueError("cells parameter is null")
        elif isinstance(cells, Cell):
            self._items = [cells]
        else:
            self._items = cells
        self._contours = []

    @property
    def items(self) -> list:
        return self._items

    @property
    def contours(self) -> list:
        return self._contours

    @property
    def nb_columns(self) -> int:
        return len(self.items)

    @property
    def x1(self) -> int:
        return min(map(lambda x: x.x1, self.items))

    @property
    def x2(self) -> int:
        return max(map(lambda x: x.x2, self.items))

    @property
    def y1(self) -> int:
        return min(map(lambda x: x.y1, self.items))

    @property
    def y2(self) -> int:
        return max(map(lambda x: x.y2, self.items))

    @property
    def v_consistent(self) -> bool:
        return all(map(lambda x: (x.y1 == self.y1) and (x.y2 == self.y2), self.items))

    def add_cells(self, cells) -> "Row":
        if isinstance(cells, Cell):
            self._items += [cells]
        else:
            self._items += cells

        return self

    def split_in_rows(self, vertical_delimiters: list) -> list:
        # Create list of tuples for vertical boundaries
        row_delimiters = [self.y1] + vertical_delimiters + [self.y2]
        row_boundaries = [(i, j) for i, j in zip(row_delimiters, row_delimiters[1:])]

        # Create new list of rows
        l_new_rows = list()
        for boundary in row_boundaries:
            cells = list()
            for cell in self.items:
                _cell = copy.deepcopy(cell)
                _cell.y1, _cell.y2 = boundary
                cells.append(_cell)
            l_new_rows.append(Row(cells=cells))

        return l_new_rows

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            try:
                assert self.items == other.items
                return True
            except AssertionError:
                return False
        return False


@dataclass
class Line(TableObject):
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def angle(self) -> float:
        delta_x = self.x2 - self.x1
        delta_y = self.y2 - self.y1

        return math.atan2(delta_y, delta_x) * 180 / np.pi

    @property
    def length(self) -> float:
        return np.sqrt(self.height**2 + self.width**2)

    @property
    def horizontal(self) -> bool:
        return self.angle % 180 == 0

    @property
    def vertical(self) -> bool:
        return self.angle % 180 == 90

    @property
    def dict(self):
        return {
            "x1": self.x1,
            "x2": self.x2,
            "y1": self.y1,
            "y2": self.y2,
            "width": self.width,
            "height": self.height,
        }

    @property
    def transpose(self) -> "Line":
        return Line(x1=self.y1, y1=self.x1, x2=self.y2, y2=self.x2)

    def reprocess(self):
        # Reallocate coordinates in proper order
        _x1 = min(self.x1, self.x2)
        _x2 = max(self.x1, self.x2)
        _y1 = min(self.y1, self.y2)
        _y2 = max(self.y1, self.y2)
        self.x1, self.x2, self.y1, self.y2 = _x1, _x2, _y1, _y2

        # Correct "almost" horizontal or vertical lines
        if abs(self.angle) <= 5:
            y_val = round((self.y1 + self.y2) / 2)
            self.y2 = self.y1 = y_val
        elif abs(self.angle - 90) <= 5:
            x_val = round((self.x1 + self.x2) / 2)
            self.x2 = self.x1 = x_val

        return self


@dataclass
class ExtractedTable:
    bbox: BBox
    title: str
    content: OrderedDict[int, list]

    @property
    def df(self) -> pd.DataFrame:
        values = [[cell.value for cell in row] for k, row in self.content.items()]
        return pd.DataFrame(values)

    def _to_worksheet(self, sheet, cell_fmt=None):
        dict_cells = dict()
        for id_row, row in self.content.items():
            for id_col, cell in enumerate(row):
                cell_pos = CellPosition(cell=cell, row=id_row, col=id_col)
                dict_cells[hash(cell)] = dict_cells.get(hash(cell), []) + [cell_pos]

        for c in dict_cells.values():
            if len(c) == 1:
                cell_pos = c.pop()
                sheet.write(cell_pos.row, cell_pos.col, cell_pos.cell.value, cell_fmt)
            else:
                sheet.merge_range(
                first_row=min(map(lambda x: x.row, c)),
                first_col=min(map(lambda x: x.col, c)),
                last_row=max(map(lambda x: x.row, c)),
                last_col=max(map(lambda x: x.col, c)),
                data=c[0].cell.value,
                cell_format=cell_fmt,
                )
        sheet.autofit()

    def html_repr(self, title: str = None) -> str:
        html = f"""{rf'<h3 style="text-align: center">{title}</h3>' if title else ''}
                   <p style=\"text-align: center\">
                       <b>Title:</b> {self.title or 'No title detected'}<br>
                       <b>Bounding box:</b> x1={self.bbox.x1}, y1={self.bbox.y1}, x2={self.bbox.x2}, y2={self.bbox.y2}
                   </p>
                   <div align=\"center\">{self.df.to_html()}</div>
                   <hr>
                """
        return html

    def __repr__(self):
        return (
            f"ExtractedTable(title={self.title}, bbox=({self.bbox.x1}, {self.bbox.y1}, {self.bbox.x2}, "
            f"{self.bbox.y2}),shape=({len(self.content)}, {len(self.content[0])}))".strip()
        )


class Table(TableObject):
    def __init__(self, rows: list):
        if rows is None:
            self._items = []
        elif isinstance(rows, Row):
            self._items = [rows]
        else:
            self._items = rows
        self._title = None

    @property
    def items(self) -> list:
        return self._items

    @property
    def title(self) -> str:
        return self._title

    def set_title(self, title: str):
        self._title = title

    @property
    def nb_rows(self) -> int:
        return len(self.items)

    @property
    def nb_columns(self) -> int:
        return self.items[0].nb_columns if self.items else 0

    @property
    def x1(self) -> int:
        return min(map(lambda x: x.x1, self.items))

    @property
    def x2(self) -> int:
        return max(map(lambda x: x.x2, self.items))

    @property
    def y1(self) -> int:
        return min(map(lambda x: x.y1, self.items))

    @property
    def y2(self) -> int:
        return max(map(lambda x: x.y2, self.items))

    def get_content(self, ocr_df: "OCRDataframe", min_confidence: int = 50) -> "Table":
        self = ocr_df.get_text_table(table=self, min_confidence=min_confidence)
        empty_rows = list()
        for idx, row in enumerate(self.items):
            if all(map(lambda c: c.content is None, row.items)):
                empty_rows.append(idx)
        for idx in reversed(empty_rows):
            self.items.pop(idx)
        empty_cols = list()
        for idx in range(self.nb_columns):
            col_cells = [row.items[idx] for row in self.items]
            if all(map(lambda c: c.content is None, col_cells)):
                empty_cols.append(idx)
        for idx in reversed(empty_cols):
            for id_row in range(self.nb_rows):
                self.items[id_row].items.pop(idx)
        unique_cells = set([cell for row in self.items for cell in row.items])
        if len(unique_cells) == 1:
            self._items = [Row(cells=self.items[0].items[0])]

        return self

    @property
    def extracted_table(self) -> ExtractedTable:
        bbox = BBox(x1=self.x1, x2=self.x2, y1=self.y1, y2=self.y2)
        content = OrderedDict(
            {
                idx: [cell.table_cell for cell in row.items]
                for idx, row in enumerate(self.items)
            }
        )
        return ExtractedTable(bbox=bbox, title=self.title, content=content)

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            try:
                assert self.items == other.items
                assert self.title == other.title
                return True
            except AssertionError:
                return False
        return False


@dataclass
class OCRDataframe:
    df: pd.DataFrame

    def page(self, page_number: int = 0) -> "OCRDataframe":
        return OCRDataframe(df=self.df[self.df["page"] == page_number])

    def get_text_cell(
        self,
        cell: Cell,
        margin: int = 0,
        page_number: int = None,
        min_confidence: int = 50,
    ) -> str:
        bbox = cell.bbox(margin=margin)

        # Filter dataframe on relevant page
        df_words = self.df[(self.df["class"] == "ocrx_word")]
        if page_number:
            df_words = df_words[df_words["page"] == page_number]
        # Filter dataframe on relevant words
        df_words = df_words[
            df_words["value"].notnull() & (df_words["confidence"] >= min_confidence)
        ]

        # Compute coordinates of intersection
        df_words = df_words.assign(
            **{
                "x1_bbox": bbox[0],
                "y1_bbox": bbox[1],
                "x2_bbox": bbox[2],
                "y2_bbox": bbox[3],
            }
        )
        df_words["x_left"] = df_words[["x1", "x1_bbox"]].max(axis=1)
        df_words["y_top"] = df_words[["y1", "y1_bbox"]].max(axis=1)
        df_words["x_right"] = df_words[["x2", "x2_bbox"]].min(axis=1)
        df_words["y_bottom"] = df_words[["y2", "y2_bbox"]].min(axis=1)

        # Filter where intersection is not empty
        df_words = df_words[df_words["x_right"] > df_words["x_left"]]
        df_words = df_words[df_words["y_bottom"] > df_words["y_top"]]

        # Compute area of word bbox and intersection
        df_words["w_area"] = (df_words["x2"] - df_words["x1"]) * (
            df_words["y2"] - df_words["y1"]
        )
        df_words["int_area"] = (df_words["x_right"] - df_words["x_left"]) * (
            df_words["y_bottom"] - df_words["y_top"]
        )

        # Filter on words where its bbox is contained in area
        df_words_contained = df_words[df_words["int_area"] / df_words["w_area"] >= 0.75]

        # Group text by parents
        df_text_parent = (
            df_words_contained.groupby("parent")
            .agg(
                x1=("x1", np.min),
                x2=("x2", np.max),
                y1=("y1", np.min),
                y2=("y2", np.max),
                value=("value", lambda x: " ".join(x)),
            )
            .sort_values(by=["y1", "x1"])
        )

        # Concatenate all lines
        return df_text_parent["value"].astype(str).str.cat(sep="\n").strip() or None

    def get_text_table(
        self, table: Table, page_number: int = None, min_confidence: int = 50
    ) -> Table:
        df_words = self.df[(self.df["class"] == "ocrx_word")]
        if page_number:
            df_words = df_words[df_words["page"] == page_number]
        # Filter dataframe on relevant words
        df_words = df_words[
            df_words["value"].notnull() & (df_words["confidence"] >= min_confidence)
        ]

        # Create dataframe containing all coordinates of Cell objects
        list_cells = [
            {
                "row": id_row,
                "col": id_col,
                "x1_w": cell.x1,
                "x2_w": cell.x2,
                "y1_w": cell.y1,
                "y2_w": cell.y2,
            }
            for id_row, row in enumerate(table.items)
            for id_col, cell in enumerate(row.items)
        ]
        df_cells = pd.DataFrame(list_cells)

        # Cartesian product between two dataframes
        df_word_cells = df_words.merge(df_cells, how="cross")

        # Compute coordinates of intersection
        df_word_cells["x_left"] = df_word_cells[["x1", "x1_w"]].max(axis=1)
        df_word_cells["y_top"] = df_word_cells[["y1", "y1_w"]].max(axis=1)
        df_word_cells["x_right"] = df_word_cells[["x2", "x2_w"]].min(axis=1)
        df_word_cells["y_bottom"] = df_word_cells[["y2", "y2_w"]].min(axis=1)

        # Filter where intersection is not empty
        df_word_cells = df_word_cells[
            df_word_cells["x_right"] > df_word_cells["x_left"]
        ]
        df_word_cells = df_word_cells[
            df_word_cells["y_bottom"] > df_word_cells["y_top"]
        ]

        # Compute area of word bbox and intersection
        df_word_cells["w_area"] = (df_word_cells["x2"] - df_word_cells["x1"]) * (
            df_word_cells["y2"] - df_word_cells["y1"]
        )
        df_word_cells["int_area"] = (
            df_word_cells["x_right"] - df_word_cells["x_left"]
        ) * (df_word_cells["y_bottom"] - df_word_cells["y_top"])

        # Filter on words where its bbox is contained in area
        df_words_contained = df_word_cells[
            df_word_cells["int_area"] / df_word_cells["w_area"] >= 0.75
        ]

        # If no words are contained, return the table
        if len(df_words_contained) == 0:
            return table

        # Group text by parent
        df_text_parent = (
            df_words_contained.groupby(["row", "col", "parent"])
            .agg(
                x1=("x1", np.min),
                x2=("x2", np.max),
                y1=("y1", np.min),
                y2=("y2", np.max),
                value=("value", lambda x: " ".join(x)),
            )
            .sort_values(by=["row", "col", "y1", "x1"])
            .groupby(["row", "col"])
            .agg(text=("value", lambda x: "\n".join(x) or None))
            .reset_index()
        )

        # Implement found values to table cells content
        for rec in df_text_parent.to_dict(orient="records"):
            table.items[rec.get("row")].items[rec.get("col")].content = (
                rec.get("text").strip() or None
            )
        return table

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            try:
                assert self.df.equals(other.df)
                return True
            except AssertionError:
                return False
        return False


@dataclass
class Document(Validations):
    src: str
    dpi: int = 200

    def validate_src(self, value, **_):
        if not isinstance(value, (str, Path, io.BytesIO, bytes)):
            raise TypeError(f"Invalid type {type(value)} for src argument")
        return value

    def validate_dpi(self, value, **_) -> int:
        if not isinstance(value, int):
            raise TypeError(f"Invalid type {type(value)} for dpi argument")
        return value

    def __post_init__(self):
        super(Document, self).__post_init__()
        # Initialize ocr_df
        self.ocr_df = None

        if isinstance(self.pages, list):
            self.pages = sorted(self.pages)

    @property
    def bytes(self) -> bytes:
        if isinstance(self.src, bytes):
            return self.src
        elif isinstance(self.src, io.BytesIO):
            self.src.seek(0)
            return self.src.read()
        elif isinstance(self.src, str):
            with io.open(self.src, "rb") as f:
                return f.read()

    @property
    def images(self) -> np.ndarray:
        raise NotImplementedError

    def extract_tables(
        self,
        ocr: "OCRInstance" = None,
        implicit_rows: bool = True,
        min_confidence: int = 50,
    ) -> dict:
        if self.ocr_df is None and ocr is not None:
            self.ocr_df = ocr.of(document=self)

        # Extract tables from document
        tables = {
            idx: TableImage(
                img=img,
                dpi=self.dpi,
                ocr_df=self.ocr_df.page(page_number=idx) if self.ocr_df else None,
                min_confidence=min_confidence,
            ).extract_tables(implicit_rows=implicit_rows)
            for idx, img in enumerate(self.images)
        }

        # If pages have been defined, modify tables keys
        if self.pages:
            tables = {self.pages[k]: v for k, v in tables.items()}

        # Reset ocr_df attribute
        self.ocr_df = None

        return tables

    def to_xlsx(
        self,
        dest,
        ocr: "OCRInstance" = None,
        implicit_rows: bool = True,
        min_confidence: int = 50,
    ) -> list:
        extracted_tables = self.extract_tables(
            ocr=ocr, implicit_rows=implicit_rows, min_confidence=min_confidence
        )
        extracted_tables = (
            {0: extracted_tables}
            if isinstance(extracted_tables, list)
            else extracted_tables
        )

        # Create workbook
        workbook = xlsxwriter.Workbook(dest, {"in_memory": True})

        # Create generic cell format
        cell_format = workbook.add_format({"align": "center", "valign": "vcenter"})
        cell_format.set_border()

        # For each extracted table, create a corresponding worksheet and populate it
        for page, tables in extracted_tables.items():
            for idx, table in enumerate(tables):
                # Create worksheet
                sheet = workbook.add_worksheet(
                    name=f"Page {page + 1} - Table {idx + 1}"
                )

                # Populate worksheet
                table._to_worksheet(sheet=sheet, cell_fmt=cell_format)

        # Close workbook
        workbook.close()

        # If destination is a BytesIO object, return it
        if isinstance(dest, io.BytesIO):
            dest.seek(0)
            return dest


class OCRInstance:
    def content(self, document: Document):
        raise NotImplementedError

    def to_ocr_dataframe(self, content) -> OCRDataframe:
        raise NotImplementedError

    def of(self, document: Document) -> OCRDataframe:
        content = self.content(document=document)
        return self.to_ocr_dataframe(content=content)


class PdfOCR(OCRInstance):
    def content(self, document: Document) -> list:
        list_pages = list()

        doc = fitz.Document(stream=document.bytes, filetype="pdf")
        for idx, page_number in enumerate(document.pages or range(doc.page_count)):
            # Get page
            page = doc.load_page(page_id=page_number)

            # Get image size and page dimensions
            img_height, img_width = list(document.images)[idx].shape[:2]
            page_height, page_width = page.mediabox.height, page.mediabox.width

            # Extract words
            list_words = list()
            for word in page.get_text("words", sort=True):
                x1, y1, x2, y2, value, block_no, line_no, word_no = word
                dict_word = {
                    "page": idx,
                    "class": "ocrx_word",
                    "id": f"word_{idx + 1}_{block_no}_{line_no}_{word_no}",
                    "parent": f"line_{idx + 1}_{block_no}_{line_no}",
                    "value": value,
                    "confidence": 99,
                    "x1": round(x1 * img_width / page_width),
                    "y1": round(y1 * img_height / page_height),
                    "x2": round(x2 * img_width / page_width),
                    "y2": round(y2 * img_height / page_height),
                }
                list_words.append(dict_word)

            # Append to list of pages
            list_pages.append(list_words)

        return list_pages

    def to_ocr_dataframe(self, content: list) -> OCRDataframe:
        # Check if any page has words
        if not content:
            return
        if min(map(len, content)) == 0:
            return None

        # Create OCRDataframe
        content_df = pd.concat(map(pd.DataFrame, content))

        return OCRDataframe(df=content_df)


@dataclass
class PDF(Document):
    pages: list = None

    def validate_pages(self, value, **_) -> list:
        if value is not None:
            if not isinstance(value, list):
                raise TypeError(f"Invalid type {type(value)} for pages argument")
            if not all(isinstance(x, int) for x in value):
                raise TypeError("All values in pages argument should be integers")
        return value

    @property
    def images(self) -> np.ndarray:
        mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
        doc = fitz.Document(stream=self.bytes, filetype="pdf")
        for page_number in self.pages or range(doc.page_count):
            page = doc.load_page(page_id=page_number)
            pix = page.get_pixmap(matrix=mat)
            img = np.frombuffer(buffer=pix.samples, dtype=np.uint8).reshape(
                (pix.height, pix.width, 3)
            )
            yield cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def extract_tables(
        self,
        ocr: "OCRInstance" = None,
        implicit_rows: bool = True,
        min_confidence: int = 50,
    ) -> dict:
        self.ocr_df = PdfOCR().of(document=self)
        return super().extract_tables(
            ocr=ocr, implicit_rows=implicit_rows, min_confidence=min_confidence
        )


class TesseractOCR(OCRInstance):
    def __init__(self, n_threads: int = 1, lang: str = "eng"):
        if isinstance(n_threads, int):
            self.n_threads = n_threads
        else:
            raise TypeError(f"Invalid type {type(n_threads)} for n_threads argument")

        if isinstance(lang, str):
            self.lang = lang
        else:
            raise TypeError(f"Invalid type {type(lang)} for lang argument")

    def hocr(self, image: np.ndarray) -> str:
        with NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_f:
            cv2.imwrite(tmp_f.name, image)
            hocr = subprocess.check_output(
                f"tesseract {tmp_f.name} stdout --psm 11 -l {self.lang} hocr",
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        while os.path.exists(tmp_f.name):
            try:
                os.remove(tmp_f.name)
            except PermissionError:
                pass
        return hocr.decode("utf-8")

    def content(self, document: Document) -> str:
        with ThreadPoolExecutor(max_workers=self.n_threads) as pool:
            hocrs = pool.map(self.hocr, document.images)

        return hocrs

    def to_ocr_dataframe(self, content: list) -> OCRDataframe:
        list_dfs = list()

        for page, hocr in enumerate(content):
            soup = BeautifulSoup(hocr, features="html.parser")
            list_elements = list()
            for element in soup.find_all(class_=True):
                d_el = {
                    "page": page,
                    "class": element["class"][0],
                    "id": element["id"],
                    "parent": element.parent.get("id"),
                    "value": re.sub(r"^(\s|\||L|_|;|\*)*$", "", element.string).strip()
                    or None
                    if element.string
                    else None,
                }
                str_conf = re.findall(r"x_wconf \d{1,2}", element["title"])
                if str_conf:
                    d_el["confidence"] = int(str_conf[0].split()[1])
                else:
                    d_el["confidence"] = np.nan
                bbox = re.findall(
                    r"bbox \d{1,4} \d{1,4} \d{1,4} \d{1,4}", element["title"]
                )[0]
                d_el["x1"], d_el["y1"], d_el["x2"], d_el["y2"] = tuple(
                    int(element) for element in re.sub(r"^bbox\s", "", bbox).split()
                )
                list_elements.append(d_el)
            list_dfs.append(pd.DataFrame(list_elements))
        return OCRDataframe(df=pd.concat(list_dfs)) if list_dfs else None


@dataclass
class TableImage:
    img: np.ndarray
    dpi: int
    ocr_df: "OCRDataframe" = None
    min_confidence: int = 50
    lines: list = None
    tables: list = None

    @property
    def white_img(self) -> np.ndarray:
        white_img = copy.deepcopy(self.img)

        # Draw white lines on detected lines
        for line in self.lines:
            cv2.rectangle(
                white_img, (line.x1, line.y1), (line.x2, line.y2), (255, 255, 255), 3
            )

        return white_img

    def extract_tables(self, implicit_rows: bool = True) -> list:
        # Detect lines in image
        try:
            h_lines, v_lines = detect_lines(
                image=self.img,
                rho=0.3,
                theta=np.pi / 180,
                threshold=10,
                minLinLength=self.dpi // 20,
                maxLineGap=self.dpi // 20,
                kernel_size=self.dpi // 10,
                ocr_df=self.ocr_df,
            )
        except:
            return []
        self.lines = h_lines + v_lines

        # Create cells from lines
        cells = get_cells(horizontal_lines=h_lines, vertical_lines=v_lines)

        # Create tables from lines
        self.tables = get_tables(cells=cells)

        # If necessary, detect implicit rows
        if implicit_rows:
            self.tables = handle_implicit_rows(
                img=self.white_img, tables=self.tables, ocr_df=self.ocr_df
            )

        # If ocr_df is available, get titles and tables content
        if self.ocr_df is not None:
            # Get title
            self.tables = get_title_tables(
                img=self.img, tables=self.tables, ocr_df=self.ocr_df
            )

            # Get content
            self.tables = [
                table.get_content(
                    ocr_df=self.ocr_df, min_confidence=self.min_confidence
                )
                for table in self.tables
            ]

        return [
            table.extracted_table
            for table in self.tables
            if table.nb_columns * table.nb_rows > 1
        ]


@dataclass
class Image(Document):
    detect_rotation: bool = False

    def validate_detect_rotation(self, value, **_) -> int:
        if not isinstance(value, bool):
            raise TypeError(f"Invalid type {type(value)} for detect_rotation argument")
        return value

    def __post_init__(self):
        self.pages = None

        super(Image, self).__post_init__()

    @property
    def images(self):
        img = cv2.imdecode(np.frombuffer(self.bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        yield fix_rotation_image(img=img) if self.detect_rotation else img

    def extract_tables(
        self,
        ocr: "OCRInstance" = None,
        implicit_rows: bool = True,
        min_confidence: int = 50,
    ) -> list:
        extracted_tables = super(Image, self).extract_tables(
            ocr=ocr, implicit_rows=implicit_rows, min_confidence=min_confidence
        )
        return extracted_tables.get(0)


def overlapping_filter(lines: list, max_gap: int = 5) -> list:
    if len(lines) == 0:
        return []
    horizontal = all(map(lambda l: l.horizontal, lines))
    if not horizontal:
        lines = [line.transpose for line in lines]
    lines = sorted(lines, key=lambda l: (l.y1, l.x1))
    previous_sequence, current_sequence = iter(lines), iter(lines)
    line_clusters = [[next(current_sequence)]]
    for previous, line in zip(previous_sequence, current_sequence):
        if line.y1 - previous.y1 > 2:
            line_clusters.append([])
        line_clusters[-1].append(line)
    final_lines = list()
    for cluster in line_clusters:
        cluster = sorted(cluster, key=lambda l: min(l.x1, l.x2))
        seq = iter(cluster)
        sub_clusters = [[next(seq)]]
        for line in seq:
            dim_2_sub_clust = max(map(lambda l: l.x2, sub_clusters[-1]))
            if line.x1 - dim_2_sub_clust <= max_gap:
                sub_clusters[-1].append(line)
            else:
                sub_clusters.append([line])

        for sub_cl in sub_clusters:
            y_value = round(
                np.average(
                    [l.y1 for l in sub_cl],
                    weights=list(map(lambda l: l.length, sub_cl)),
                )
            )
            line = Line(
                x1=min(map(lambda l: l.x1, sub_cl)),
                x2=max(map(lambda l: l.x2, sub_cl)),
                y1=y_value,
                y2=y_value,
            )

            if line.length > 0:
                final_lines.append(line)
    if not horizontal:
        final_lines = [line.transpose for line in final_lines]

    return final_lines


def remove_word_lines(lines: list, ocr_df: OCRDataframe) -> list:
    if len(lines) == 0:
        return lines

    df_words = ocr_df.df[ocr_df.df["class"] == "ocrx_word"]
    df_words = df_words[(df_words["confidence"] >= 50) | df_words["confidence"].isna()]
    df_lines = pd.DataFrame(data=[line.dict for line in lines])
    df_lines["length"] = pd.concat([df_lines["width"], df_lines["height"]], axis=1).max(
        axis=1
    )
    df_lines["vertical"] = df_lines["x1"] == df_lines["x2"]
    df_lines["line_id"] = range(len(df_lines))
    df_lines.columns = [
        "x1_line",
        "x2_line",
        "y1_line",
        "y2_line",
        "width",
        "height",
        "length",
        "vertical",
        "line_id",
    ]
    df_w_l = df_words.merge(df_lines, how="cross")
    vert_int = (
        (df_w_l["x1_line"] > df_w_l["x1"]) & (df_w_l["x1_line"] < df_w_l["x2"])
    ).astype(int) * (
        df_w_l[["y2", "y2_line"]].min(axis=1) - df_w_l[["y1", "y1_line"]].max(axis=1)
    ).clip(
        0, None
    )
    hor_int = (
        (df_w_l["y1_line"] > df_w_l["y1"]) & (df_w_l["y1_line"] < df_w_l["y2"])
    ).astype(int) * (
        df_w_l[["x2", "x2_line"]].min(axis=1) - df_w_l[["x1", "x1_line"]].max(axis=1)
    ).clip(
        0, None
    )
    df_w_l["intersection"] = (
        df_w_l["vertical"].astype(int) * vert_int
        + (1 - df_w_l["vertical"].astype(int)) * hor_int
    )
    df_inter = (
        df_w_l.groupby(["line_id", "length"])
        .agg(intersection=("intersection", np.sum))
        .reset_index()
    )
    intersecting_lines = df_inter[df_inter["intersection"] / df_inter["length"] > 0.5][
        "line_id"
    ].values.tolist()

    return [line for idx, line in enumerate(lines) if idx not in intersecting_lines]


def detect_lines(
    image: np.ndarray,
    rho: float = 1,
    theta: float = np.pi / 180,
    threshold: int = 50,
    minLinLength: int = 150,
    maxLineGap: int = 20,
    kernel_size: int = 10,
    ocr_df: OCRDataframe = None,
):
    img = image.copy()
    _, img = cv2.threshold(img, 215, 255, cv2.THRESH_TOZERO)
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
    )
    for kernel_tup, gap in [
        ((kernel_size, 1), 2 * maxLineGap),
        ((1, kernel_size), maxLineGap),
    ]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_tup)
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        hough_lines = cv2.HoughLinesP(
            mask, rho, theta, threshold, None, minLinLength, maxLineGap
        )
        if hough_lines is None:
            continue
        lines = [Line(*line[0].tolist()).reprocess() for line in hough_lines]
        lines = [line for line in lines if line.horizontal or line.vertical]
        merged_lines = overlapping_filter(lines=lines, max_gap=gap)
        if ocr_df is not None:
            merged_lines = remove_word_lines(lines=merged_lines, ocr_df=ocr_df)

        yield merged_lines

def is_contained_cell(
    inner_cell: tuple,
    outer_cell: tuple,
    percentage: float = 0.9,
) -> bool:
    if isinstance(inner_cell, tuple):
        inner_cell = Cell(*inner_cell)
    if isinstance(outer_cell, tuple):
        outer_cell = Cell(*outer_cell)
    x_left = max(inner_cell.x1, outer_cell.x1)
    y_top = max(inner_cell.y1, outer_cell.y1)
    x_right = min(inner_cell.x2, outer_cell.x2)
    y_bottom = min(inner_cell.y2, outer_cell.y2)

    if x_right < x_left or y_bottom < y_top:
        return False

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    inner_cell_area = inner_cell.height * inner_cell.width

    return intersection_area / inner_cell_area >= percentage


def merge_contours(
    contours: list, vertically: bool = True
) -> list:
    if len(contours) == 0:
        return contours
    if vertically is None:
        sorted_cnt = sorted(
            contours, key=lambda cnt: cnt.height * cnt.width, reverse=True
        )

        seq = iter(sorted_cnt)
        list_cnts = [copy.deepcopy(next(seq))]
        for cnt in seq:
            contained_cnt = [
                idx
                for idx, el in enumerate(list_cnts)
                if is_contained_cell(inner_cell=cnt, outer_cell=el, percentage=0.75)
            ]
            if len(contained_cnt) == 1:
                id = contained_cnt.pop()
                list_cnts[id].x1 = min(list_cnts[id].x1, cnt.x1)
                list_cnts[id].y1 = min(list_cnts[id].y1, cnt.y1)
                list_cnts[id].x2 = max(list_cnts[id].x2, cnt.x2)
                list_cnts[id].y2 = max(list_cnts[id].y2, cnt.y2)
            else:
                list_cnts.append(copy.deepcopy(cnt))

        return list_cnts

    # Define dimensions used to merge contours
    idx_1 = "y1" if vertically else "x1"
    idx_2 = "y2" if vertically else "x2"
    sort_idx_1 = "x1" if vertically else "y1"
    sort_idx_2 = "x2" if vertically else "y2"

    # Sort contours
    sorted_cnts = sorted(
        contours,
        key=lambda cnt: (
            getattr(cnt, idx_1),
            getattr(cnt, idx_2),
            getattr(cnt, sort_idx_1),
        ),
    )

    # Loop over contours and merge overlapping contours
    seq = iter(sorted_cnts)
    list_cnts = [copy.deepcopy(next(seq))]
    for cnt in seq:
        # If contours overlap, update current contour
        if getattr(cnt, idx_1) <= getattr(list_cnts[-1], idx_2):
            # Update current contour coordinates
            setattr(
                list_cnts[-1],
                idx_2,
                max(getattr(list_cnts[-1], idx_2), getattr(cnt, idx_2)),
            )
            setattr(
                list_cnts[-1],
                sort_idx_1,
                min(getattr(list_cnts[-1], sort_idx_1), getattr(cnt, sort_idx_1)),
            )
            setattr(
                list_cnts[-1],
                sort_idx_2,
                max(getattr(list_cnts[-1], sort_idx_2), getattr(cnt, sort_idx_2)),
            )
        else:
            list_cnts.append(copy.deepcopy(cnt))

    return list_cnts


def get_contours_cell(
    img: np.ndarray,
    cell: Cell,
    margin: int = 5,
    blur_size: int = 9,
    kernel_size: int = 15,
    merge_vertically: bool = True,
) -> list:
    height, width = img.shape[:2]
    cropped_img = img[
        max(cell.y1 - margin, 0) : min(cell.y2 + margin, height),
        max(cell.x1 - margin, 0) : min(cell.x2 + margin, width),
    ]
    height_cropped, width_cropped = cropped_img.shape[:2]
    if height_cropped <= 0 or width_cropped <= 0:
        return []
    blur = cv2.GaussianBlur(cropped_img, (blur_size, blur_size), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 30
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilate = cv2.dilate(thresh, kernel, iterations=4)
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    list_cnts_cell = list()
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        x = x + cell.x1 - margin
        y = y + cell.y1 - margin
        contour_cell = Cell(x, y, x + w, y + h)
        list_cnts_cell.append(contour_cell)
    contours = merge_contours(contours=list_cnts_cell, vertically=merge_vertically)

    return contours


def normalize_table_cells(cluster_cells: list) -> list:
    l_bound_tb = min(map(lambda c: c.x1, cluster_cells))
    r_bound_tb = max(map(lambda c: c.x2, cluster_cells))
    up_bound_tb = min(map(lambda c: c.y1, cluster_cells))
    low_bound_tb = max(map(lambda c: c.y2, cluster_cells))

    normalized_cells = list()
    list_y_values = list()
    for cell in sorted(cluster_cells, key=lambda c: (c.y1, c.y2)):
        if (cell.x1 - l_bound_tb) / (r_bound_tb - l_bound_tb) <= 0.02:
            cell.x1 = l_bound_tb
        if (r_bound_tb - cell.x2) / (r_bound_tb - l_bound_tb) <= 0.02:
            cell.x2 = r_bound_tb
        close_upper_values = [
            y
            for y in list_y_values
            if abs(y - cell.y1) / (low_bound_tb - up_bound_tb) <= 0.02
        ]
        if len(close_upper_values) == 1:
            cell.y1 = close_upper_values.pop()
        else:
            list_y_values.append(cell.y1)
        close_lower_values = [
            y
            for y in list_y_values
            if abs(y - cell.y2) / (low_bound_tb - up_bound_tb) <= 0.02
        ]
        if len(close_lower_values) == 1:
            cell.y2 = close_lower_values.pop()
        else:
            list_y_values.append(cell.y2)

        normalized_cells.append(cell)

    return normalized_cells


def create_rows_table(cluster_cells: list) -> Table:
    normalized_cells = normalize_table_cells(cluster_cells=cluster_cells)
    sorted_cells = sorted(normalized_cells, key=lambda c: (c.y1, c.x1, c.y2))
    seq = iter(sorted_cells)
    list_rows = [Row(cells=next(seq))]
    for cell in seq:
        if cell.y1 < list_rows[-1].y2:
            list_rows[-1].add_cells(cells=cell)
        else:
            list_rows.append(Row(cells=cell))

    return Table(rows=list_rows)


def handle_vertical_merged_cells(row: Row) -> list:
    cells = sorted(row.items, key=lambda c: (c.x1, c.y1, c.x2))
    seq = iter(cells)
    cols = [[next(seq)]]
    for cell in seq:
        if abs(cols[-1][0].x1 - cell.x1) / row.width >= 0.02:
            cols.append([])
        cols[-1].append(cell)
    nb_rows = max(map(len, cols))
    v_delimiters = [
        statistics.mean(
            [(col[idx].y1 + col[idx].y2) / 2 for col in cols if len(col) == nb_rows]
        )
        for idx in range(nb_rows)
    ]
    recomputed_columns = list()
    for col in cols:
        if len(col) == nb_rows:
            recomputed_columns.append(col)
        else:
            _col = list()
            for delim in v_delimiters:
                intersecting_cells = [
                    cell for cell in col if cell.y1 <= delim <= cell.y2
                ]
                closest_cell = (
                    intersecting_cells.pop()
                    if intersecting_cells
                    else Cell(x1=col[0].x1, x2=col[0].x2, y1=int(delim), y2=int(delim))
                )
                _col.append(closest_cell)
            recomputed_columns.append(_col)
    new_rows = [
        Row(cells=[col[idx] for col in recomputed_columns]) for idx in range(nb_rows)
    ]

    return new_rows


def handle_horizontal_merged_cells(table: Table) -> Table:
    nb_cols = max([row.nb_columns for row in table.items])
    list_delimiters = [
        [(cell.x1 + cell.x2) / 2 for cell in row.items]
        for row in table.items
        if row.nb_columns == nb_cols
    ]
    average_delimiters = [
        statistics.mean([delim[idx] for delim in list_delimiters])
        for idx in range(nb_cols)
    ]
    new_rows = list()
    for row in table.items:
        if row.nb_columns == nb_cols:
            new_rows.append(row)
        else:
            _cells = list()
            for delim in average_delimiters:
                intersecting_cells = [
                    cell for cell in row.items if cell.x1 <= delim <= cell.x2
                ]
                closest_cell = (
                    intersecting_cells.pop()
                    if intersecting_cells
                    else Cell(
                        x1=int(delim),
                        x2=int(delim),
                        y1=row.items[0].y1,
                        y2=row.items[0].y1,
                    )
                )
                _cells.append(closest_cell)
            new_rows.append(Row(cells=_cells))

    return Table(rows=new_rows)


def handle_merged_cells(table: Table) -> Table:
    table_v_merged = Table(
        rows=[
            _row
            for row in table.items
            for _row in handle_vertical_merged_cells(row=row)
        ]
    )

    table_h_merged = handle_horizontal_merged_cells(table=table_v_merged)

    return table_h_merged


def create_word_image(img: np.ndarray, ocr_df: OCRDataframe) -> np.ndarray:
    df_words = ocr_df.df[ocr_df.df["class"] == "ocrx_word"]

    words_img = np.zeros(img.shape, dtype=np.uint8)
    words_img.fill(255)

    for word in df_words.to_dict(orient="records"):
        # Get cropped image of word
        cropped_img = img[
            word.get("y1") : word.get("y2"), word.get("x1") : word.get("x2")
        ]

        words_img[
            word.get("y1") : word.get("y2"), word.get("x1") : word.get("x2")
        ] = cropped_img

    return words_img


def handle_implicit_rows_table(img: np.ndarray, table: Table) -> Table:
    if table.nb_columns * table.nb_rows <= 1:
        return table

    list_splitted_rows = list()
    for row in table.items:
        if not row.v_consistent:
            list_splitted_rows.append(row)
            continue
        contours = get_contours_cell(
            img=copy.deepcopy(img),
            cell=row,
            margin=-5,
            blur_size=5,
            kernel_size=5,
            merge_vertically=True,
        )
        vertical_delimiters = sorted(
            [
                round((cnt_1.y2 + cnt_2.y1) / 2)
                for cnt_1, cnt_2 in zip(contours, contours[1:])
            ]
        )
        list_splitted_rows += row.split_in_rows(vertical_delimiters=vertical_delimiters)

    return Table(rows=list_splitted_rows)


def handle_implicit_rows(
    img: np.ndarray, tables: list, ocr_df: OCRDataframe = None
) -> list:
    words_img = create_word_image(img=img, ocr_df=ocr_df) if ocr_df is not None else img
    tables_implicit_rows = [
        handle_implicit_rows_table(img=words_img, table=table) for table in tables
    ]

    return tables_implicit_rows


def adjacent_cells(cell_1: Cell, cell_2: Cell) -> bool:
    overlapping_y = min(cell_1.y2, cell_2.y2) - max(cell_1.y1, cell_2.y1)
    diff_x = min(
        abs(cell_1.x2 - cell_2.x1),
        abs(cell_1.x1 - cell_2.x2),
        abs(cell_1.x1 - cell_2.x1),
        abs(cell_1.x2 - cell_2.x2),
    )
    if overlapping_y > 5 and diff_x / max(cell_1.width, cell_2.width) <= 0.05:
        return True
    overlapping_x = min(cell_1.x2, cell_2.x2) - max(cell_1.x1, cell_2.x1)
    diff_y = min(
        abs(cell_1.y2 - cell_2.y1),
        abs(cell_1.y1 - cell_2.y2),
        abs(cell_1.y1 - cell_2.y1),
        abs(cell_1.y2 - cell_2.y2),
    )
    if overlapping_x > 5 and diff_y / max(cell_1.height, cell_2.height) <= 0.05:
        return True

    return False


def cluster_cells_in_tables(cells: list) -> list:
    clusters = list()
    for i in range(len(cells)):
        for j in range(i, len(cells)):
            if not any(map(lambda cl: {i, j}.intersection(cl) == {i, j}, clusters)):
                adjacent = adjacent_cells(cells[i], cells[j])
                if adjacent:
                    matching_clusters = [
                        idx
                        for idx, cl in enumerate(clusters)
                        if {i, j}.intersection(cl)
                    ]
                    if matching_clusters:
                        cl_id = matching_clusters.pop()
                        clusters[cl_id] = clusters[cl_id].union({i, j})
                    else:
                        clusters.append({i, j})
    list_table_cells = [[cells[idx] for idx in cl] for cl in clusters]

    return list_table_cells


def get_tables(cells: list) -> list:
    list_cluster_cells = cluster_cells_in_tables(cells=cells)
    tables = [
        create_rows_table(cluster_cells=cluster_cells)
        for cluster_cells in list_cluster_cells
    ]
    list_tables = [handle_merged_cells(table=table) for table in tables]

    return list_tables


def get_cells_dataframe(
    horizontal_lines: list, vertical_lines: list
) -> pd.DataFrame:
    default_df = pd.DataFrame(columns=["x1", "x2", "y1", "y2", "width", "height"])
    df_h_lines = (
        pd.DataFrame(map(lambda l: l.dict, horizontal_lines))
        if horizontal_lines
        else default_df.copy()
    )
    df_v_lines = (
        pd.DataFrame(map(lambda l: l.dict, vertical_lines))
        if vertical_lines
        else default_df.copy()
    )
    df_h_lines_cp = df_h_lines.copy()
    df_h_lines_cp.columns = ["x1_", "x2_", "y1_", "y2_", "width_", "height_"]
    cross_h_lines = df_h_lines.merge(df_h_lines_cp, how="cross")
    cross_h_lines = cross_h_lines[cross_h_lines["y1"] < cross_h_lines["y1_"]]
    cross_h_lines["l_corresponds"] = (
        cross_h_lines["x1"] - cross_h_lines["x1_"] / cross_h_lines["width"]
    ).abs() <= 0.02
    cross_h_lines["r_corresponds"] = (
        cross_h_lines["x2"] - cross_h_lines["x2_"] / cross_h_lines["width"]
    ).abs() <= 0.02
    cross_h_lines["l_contained"] = (
        (cross_h_lines["x1"] <= cross_h_lines["x1_"])
        & (cross_h_lines["x1_"] <= cross_h_lines["x2"])
    ) | (
        (cross_h_lines["x1_"] <= cross_h_lines["x1"])
        & (cross_h_lines["x1"] <= cross_h_lines["x2_"])
    )
    cross_h_lines["r_contained"] = (
        (cross_h_lines["x1"] <= cross_h_lines["x2_"])
        & (cross_h_lines["x2_"] <= cross_h_lines["x2"])
    ) | (
        (cross_h_lines["x1_"] <= cross_h_lines["x2"])
        & (cross_h_lines["x2"] <= cross_h_lines["x2_"])
    )
    matching_condition = (
        cross_h_lines["l_corresponds"] | cross_h_lines["l_contained"]
    ) & (cross_h_lines["r_corresponds"] | cross_h_lines["r_contained"])
    cross_h_lines = cross_h_lines[matching_condition]
    cross_h_lines["x1_bbox"] = cross_h_lines[["x1", "x1_"]].max(axis=1)
    cross_h_lines["x2_bbox"] = cross_h_lines[["x2", "x2_"]].min(axis=1)
    cross_h_lines["y1_bbox"] = cross_h_lines["y1"]
    cross_h_lines["y2_bbox"] = cross_h_lines["y1_"]
    df_bbox = cross_h_lines[["x1_bbox", "y1_bbox", "x2_bbox", "y2_bbox"]].reset_index()
    df_bbox["h_margin"] = (
        pd.concat(
            [
                (df_bbox["x2_bbox"] - df_bbox["x1_bbox"]) * 0.05,
                pd.Series(5.0, index=range(len(df_bbox))),
            ],
            axis=1,
        )
        .max(axis=1)
        .round()
    )
    df_bbox_v = df_bbox.merge(df_v_lines, how="cross")
    horizontal_cond = (
        df_bbox_v["x1_bbox"] - df_bbox_v["h_margin"] <= df_bbox_v["x1"]
    ) & (df_bbox_v["x2_bbox"] + df_bbox_v["h_margin"] > +df_bbox_v["x1"])
    df_bbox_v = df_bbox_v[horizontal_cond]
    df_bbox_v["overlapping"] = df_bbox_v[["y2", "y2_bbox"]].min(axis=1) - df_bbox_v[
        ["y1", "y1_bbox"]
    ].max(axis=1)
    df_bbox_v = df_bbox_v[
        df_bbox_v["overlapping"] / (df_bbox_v["y2_bbox"] - df_bbox_v["y1_bbox"]) >= 0.8
    ]
    df_bbox_delimiters = df_bbox_v.groupby(
        ["index", "x1_bbox", "x2_bbox", "y1_bbox", "y2_bbox"]
    ).agg(
        dels=(
            "x1",
            lambda x: [bound for bound in zip(sorted(x), sorted(x)[1:])] or None,
        )
    )

    try:
        # Create new cells based on vertical delimiters
        df_bbox_delimiters = (
            df_bbox_delimiters[df_bbox_delimiters["dels"].notnull()]
            .explode(column="dels")
            .reset_index()
        )
        df_bbox_delimiters[["del1", "del2"]] = pd.DataFrame(
            df_bbox_delimiters.dels.tolist(), index=df_bbox_delimiters.index
        )
        df_bbox_delimiters["x1_bbox"] = df_bbox_delimiters["del1"]
        df_bbox_delimiters["x2_bbox"] = df_bbox_delimiters["del2"]

        # Reformat output dataframe
        df_cells = df_bbox_delimiters[["x1_bbox", "y1_bbox", "x2_bbox", "y2_bbox"]]
        df_cells.columns = ["x1", "y1", "x2", "y2"]

        return df_cells.reset_index()
    except ValueError:
        return pd.DataFrame(columns=["index", "x1", "y1", "x2", "y2"])


def deduplicate_cells_vertically(df_cells: pd.DataFrame) -> pd.DataFrame:
    orig_cols = df_cells.columns

    df_cells = df_cells.sort_values(by=["x1", "x2", "y1", "y2"])
    df_cells["cell_rk"] = df_cells.groupby(["x1", "x2", "y1"]).cumcount()
    df_cells = df_cells[df_cells["cell_rk"] == 0]

    df_cells = df_cells.sort_values(
        by=["x1", "x2", "y2", "y1"], ascending=[True, True, True, False]
    )
    df_cells["cell_rk"] = df_cells.groupby(["x1", "x2", "y2"]).cumcount()
    df_cells = df_cells[df_cells["cell_rk"] == 0]

    return df_cells[orig_cols]


def deduplicate_nested_cells(df_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate nested cells in order to keep the smallest ones
    :param df_cells: dataframe containing cells
    :return: dataframe containing cells after deduplication of the nested ones
    """
    df_cells["width"] = df_cells["x2"] - df_cells["x1"]
    df_cells["height"] = df_cells["y2"] - df_cells["y1"]
    df_cells["area"] = df_cells["width"] * df_cells["height"]

    df_cells_cp = df_cells.copy()
    df_cells_cp.columns = [
        "index_",
        "x1_",
        "y1_",
        "x2_",
        "y2_",
        "width_",
        "height_",
        "area_",
    ]

    df_cross_cells = df_cells.reset_index().merge(df_cells_cp, how="cross")
    df_cross_cells = df_cross_cells[df_cross_cells["index"] != df_cross_cells["index_"]]
    df_cross_cells = df_cross_cells[df_cross_cells["area"] <= df_cross_cells["area_"]]

    df_cross_cells["x_left"] = df_cross_cells[["x1", "x1_"]].max(axis=1)
    df_cross_cells["y_top"] = df_cross_cells[["y1", "y1_"]].max(axis=1)
    df_cross_cells["x_right"] = df_cross_cells[["x2", "x2_"]].min(axis=1)
    df_cross_cells["y_bottom"] = df_cross_cells[["y2", "y2_"]].min(axis=1)

    df_cross_cells["int_area"] = (
        df_cross_cells["x_right"] - df_cross_cells["x_left"]
    ) * (df_cross_cells["y_bottom"] - df_cross_cells["y_top"])

    df_cross_cells["contained"] = (
        (df_cross_cells["x_right"] >= df_cross_cells["x_left"])
        & (df_cross_cells["y_bottom"] >= df_cross_cells["y_top"])
        & (df_cross_cells["int_area"] / df_cross_cells["area"] >= 0.9)
    )

    df_cross_cells["overlapping_x"] = (
        df_cross_cells["x_right"] - df_cross_cells["x_left"]
    )
    df_cross_cells["overlapping_y"] = (
        df_cross_cells["y_bottom"] - df_cross_cells["y_top"]
    )
    df_cross_cells["diff_x"] = pd.concat(
        [
            (df_cross_cells["x2"] - df_cross_cells["x1_"]).abs(),
            (df_cross_cells["x1"] - df_cross_cells["x2_"]).abs(),
            (df_cross_cells["x1"] - df_cross_cells["x1_"]).abs(),
            (df_cross_cells["x2"] - df_cross_cells["x2_"]).abs(),
        ],
        axis=1,
    ).min(axis=1)
    df_cross_cells["diff_y"] = pd.concat(
        [
            (df_cross_cells["y1"] - df_cross_cells["y1_"]).abs(),
            (df_cross_cells["y2"] - df_cross_cells["y1_"]).abs(),
            (df_cross_cells["y1"] - df_cross_cells["y2_"]).abs(),
            (df_cross_cells["y2"] - df_cross_cells["y2_"]).abs(),
        ],
        axis=1,
    ).min(axis=1)
    condition_adjacent = (
        (df_cross_cells["overlapping_y"] > 5)
        & (
            df_cross_cells["diff_x"] / df_cross_cells[["width", "width_"]].max(axis=1)
            <= 0.05
        )
    ) | (
        (df_cross_cells["overlapping_x"] > 5)
        & (
            df_cross_cells["diff_y"] / df_cross_cells[["height", "height_"]].max(axis=1)
            <= 0.05
        )
    )
    df_cross_cells["adjacent"] = condition_adjacent
    df_cross_cells["redundant"] = (
        df_cross_cells["contained"] & df_cross_cells["adjacent"]
    )
    redundant_cells = (
        df_cross_cells[df_cross_cells["redundant"]]["index_"]
        .drop_duplicates()
        .values.tolist()
    )
    df_final_cells = df_cells.drop(labels=redundant_cells)
    return df_final_cells


def deduplicate_cells(df_cells: pd.DataFrame) -> pd.DataFrame:
    df_cells = deduplicate_cells_vertically(df_cells=df_cells)
    df_cells_final = deduplicate_nested_cells(df_cells=df_cells)

    return df_cells_final


def get_cells(horizontal_lines: list, vertical_lines: list) -> list:
    df_cells = get_cells_dataframe(
        horizontal_lines=horizontal_lines, vertical_lines=vertical_lines
    )
    df_cells_dedup = deduplicate_cells(df_cells=df_cells)
    cells = [
        Cell(x1=row["x1"], x2=row["x2"], y1=row["y1"], y2=row["y2"])
        for row in df_cells_dedup.to_dict(orient="records")
    ]

    return cells


def rotate_img(img: np.ndarray, angle: float) -> np.ndarray:
    height, width = img.shape
    image_center = (width // 2, height // 2)
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    bound_w = int(height * abs(rotation_mat[0, 1]) + width * abs(rotation_mat[0, 0]))
    bound_h = int(height * abs(rotation_mat[0, 0]) + width * abs(rotation_mat[0, 1]))
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]
    rotated_img = cv2.warpAffine(
        img,
        rotation_mat,
        (bound_w, bound_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return rotated_img


def straightened_img(img: np.ndarray) -> tuple:
    edges = cv2.Canny(img, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=math.pi / 180.0,
        threshold=50,
        minLineLength=100,
        maxLineGap=20,
    )
    median_angle = float(
        np.median(
            [math.degrees(math.atan2(y2 - y1, x2 - x1)) for [[x1, y1, x2, y2]] in lines]
        )
    )
    if median_angle % 180 != 0:
        # Rotate the image to straighten it
        straight_img = rotate_img(img=img, angle=median_angle)
        return straight_img, median_angle

    return img, 0.0


def upside_down(img: np.ndarray) -> bool:
    height, width = img.shape

    top_area = 0
    bottom_area = 0

    for id_col in range(width):
        # Identify black pixels in the column
        black_pixels = np.where(img[:, id_col] == 0)[0]

        if black_pixels.size > 0:
            # Add column area to the totals
            top_area += np.amin(black_pixels)
            bottom_area += height - np.amax(black_pixels) - 1

    return top_area > bottom_area


def fix_rotation_image(img: np.ndarray) -> np.ndarray:
    denoised = cv2.medianBlur(img.copy(), 3)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Straighten image
    straight_thresh, angle = straightened_img(img=thresh)

    # Identify if the straightened image is upside down
    is_inverted = upside_down(img=straight_thresh)

    # Compute final rotation angle to apply
    rotation_angle = angle + 180 * int(is_inverted)

    if rotation_angle % 360 > 0:
        return rotate_img(img=img, angle=rotation_angle)

    return img


def get_title_tables(
    img: np.ndarray, tables: list, ocr_df: OCRDataframe, margin: int = 5
) -> list:
    height, width = img.shape[:2]
    if len(tables) == 0:
        return []
    sorted_tables = sorted(tables, key=lambda tb: (tb.y1, tb.x1, tb.x2))
    seq = iter(sorted_tables)
    tb_cl = [[next(seq)]]
    for tb in seq:
        if tb.y1 > tb_cl[-1][-1].y2:
            tb_cl.append([])
        tb_cl[-1].append(tb)
    final_tables = list()
    for id_cl, cluster in enumerate(tb_cl):
        x_delimiters = (
            [10]
            + [
                round((tb_1.x2 + tb_2.x1) / 2)
                for tb_1, tb_2 in zip(cluster, cluster[1:])
            ]
            + [width - 10]
        )
        x_bounds = [
            (del_1, del_2) for del_1, del_2 in zip(x_delimiters, x_delimiters[1:])
        ]
        y_bounds = (
            max([tb.y2 for tb in tb_cl[id_cl - 1]]) if id_cl > 0 else 0,
            min([tb.y1 for tb in cluster]),
        )
        for id_tb, table in enumerate(cluster):
            cell_title = Cell(
                x1=x_bounds[id_tb][0],
                x2=x_bounds[id_tb][1],
                y1=y_bounds[0],
                y2=y_bounds[1],
            )
            contours = get_contours_cell(
                img=copy.deepcopy(img),
                cell=cell_title,
                margin=0,
                blur_size=5,
                kernel_size=9,
            )
            title = (
                ocr_df.get_text_cell(cell=contours[-1], margin=margin)
                if contours
                else None
            )

            table.set_title(title=title)
            final_tables.append(table)

    return final_tables


def cluster_to_table(cluster_cells) -> Table:
    """
    Convert a cell cluster to a Table object
    :param cluster_cells: list of cells that form a table
    :return: table with rows inferred from table cells
    """
    # Get list of vertical delimiters
    v_delims = sorted(list(set([y_val for cell in cluster_cells for y_val in [cell.y1, cell.y2]])))

    # Get list of horizontal delimiters
    h_delims = sorted(list(set([x_val for cell in cluster_cells for x_val in [cell.x1, cell.x2]])))

    # Create rows and cells
    list_rows = list()
    for y_top, y_bottom in zip(v_delims, v_delims[1:]):
        list_cells = list()
        for x_left, x_right in zip(h_delims, h_delims[1:]):
            # Create default cell
            default_cell = Cell(x1=x_left, y1=y_top, x2=x_right, y2=y_bottom)

            # Check cells that contain the default cell
            containing_cells = sorted([c for c in cluster_cells
                                       if is_contained_cell(inner_cell=default_cell, outer_cell=c, percentage=0.9)],
                                      key=lambda c: c.area)

            # Append either a cell that contain the default cell, or the default cell itself
            list_cells.append(containing_cells.pop(0) if containing_cells else default_cell)

        list_rows.append(Row(cells=list_cells))

    return Table(rows=list_rows)


def normalize_table_cells(cluster_cells):
    """
    Normalize cells from table cells
    :param cluster_cells: list of cells that form a table
    :return: list of normalized cells
    """
    # Compute table shape
    width = max(map(lambda c: c.x2, cluster_cells)) - min(map(lambda c: c.x1, cluster_cells))
    height = max(map(lambda c: c.y2, cluster_cells)) - min(map(lambda c: c.y1, cluster_cells))

    # Get list of existing horizontal values
    h_values = sorted(list(set([x_val for cell in cluster_cells for x_val in [cell.x1, cell.x2]])))
    # Compute delimiters by grouping close values together
    h_delims = [round(np.mean(h_group)) for h_group in
                np.split(h_values, np.where(np.diff(h_values) >= min(width * 0.02, 10))[0] + 1)]

    # Get list of existing vertical values
    v_values = sorted(list(set([y_val for cell in cluster_cells for y_val in [cell.y1, cell.y2]])))
    # Compute delimiters by grouping close values together
    v_delims = [round(np.mean(v_group)) for v_group in
                np.split(v_values, np.where(np.diff(v_values) >= min(height * 0.02, 10))[0] + 1)]

    # Normalize all cells
    normalized_cells = list()
    for cell in cluster_cells:
        normalized_cell = Cell(x1=sorted(h_delims, key=lambda d: abs(d - cell.x1)).pop(0),
                               x2=sorted(h_delims, key=lambda d: abs(d - cell.x2)).pop(0),
                               y1=sorted(v_delims, key=lambda d: abs(d - cell.y1)).pop(0),
                               y2=sorted(v_delims, key=lambda d: abs(d - cell.y2)).pop(0))
        # Check if cell is not empty
        if cell.area > 0:
            normalized_cells.append(normalized_cell)

    return normalized_cells
