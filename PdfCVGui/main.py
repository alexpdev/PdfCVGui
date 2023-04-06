import os
import sys
from pathlib import Path
import docx
import fitz
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_ROW_HEIGHT_RULE, WD_TABLE_ALIGNMENT
from docx.shared import Pt
from PdfCVGui.img import get_table_images, get_cell_images
from PdfCVGui.ocr import Image, TesseractOCR
import itertools

if "MEI" not in str(Path(__file__).parent):
    PDFS = Path(__file__).parent.parent / "pdfs"
    DATA = Path(__file__).parent.parent.parent / "data"
    RESULTS = Path(__file__).parent.parent / "results"
    TEMP = Path(__file__).parent.parent / "temp"
else:
    PDFS = Path(sys.executable).parent / "pdfs"
    RESULTS = Path(sys.executable).parent / "results"
    TEMP = Path(sys.executable).parent / "temp"

def main():
    if not os.path.exists(PDFS):
        return
    for pdf in PDFS.iterdir():
        name = pdf.name
        if pdf.suffix != ".pdf":
            continue
        if not os.path.exists(RESULTS):
            os.mkdir(RESULTS)
        images = get_images_from_pdf(pdf)
        results = get_table_images(images)
        for result in results:
            r = get_cell_images(result)
        convert_to_docx(results, name)
    # shutil.rmtree(TEMP)
    return True

def run_main(filepath):
    images = get_images_from_pdf(filepath)
    results = get_table_images(images)
    for result in results:
        r = get_cell_images(result)
    convert_to_docx(results, os.path.basename(filepath))

def get_images_from_pdf(path):
    name = path.name
    doc = fitz.open(str(path))
    mat = fitz.Matrix(2.0, 2.0)
    lst = []
    if not os.path.exists(TEMP):
        os.mkdir(TEMP)
    for i, page in enumerate(doc.pages()):
        pix = page.get_pixmap(matrix=mat)
        filename = TEMP / (name + str(i) + ".png")
        pix.save(str(filename))
        lst.append(str(filename))
    return lst

def convert_to_docx(results, name):
    ocr = TesseractOCR(3, "eng")
    doc = docx.Document()
    for imgs in results:
        for i in imgs:
            image = Image(i)
            tables = image.extract_tables(ocr=ocr)
            for table in tables:
                add_table(doc, table)
    doc.save(str(RESULTS / (name + ".docx")))

def add_table(doc, table):
    data = {}
    maxr = maxc = 0
    for r, row in table.content.items():
        if r > maxr:
            maxr = r
        for c, col in enumerate(row):
            if c > maxc:
                maxc = c
            data.setdefault(hash(col), [])
            data[hash(col)] += [(col, r, c)]
    tab = doc.add_table(cols=maxc+1, rows=maxr+ 1)
    tab.style = "Table Grid"
    tab.alignment = WD_TABLE_ALIGNMENT.LEFT
    tab.autofit = True
    for row in tab.rows:
        row.height_rule = WD_ROW_HEIGHT_RULE.AT_LEAST
        row.height = Pt(12)
    for c in data.values():
        if len(c) == 1:
            pos = c.pop()
            if pos[0].value is not None:
                tab.cell(pos[1], pos[2]).text = pos[0].value
        else:
            for c1, c2 in itertools.pairwise(c):
                if [c2[1], c2[2]] in gen_neighbors(c1):
                    row1 = tab.rows[c1[1]]
                    row2 = tab.rows[c2[1]]
                    cell1 = row1.cells[c1[2]]
                    cell2 = row2.cells[c2[2]]
                    try:
                        cell1.merge(cell2)
                    except:
                        pass
            if c1[0].value is not None:
                cell1.text = c1[0].value
    p = doc.add_paragraph()
    p.add_run("")

def gen_neighbors(c1):
    for x,y in [(1,0), (0,1), (-1,0), (-1,1), (1,-1), (0,-1), (-1,-1), (1,1)]:
        yield [c1[1] + x, c1[2] + y]