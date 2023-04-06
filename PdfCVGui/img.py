import os
import cv2

def extract_cells(image):
    blurred = cv2.GaussianBlur(image, (17,17), 0, 0)
    img_bin = cv2.adaptiveThreshold(
        ~blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2
    )
    horizontal = img_bin.copy()
    SCALE = 10
    width, height = horizontal.shape
    try:
        hkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(width / SCALE), 1))
    except:
        return []
    hopened = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, hkernel)
    vkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(height / SCALE)))
    vopened = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, vkernel)

    hdilated = cv2.dilate(hopened, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1)))
    vdilated = cv2.dilate(vopened, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60)))
    mask = hdilated + vdilated
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    perimeter_lengths = [cv2.arcLength(c, True) for c in contours]
    epsilons = [0.05 * p for p in perimeter_lengths]
    approx_polys = [cv2.approxPolyDP(c, e, True) for c, e in zip(contours, epsilons)]
    # approx_rects = [p for p in approx_polys if len(p) == 4]
    brects = [cv2.boundingRect(a) for a in approx_polys]
    MINWIDTH = 20
    MINHEIGHT = 10
    brects = [r for r in brects if MINWIDTH < r[2] and MINHEIGHT < r[3]]
    largest_rect = max(brects, key=lambda r: r[2] * r[3])
    brects = [b for b in brects if b is not largest_rect]

    cells = [c for c in brects]
    def cell_in_same_row(c1, c2):
        c1_center = c1[1] + c1[3] - c1[3] / 2
        c2_bottom = c2[1] + c2[3]
        c2_top = c2[1]
        return c2_top < c1_center < c2_bottom

    # orig_cells = [c for c in cells]
    rows = []
    while cells:
        first = cells[0]
        rest = cells[1:]
        cells_in_same_row = sorted(
            [
                c for c in rest
                if cell_in_same_row(c, first)
            ],
            key=lambda c: c[0]
        )

        row_cells = sorted([first] + cells_in_same_row, key=lambda c: c[0])
        rows.append(row_cells)
        cells = [
            c for c in rest
            if not cell_in_same_row(c, first)
        ]

    # Sort rows by average height of their center.
    def avg_height_of_center(row):
        centers = [y + h - h / 2 for x, y, w, h in row]
        return sum(centers) / len(centers)

    rows.sort(key=avg_height_of_center)
    cell_images_rows = []
    for row in rows:
        cell_images_row = []
        for x, y, w, h in row:
            cell_images_row.append(image[y:y+h, x:x+w])
        cell_images_rows.append(cell_images_row)
    return cell_images_rows


def extract_tables(image):
    blurred = cv2.GaussianBlur(image, (17, 17), 0, 0)
    img_bin = cv2.adaptiveThreshold(
        ~blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 15, -2)
    horizontal = img_bin.copy()
    SCALE = 10
    width, height = horizontal.shape
    hker = cv2.getStructuringElement(cv2.MORPH_RECT, (int(width / SCALE), 1))
    hopened = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, hker)
    vker = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, int(height / SCALE))
    )
    vopened = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, vker)
    horizontally_dilated = cv2.dilate(
        hopened, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    )
    vertically_dilated = cv2.dilate(
        vopened, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60))
    )
    mask = horizontally_dilated + vertically_dilated
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contours = [c for c in contours if cv2.contourArea(c) > 1e5]
    perimeter_lengths = [cv2.arcLength(c, True) for c in contours]
    epsilons = [0.1 * p for p in perimeter_lengths]
    approx_polys = [cv2.approxPolyDP(c, e, True) for c, e in zip(contours, epsilons)]
    bounding_rects = [cv2.boundingRect(a) for a in approx_polys]
    return bounding_rects

def table_image(image, bounding_rects):
    images = [image[y - 1 : y + h + 1, x : x + w] for x, y, w, h in bounding_rects]
    return images

def get_cell_images(files):
    results = []
    for f in files:
        directory, filename = os.path.split(f)
        table = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        rows = extract_cells(table)
        cell_img_dir = os.path.join(directory, "cells")
        os.makedirs(cell_img_dir, exist_ok=True)
        paths = []
        for i, row in enumerate(rows):
            for j, cell in enumerate(row):
                cell_filename = "{:03d}-{:03d}.png".format(i, j)
                path = os.path.join(cell_img_dir, cell_filename)
                cv2.imwrite(path, cell)
                paths.append(path)
        results.append(paths)
    return results

def get_table_images(files):
    results = []
    for f in files:
        directory, filename = os.path.split(f)
        image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        bounding_rects = extract_tables(image)
        tables = table_image(image, bounding_rects)
        img_files = []
        ext = os.path.splitext(filename)[0]
        if tables:
            os.makedirs(os.path.join(directory, ext), exist_ok=True)
        for i, table in enumerate(tables):
            table_filename = "table-{:03d}.png".format(i)
            table_filepath = os.path.join(directory, ext, table_filename)
            img_files.append(table_filepath)
            cv2.imwrite(table_filepath, table)
        if tables:
            results.append(img_files)
    return results
