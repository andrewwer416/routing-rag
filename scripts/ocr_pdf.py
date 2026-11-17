"""Create a searchable PDF copy without changing the original page images.

Install with: pip install -e '.[ocr]'
Run with: python scripts/ocr_pdf.py football.pdf data/football-searchable.pdf
OCR output must be checked against the original, especially tables and proper names.
"""

import argparse
import io
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
from pypdf import PdfReader, PdfWriter
from rapidocr_onnxruntime import RapidOCR
from reportlab.pdfgen import canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.destination.resolve():
        parser.error("Destination must differ from source")
    if args.destination.exists():
        parser.error("Destination already exists; choose a new path")
    engine = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=2)
    document = pdfium.PdfDocument(str(args.source))
    original = PdfReader(args.source)
    writer = PdfWriter()
    for index, page in enumerate(original.pages):
        rendered = document[index].render(scale=2).to_pil().convert("RGB")
        # RapidOCR expects OpenCV/BGR channel order for NumPy input.
        results, _ = engine(np.asarray(rendered)[:, :, ::-1].copy())
        overlay = io.BytesIO()
        width, height = float(page.mediabox.width), float(page.mediabox.height)
        layer = canvas.Canvas(overlay, pagesize=(width, height))
        for box, text, _ in results or []:
            x = min(p[0] for p in box) * width / rendered.width
            y = height - max(p[1] for p in box) * height / rendered.height
            box_height = max(p[1] for p in box) - min(p[1] for p in box)
            font_size = max(3, min(14, box_height * height / rendered.height * 0.85))
            t = layer.beginText(x, y)
            t.setFont("Helvetica", font_size)
            t.setTextRenderMode(3)  # Invisible, searchable text over original image.
            t.textLine(text)
            layer.drawText(t)
        layer.save()
        overlay.seek(0)
        page.merge_page(PdfReader(overlay).pages[0])
        writer.add_page(page)
        writer.pages[-1].compress_content_streams()
        print(
            f"OCR page {index + 1}/{len(original.pages)}: {len(results or [])} text boxes",
            flush=True,
        )
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with args.destination.open("wb") as output:
        writer.write(output)
    print(f"Searchable copy: {args.destination}", flush=True)


if __name__ == "__main__":
    main()
