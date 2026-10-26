"""Read PPTX slide text and raster pictures without office software."""

import posixpath
import re
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Callable
from xml.etree import ElementTree as ET

from PIL import Image


P = "http://schemas.openxmlformats.org/presentationml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL = "http://schemas.openxmlformats.org/package/2006/relationships"


def _member(part: str, target: str) -> str:
    member = posixpath.normpath(target.lstrip("/") if target.startswith("/") else
                               posixpath.join(posixpath.dirname(part), target))
    if not member.startswith("ppt/"):
        raise ValueError(f"Unsafe PPTX relationship target: {target}")
    return member


def _relationships(archive: zipfile.ZipFile, part: str) -> dict[str, tuple[str, str]]:
    rels = posixpath.join(posixpath.dirname(part), "_rels", posixpath.basename(part) + ".rels")
    if rels not in archive.namelist():
        return {}
    root = ET.fromstring(archive.read(rels))
    return {node.get("Id", ""): (node.get("Type", ""), _member(part, node.get("Target", "")))
            for node in root.findall(f"{{{REL}}}Relationship") if node.get("TargetMode") != "External"}


def _text(shape: ET.Element) -> str:
    paragraphs = shape.findall(f".//{{{A}}}p")
    return "\n".join("".join(node.text or "" for node in para.iter(f"{{{A}}}t")).strip()
                     for para in paragraphs).strip()


def _notes(archive: zipfile.ZipFile, rels: dict[str, tuple[str, str]]) -> str:
    for kind, part in rels.values():
        if kind.endswith("/notesSlide"):
            root = ET.fromstring(archive.read(part))
            values = []
            for shape in root.findall(f".//{{{P}}}sp"):
                placeholder = shape.find(f".//{{{P}}}ph")
                if placeholder is not None and placeholder.get("type", "body") != "body":
                    continue
                value = _text(shape)
                if value:
                    values.append(value)
            return "\n".join(values)
    return ""


def parse_pptx(source: Path, images_dir: Path, manifest: dict[str, Any],
               ocr_text: Callable[[Path, dict[str, Any]], str]) -> str:
    sections = []
    with zipfile.ZipFile(source) as archive:
        presentation = "ppt/presentation.xml"
        if presentation in archive.namelist():
            order = ET.fromstring(archive.read(presentation))
            presentation_rels = _relationships(archive, presentation)
            slides = [presentation_rels[slide.get(f"{{{R}}}id")][1]
                      for slide in order.findall(f".//{{{P}}}sldId")
                      if slide.get(f"{{{R}}}id") in presentation_rels]
        else:
            slides = sorted((name for name in archive.namelist()
                             if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)),
                            key=lambda name: int(re.search(r"slide(\d+)\.xml$", name).group(1)))
        if not slides:
            raise ValueError("PPTX contains no slide XML")
        manifest["slide_map"] = []
        for number, part in enumerate(slides, start=1):
            root = ET.fromstring(archive.read(part))
            rels = _relationships(archive, part)
            blocks = [f"<!-- SLIDE:{number} -->", f"# Slide {number}"]
            manifest["slide_map"].append({"slide": number, "marker": f"<!-- SLIDE:{number} -->"})
            tree = root.find(f".//{{{P}}}spTree")
            if tree is None:
                sections.append("\n\n".join(blocks))
                continue
            recent_text = ""
            for shape in tree:
                if shape.tag in {f"{{{P}}}sp", f"{{{P}}}graphicFrame"}:
                    value = _text(shape)
                    if not value:
                        continue
                    placeholder = shape.find(f".//{{{P}}}ph")
                    if placeholder is not None and placeholder.get("type") in {"title", "ctrTitle"}:
                        value = f"## {value}"
                    blocks.append(value)
                    recent_text = value
                elif shape.tag == f"{{{P}}}pic":
                    blip = shape.find(f".//{{{A}}}blip")
                    rel_id = blip.get(f"{{{R}}}embed") if blip is not None else None
                    if not rel_id or rel_id not in rels or not rels[rel_id][0].endswith("/image"):
                        continue
                    data = archive.read(rels[rel_id][1])
                    with Image.open(BytesIO(data)) as image:
                        image.verify()
                    with Image.open(BytesIO(data)) as image:
                        image_format = image.format
                        width, height = image.size
                    if image_format not in {"PNG", "JPEG", "GIF", "BMP", "WEBP", "TIFF"}:
                        manifest["errors"].append({"file": part, "slide": number,
                                                   "type": "UnsupportedImage", "message": str(image_format)})
                        continue
                    extension = {"JPEG": "jpg", "TIFF": "tiff"}.get(image_format, image_format.lower())
                    asset_id = f"slide-{number:04d}-image-{len(manifest['image_assets']) + 1:03d}"
                    path = f"images/{asset_id}.{extension}"
                    asset_path = images_dir / f"{asset_id}.{extension}"
                    asset_path.write_bytes(data)
                    ocr = ocr_text(asset_path, manifest)
                    manifest["image_assets"].append({"asset_id": asset_id, "source_doc_id": manifest["doc_id"],
                                                     "path": path, "slide": number,
                                                     "kind": "embedded", "format": image_format,
                                                     "nearby_text": recent_text[:200], "caption": None,
                                                     "width": width, "height": height,
                                                     "ocr_character_count": len(ocr)})
                    blocks.append(f"![PPTX image {number}]({path})")
                    if ocr:
                        blocks.append(f"<!-- OCR:local image={path} -->\n{ocr}")
            notes = _notes(archive, rels)
            if notes:
                blocks.append(f"### Speaker notes\n{notes}")
            sections.append("\n\n".join(blocks))
    manifest["slide_count"] = len(slides)
    return "\n\n".join(sections) + "\n"
