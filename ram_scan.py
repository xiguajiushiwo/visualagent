from __future__ import annotations

import argparse
import base64
import dataclasses
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ZHIPU_API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"


@dataclass
class Box:
    x0: float
    y0: float
    x1: float
    y1: float

    def clip(self, width: int, height: int) -> "Box":
        return Box(
            x0=max(0.0, min(float(width), self.x0)),
            y0=max(0.0, min(float(height), self.y0)),
            x1=max(0.0, min(float(width), self.x1)),
            y1=max(0.0, min(float(height), self.y1)),
        )

    def expand(self, ratio: float, width: int, height: int) -> "Box":
        dx = (self.x1 - self.x0) * ratio
        dy = (self.y1 - self.y0) * ratio
        return Box(self.x0 - dx, self.y0 - dy, self.x1 + dx, self.y1 + dy).clip(width, height)

    @property
    def w(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def h(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.w * self.h

    def as_list(self) -> list[float]:
        return [round(self.x0, 2), round(self.y0, 2), round(self.x1, 2), round(self.y1, 2)]

    def iou(self, other: "Box") -> float:
        ix0 = max(self.x0, other.x0)
        iy0 = max(self.y0, other.y0)
        ix1 = min(self.x1, other.x1)
        iy1 = min(self.y1, other.y1)
        iw = max(0.0, ix1 - ix0)
        ih = max(0.0, iy1 - iy0)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass
class Detection:
    label: str
    box: Box
    score: float = 0.0
    note: str = ""
    needs_recheck: bool = False
    source: str = ""
    tile_id: str = ""
    raw: dict[str, Any] = dataclasses.field(default_factory=dict)
    recheck: dict[str, Any] = dataclasses.field(default_factory=dict)
    sam: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["box"] = self.box.as_list()
        return data


@dataclass
class Tile:
    tile_id: str
    box: Box
    image: Image.Image


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def image_to_data_url(image: Image.Image, fmt: str = "PNG") -> str:
    import io

    buf = io.BytesIO()
    image.save(buf, format=fmt)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def extract_json_block(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    for candidate in (text,):
        try:
            return json.loads(candidate)
        except Exception:
            pass

    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    match = re.search(r"\[.*\]", text, re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    raise ValueError(f"Unable to parse JSON from model output: {text[:400]}")


def normalize_box(box: Box, width: int, height: int) -> list[float]:
    return [
        round(box.x0 / width, 6),
        round(box.y0 / height, 6),
        round(box.x1 / width, 6),
        round(box.y1 / height, 6),
    ]


def denormalize_box(vals: Iterable[float], width: int, height: int) -> Box:
    x0, y0, x1, y1 = list(vals)
    return Box(x0 * width, y0 * height, x1 * width, y1 * height).clip(width, height)


def parse_box(values: Any, width: int, height: int, normalized: bool = True) -> Optional[Box]:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        return None
    try:
        vals = [float(v) for v in values]
    except Exception:
        return None
    if normalized:
        box = denormalize_box(vals, width, height)
    else:
        box = Box(*vals).clip(width, height)
    if box.w < 4 or box.h < 4:
        return None
    return box


def nms(detections: list[Detection], iou_threshold: float = 0.35) -> list[Detection]:
    detections = sorted(detections, key=lambda d: d.score, reverse=True)
    kept: list[Detection] = []
    for det in detections:
        if all(det.box.iou(other.box) < iou_threshold or det.label != other.label for other in kept):
            kept.append(det)
    return kept


def tile_grid(box: Box, tile_size: int, stride: int, image_w: int, image_h: int) -> list[Box]:
    x0, y0, x1, y1 = [int(round(v)) for v in box.clip(image_w, image_h).as_list()]
    tiles: list[Box] = []
    xs = list(range(x0, max(x0, x1 - tile_size) + 1, stride))
    ys = list(range(y0, max(y0, y1 - tile_size) + 1, stride))
    if not xs:
        xs = [max(0, min(image_w - tile_size, x0))]
    if not ys:
        ys = [max(0, min(image_h - tile_size, y0))]
    if xs[-1] != max(x0, x1 - tile_size):
        xs.append(max(x0, x1 - tile_size))
    if ys[-1] != max(y0, y1 - tile_size):
        ys.append(max(y0, y1 - tile_size))

    for y in ys:
        for x in xs:
            tiles.append(Box(x, y, min(x + tile_size, image_w), min(y + tile_size, image_h)))
    return tiles


def crop_image(img: Image.Image, box: Box) -> Image.Image:
    return img.crop((int(round(box.x0)), int(round(box.y0)), int(round(box.x1)), int(round(box.y1))))


def draw_annotations(image: Image.Image, detections: list[Detection], tiles: list[Tile], out_path: Path) -> None:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for tile in tiles:
        draw.rectangle(tile.box.as_list(), outline=(40, 180, 255), width=2)

    for det in detections:
        color = (255, 64, 64) if det.needs_recheck else (60, 255, 90)
        draw.rectangle(det.box.as_list(), outline=color, width=4)
        label = f"{det.label} {det.score:.2f}"
        draw.text((det.box.x0 + 2, max(0, det.box.y0 - 18)), label, fill=color, font=font)

    canvas.save(out_path)


class ZhipuClient:
    def __init__(self, api_key: str, base_url: str = ZHIPU_API_URL) -> None:
        self.api_key = api_key
        self.base_url = base_url

    def chat(self, model: str, prompt: str, image: Image.Image, temperature: float = 0.0, max_tokens: int = 1200) -> str:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_to_data_url(image)}},
                    ],
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        req = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Zhipu HTTP {exc.code}: {exc.read().decode('utf-8', errors='ignore')}") from exc
        data = json.loads(body)
        return data["choices"][0]["message"]["content"]


LOCATE_PROMPT = """You inspect images of RAM modules.
Find the visible memory module to be scanned.
Return ONLY valid JSON with this schema:
{
  "subject_box": [x1, y1, x2, y2],
  "confidence": 0.0,
  "note": "short reason"
}
Coordinates must be normalized to the full image, in [0,1].
If there are multiple modules, choose the clearest fully visible one.
"""


SCAN_PROMPT = """You inspect a small crop of a RAM module for manufacturing defects.
Focus on tiny defects:
- missing component / drop-off
- shifted / lifted / tilted component
- burn mark / scorch
- scratch / crack
- foreign object / debris
- exposed pad / abnormal solder / abnormal reflection that suggests damage

Return ONLY JSON with this schema:
{
  "tile_status": "ok" or "suspect",
  "detections": [
    {
      "label": "missing_component | scratch | burn_mark | crack | foreign_object | abnormal_solder | lifted_part | other",
      "bbox": [x1, y1, x2, y2],
      "score": 0.0,
      "needs_recheck": true,
      "reason": "short reason"
    }
  ]
}
Coordinates must be normalized to the tile crop in [0,1].
If unsure, keep score under 0.6 and set needs_recheck true.
Return an empty detections array when nothing is suspicious.
"""


RECHECK_PROMPT = """This is a zoomed crop around one suspicious area on a RAM module.
Judge whether the defect is real.
Return ONLY JSON with this schema:
{
  "confirmed": true or false,
  "label": "same label or other",
  "bbox": [x1, y1, x2, y2],
  "score": 0.0,
  "reason": "short reason"
}
Coordinates must be normalized to the crop in [0,1].
If you cannot confirm, set confirmed false and score below 0.5.
"""


OCR_PROMPT = """Extract every readable text fragment visible in this crop.
Return ONLY JSON with this schema:
{
  "texts": [
    {
      "text": "string",
      "bbox": [x1, y1, x2, y2],
      "score": 0.0
    }
  ]
}
Coordinates must be normalized to the crop in [0,1].
If no text is legible, return an empty array.
"""


class SamRefiner:
    def __init__(self, checkpoint: str, model_type: str = "vit_h") -> None:
        self.checkpoint = checkpoint
        self.model_type = model_type
        self.available = False
        self._predictor = None
        try:
            import torch  # noqa: F401
            from segment_anything import SamPredictor, sam_model_registry
        except Exception:
            return

        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self._predictor = SamPredictor(sam)
        self.available = True

    def refine(self, image: Image.Image, box: Box) -> dict[str, Any]:
        if not self.available or self._predictor is None:
            return {}
        image_np = np.array(image)
        self._predictor.set_image(image_np)
        masks, scores, _ = self._predictor.predict(box=np.array(box.as_list(), dtype=np.float32), multimask_output=True)
        best_idx = int(np.argmax(scores))
        mask = masks[best_idx]
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return {"score": float(scores[best_idx]), "mask_found": False}
        x0 = int(xs.min())
        y0 = int(ys.min())
        x1 = int(xs.max()) + 1
        y1 = int(ys.max()) + 1
        return {
            "score": float(scores[best_idx]),
            "mask_found": True,
            "bbox": [x0, y0, x1, y1],
        }


def clamp_subject_box(subject: Box, width: int, height: int, pad: float = 0.04) -> Box:
    return subject.expand(pad, width, height)


class Pipeline:
    def __init__(
        self,
        client: Optional[ZhipuClient],
        model: str,
        tile_size: int = 256,
        stride: int = 128,
        max_tiles: int = 999,
        recheck_pad: float = 0.35,
        ocr: bool = True,
        sam_refiner: Optional[SamRefiner] = None,
    ) -> None:
        self.client = client
        self.model = model
        self.tile_size = tile_size
        self.stride = stride
        self.max_tiles = max_tiles
        self.recheck_pad = recheck_pad
        self.ocr = ocr
        self.sam_refiner = sam_refiner

    def locate_subject(self, image: Image.Image, manual_box: Optional[Box] = None) -> Box:
        if manual_box is not None:
            return manual_box.clip(image.width, image.height)
        if self.client is None:
            return Box(0, 0, image.width, image.height)
        raw = self.client.chat(self.model, LOCATE_PROMPT, image, temperature=0.0, max_tokens=300)
        data = extract_json_block(raw)
        box = parse_box(data.get("subject_box"), image.width, image.height, normalized=True)
        if box is None:
            return Box(0, 0, image.width, image.height)
        return clamp_subject_box(box, image.width, image.height)

    def scan_tile(self, tile: Tile) -> tuple[list[Detection], list[Detection]]:
        if self.client is None:
            return [], []
        raw = self.client.chat(self.model, SCAN_PROMPT, tile.image, temperature=0.0, max_tokens=900)
        data = extract_json_block(raw)
        out: list[Detection] = []
        ocrs: list[Detection] = []

        if isinstance(data, dict):
            dets = data.get("detections") or []
            for idx, item in enumerate(dets):
                box = parse_box(item.get("bbox"), tile.image.width, tile.image.height, normalized=True)
                if box is None:
                    continue
                global_box = Box(
                    tile.box.x0 + box.x0,
                    tile.box.y0 + box.y0,
                    tile.box.x0 + box.x1,
                    tile.box.y0 + box.y1,
                )
                score = float(item.get("score", 0.0) or 0.0)
                det = Detection(
                    label=str(item.get("label", "other")),
                    box=global_box,
                    score=score,
                    note=str(item.get("reason", "")),
                    needs_recheck=bool(item.get("needs_recheck", False) or score < 0.6),
                    source="scan",
                    tile_id=tile.tile_id,
                    raw=item,
                )
                out.append(det)

        if self.ocr:
            try:
                raw_ocr = self.client.chat(self.model, OCR_PROMPT, tile.image, temperature=0.0, max_tokens=700)
                ocr_data = extract_json_block(raw_ocr)
                if isinstance(ocr_data, dict):
                    for item in ocr_data.get("texts") or []:
                        box = parse_box(item.get("bbox"), tile.image.width, tile.image.height, normalized=True)
                        if box is None:
                            continue
                        ocrs.append(
                            Detection(
                                label="ocr",
                                box=Box(
                                    tile.box.x0 + box.x0,
                                    tile.box.y0 + box.y0,
                                    tile.box.x0 + box.x1,
                                    tile.box.y0 + box.y1,
                                ),
                                score=float(item.get("score", 0.0) or 0.0),
                                note=str(item.get("text", "")),
                                source="ocr",
                                tile_id=tile.tile_id,
                                raw=item,
                            )
                        )
            except Exception:
                pass

        return out, ocrs

    def recheck(self, image: Image.Image, det: Detection) -> Detection:
        if self.client is None:
            return det
        crop = det.box.expand(self.recheck_pad, image.width, image.height)
        crop_img = crop_image(image, crop)
        raw = self.client.chat(self.model, RECHECK_PROMPT, crop_img, temperature=0.0, max_tokens=700)
        data = extract_json_block(raw)
        if isinstance(data, dict):
            det.recheck = data
            det.score = float(data.get("score", det.score) or det.score)
            det.note = data.get("reason", det.note)
            confirmed = bool(data.get("confirmed", False))
            det.needs_recheck = not confirmed
            box = parse_box(data.get("bbox"), crop_img.width, crop_img.height, normalized=True)
            if box is not None:
                det.box = Box(
                    crop.x0 + box.x0,
                    crop.y0 + box.y0,
                    crop.x0 + box.x1,
                    crop.y0 + box.y1,
                ).clip(image.width, image.height)
        if self.sam_refiner is not None and self.sam_refiner.available:
            try:
                refined = self.sam_refiner.refine(image, det.box)
                det.sam = refined
                if refined.get("mask_found"):
                    rb = refined.get("bbox")
                    if rb:
                        det.box = Box(*map(float, rb)).clip(image.width, image.height)
            except Exception as exc:
                det.sam = {"error": str(exc)}
        return det

    def build_tiles(self, image: Image.Image, subject: Box) -> list[Tile]:
        tiles = tile_grid(subject, self.tile_size, self.stride, image.width, image.height)
        if len(tiles) > self.max_tiles:
            tiles = tiles[: self.max_tiles]
        out: list[Tile] = []
        for idx, box in enumerate(tiles):
            crop = crop_image(image, box)
            out.append(Tile(tile_id=f"t{idx:04d}", box=box, image=crop))
        return out

    def run(self, image: Image.Image, manual_subject: Optional[Box] = None) -> dict[str, Any]:
        subject = self.locate_subject(image, manual_subject)
        tiles = self.build_tiles(image, subject)

        detections: list[Detection] = []
        ocrs: list[Detection] = []
        for tile in tiles:
            dets, tile_ocr = self.scan_tile(tile)
            detections.extend(dets)
            ocrs.extend(tile_ocr)

        detections = nms(detections)
        rechecked: list[Detection] = []
        for det in detections:
            if det.needs_recheck or det.score < 0.65:
                rechecked.append(self.recheck(image, det))
            else:
                rechecked.append(det)
        rechecked = nms(rechecked)

        return {
            "subject_box": subject.as_list(),
            "tiles": [{"tile_id": t.tile_id, "box": t.box.as_list()} for t in tiles],
            "detections": [d.to_json() for d in rechecked],
            "ocr": [o.to_json() for o in ocrs],
            "sam_enabled": bool(self.sam_refiner and self.sam_refiner.available),
        }


def parse_manual_box(value: Optional[str]) -> Optional[Box]:
    if not value:
        return None
    parts = [float(v) for v in re.split(r"[,\s]+", value.strip()) if v]
    if len(parts) != 4:
        raise ValueError("--subject-bbox needs four numbers: x0 y0 x1 y1")
    return Box(*parts)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RAM inspection pipeline: tiling + Zhipu VLM + recheck + optional SAM.")
    p.add_argument("image", type=Path, help="input image path")
    p.add_argument("--outdir", type=Path, default=Path("runs/ram_scan"), help="output directory")
    p.add_argument("--api-key", default=os.getenv("ZHIPU_API_KEY", ""), help="Zhipu API key")
    p.add_argument("--model", default=os.getenv("ZHIPU_MODEL", "glm-5v-turbo"), help="Zhipu VLM model")
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--max-tiles", type=int, default=64)
    p.add_argument("--recheck-pad", type=float, default=0.35)
    p.add_argument("--subject-bbox", default="", help="manual subject box x0,y0,x1,y1 in pixels")
    p.add_argument("--no-ocr", action="store_true", help="skip OCR pass")
    p.add_argument("--sam-checkpoint", default=os.getenv("SAM_CHECKPOINT", ""), help="SAM checkpoint path")
    p.add_argument("--sam-model-type", default=os.getenv("SAM_MODEL_TYPE", "vit_h"), help="SAM model type")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    image = load_image(args.image)
    manual_box = parse_manual_box(args.subject_bbox)

    client = ZhipuClient(args.api_key) if args.api_key else None
    sam_refiner = None
    if args.sam_checkpoint:
        sam_refiner = SamRefiner(args.sam_checkpoint, args.sam_model_type)

    pipeline = Pipeline(
        client=client,
        model=args.model,
        tile_size=args.tile_size,
        stride=args.stride,
        max_tiles=args.max_tiles,
        recheck_pad=args.recheck_pad,
        ocr=not args.no_ocr,
        sam_refiner=sam_refiner,
    )
    result = pipeline.run(image, manual_box)

    out_json = args.outdir / "results.json"
    out_img = args.outdir / "annotated.png"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    tiles = [Tile(tile_id=t["tile_id"], box=Box(*t["box"]), image=image.crop((0, 0, 1, 1))) for t in result["tiles"]]
    dets = [
        Detection(
            label=d["label"],
            box=Box(*d["box"]),
            score=float(d.get("score", 0.0)),
            note=str(d.get("note", "")),
            needs_recheck=bool(d.get("needs_recheck", False)),
        )
        for d in result["detections"]
    ]
    draw_annotations(image, dets, tiles, out_img)

    print(f"saved: {out_json}")
    print(f"saved: {out_img}")
    print(f"sam_enabled: {result['sam_enabled']}")
    print(f"detections: {len(result['detections'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
