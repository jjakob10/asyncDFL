"""Recreate the two asyncDFL architecture diagrams as editable Excalidraw files."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
FONT = 3  # Excalidraw's monospace font.


class Canvas:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.elements: list[dict] = []
        self.svg: list[str] = []
        self._counter = 0

    def _base(
        self,
        kind: str,
        x: float,
        y: float,
        width: float,
        height: float,
        stroke: str,
        fill: str = "transparent",
        stroke_width: int = 3,
        stroke_style: str = "solid",
        roughness: int = 1,
    ) -> dict:
        self._counter += 1
        return {
            "id": f"asyncdfl-{self._counter:03d}",
            "type": kind,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "angle": 0,
            "strokeColor": stroke,
            "backgroundColor": fill,
            "fillStyle": "solid",
            "strokeWidth": stroke_width,
            "strokeStyle": stroke_style,
            "roughness": roughness,
            "opacity": 100,
            "groupIds": [],
            "frameId": None,
            "index": f"a{self._counter:03d}",
            "roundness": {"type": 3} if kind == "rectangle" else None,
            "seed": random.Random(7319 + self._counter).randint(1, 2_147_483_646),
            "version": 1,
            "versionNonce": random.Random(9901 + self._counter).randint(1, 2_147_483_646),
            "isDeleted": False,
            "boundElements": [],
            "updated": 1_726_700_000_000,
            "link": None,
            "locked": False,
        }

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        stroke: str,
        fill: str,
        *,
        stroke_width: int = 3,
        stroke_style: str = "solid",
        radius: int = 24,
    ) -> None:
        element = self._base(
            "rectangle", x, y, width, height, stroke, fill, stroke_width, stroke_style
        )
        self.elements.append(element)
        dash = ' stroke-dasharray="14 12"' if stroke_style == "dashed" else ""
        self.svg.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
            f'rx="{radius}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{stroke_width}"{dash}/>'
        )

    def ellipse(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        stroke: str,
        fill: str,
        *,
        stroke_width: int = 3,
        stroke_style: str = "solid",
    ) -> None:
        element = self._base(
            "ellipse", x, y, width, height, stroke, fill, stroke_width, stroke_style
        )
        self.elements.append(element)
        dash = ' stroke-dasharray="14 12"' if stroke_style == "dashed" else ""
        self.svg.append(
            f'<ellipse cx="{x + width / 2}" cy="{y + height / 2}" '
            f'rx="{width / 2}" ry="{height / 2}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}"{dash}/>'
        )

    def text(
        self,
        x: float,
        y: float,
        value: str,
        color: str,
        *,
        size: int = 28,
        align: str = "left",
        width: float | None = None,
    ) -> None:
        lines = value.split("\n")
        line_height = 1.25
        estimated_width = max((len(line) for line in lines), default=1) * size * 0.62
        element_width = width if width is not None else estimated_width
        element_height = len(lines) * size * line_height
        element = self._base("text", x, y, element_width, element_height, color, "transparent", 1)
        element.update(
            {
                "fontSize": size,
                "fontFamily": FONT,
                "text": value,
                "textAlign": align,
                "verticalAlign": "top",
                "containerId": None,
                "originalText": value,
                "autoResize": True,
                "lineHeight": line_height,
            }
        )
        self.elements.append(element)
        anchor = {"left": "start", "center": "middle", "right": "end"}[align]
        tx = x if align == "left" else x + element_width / 2 if align == "center" else x + element_width
        svg_lines = []
        for index, line in enumerate(lines):
            dy = size if index == 0 else size * line_height
            svg_lines.append(
                f'<tspan x="{tx}" dy="{dy}">{escape(line)}</tspan>'
            )
        self.svg.append(
            f'<text x="{tx}" y="{y}" text-anchor="{anchor}" fill="{color}" '
            f'font-family="Cascadia Mono, Consolas, monospace" font-size="{size}">'
            + "".join(svg_lines)
            + "</text>"
        )

    def arrow(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str,
        *,
        stroke_width: int = 3,
        start_arrow: bool = False,
        end_arrow: bool = True,
    ) -> None:
        width = x2 - x1
        height = y2 - y1
        element = self._base("arrow", x1, y1, width, height, color, "transparent", stroke_width)
        element.update(
            {
                "points": [[0, 0], [width, height]],
                "lastCommittedPoint": None,
                "startBinding": None,
                "endBinding": None,
                "startArrowhead": "arrow" if start_arrow else None,
                "endArrowhead": "arrow" if end_arrow else None,
                "elbowed": False,
            }
        )
        self.elements.append(element)
        self.svg.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round"/>'
        )
        if start_arrow:
            self._svg_arrowhead(x2, y2, x1, y1, color, stroke_width)
        if end_arrow:
            self._svg_arrowhead(x1, y1, x2, y2, color, stroke_width)

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str,
        *,
        stroke_width: int = 4,
    ) -> None:
        width = x2 - x1
        height = y2 - y1
        element = self._base("line", x1, y1, width, height, color, "transparent", stroke_width)
        element.update(
            {
                "points": [[0, 0], [width, height]],
                "lastCommittedPoint": None,
                "startBinding": None,
                "endBinding": None,
                "startArrowhead": None,
                "endArrowhead": None,
            }
        )
        self.elements.append(element)
        self.svg.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round"/>'
        )

    def _svg_arrowhead(
        self,
        from_x: float,
        from_y: float,
        tip_x: float,
        tip_y: float,
        color: str,
        stroke_width: int,
    ) -> None:
        angle = math.atan2(tip_y - from_y, tip_x - from_x)
        length = 24
        spread = 0.55
        ax = tip_x - length * math.cos(angle - spread)
        ay = tip_y - length * math.sin(angle - spread)
        bx = tip_x - length * math.cos(angle + spread)
        by = tip_y - length * math.sin(angle + spread)
        self.svg.append(
            f'<path d="M {ax:.1f} {ay:.1f} L {tip_x} {tip_y} L {bx:.1f} {by:.1f}" '
            f'fill="none" stroke="{color}" stroke-width="{stroke_width}" '
            'stroke-linecap="round" stroke-linejoin="round"/>'
        )

    def write(self, stem: str) -> None:
        drawing = {
            "type": "excalidraw",
            "version": 2,
            "source": "https://excalidraw.com",
            "elements": self.elements,
            "appState": {
                "gridSize": None,
                "viewBackgroundColor": "#ffffff",
                "currentItemFontFamily": FONT,
            },
            "files": {},
        }
        (OUTPUTS / f"{stem}.excalidraw").write_text(
            json.dumps(drawing, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" '
            f'height="{self.height}" viewBox="0 0 {self.width} {self.height}">'
            '<rect width="100%" height="100%" fill="#ffffff"/>'
            + "".join(self.svg)
            + "</svg>"
        )
        (OUTPUTS / f"{stem}.svg").write_text(svg, encoding="utf-8")


def architecture_diagram() -> Canvas:
    c = Canvas(2364, 1516)
    blue, blue_fill = "#1971c2", "#d0ebff"
    orange, orange_fill = "#e67700", "#fff3bf"
    red, peach = "#d9480f", "#ffe8cc"
    purple, purple_fill = "#7048e8", "#f3f0ff"
    green, green_fill = "#2b8a3e", "#d3f9d8"
    teal, teal_fill = "#087f5b", "#e6fcf5"
    gray, gray_fill = "#495057", "#f1f3f5"

    # Connectors behind the boxes.
    c.arrow(414, 211, 633, 211, blue, stroke_width=4, start_arrow=True)
    c.arrow(1200, 220, 1375, 220, orange, stroke_width=4)
    c.arrow(1650, 382, 1308, 635, purple, stroke_width=4, start_arrow=True)
    c.arrow(1904, 382, 1904, 706, green, stroke_width=4, start_arrow=True)
    c.arrow(532, 802, 693, 875, red, stroke_width=3)
    c.arrow(1918, 1177, 1918, 1290, teal, stroke_width=4)

    c.rect(24, 49, 383, 328, blue, "#e7f5ff", stroke_width=3)
    c.text(57, 77, "Peer clients", blue, size=39)
    c.text(57, 143, "POST /start\nGET /model\nGET /health\nGET /status", "#343a40", size=28)

    c.rect(633, 12, 567, 437, blue, blue_fill, stroke_width=3)
    c.text(665, 43, "API layer – app.main", blue, size=38)
    c.text(
        665,
        103,
        "FastAPI on :9000\nJSON control/status\nBinary model payload\n202 / 409 idempotency\nLifespan creates runtime",
        "#343a40",
        size=28,
    )
    c.text(660, 351, "/health  /start  /model  /status", blue, size=28)
    c.text(429, 259, "HTTP / REST", blue, size=25)
    c.text(1216, 162, "delegates", orange, size=25)

    c.rect(1376, 70, 912, 304, orange, orange_fill, stroke_width=3)
    c.text(1410, 98, "Orchestration – ClientRuntime", orange, size=37)
    c.text(
        1410,
        155,
        "RLock protects published state\nDaemon worker executes rounds\nRetries + timeout with httpx\nStart floods to configured neighbors\nSnapshot headers: client, version, round, samples",
        "#343a40",
        size=27,
    )
    c.text(1240, 481, "protocol call", purple, size=25)
    c.text(1740, 526, "aggregate", green, size=25)

    c.rect(38, 620, 494, 312, red, peach, stroke_width=3)
    c.text(71, 652, "Local data", red, size=37)
    c.text(
        71,
        716,
        "IID or Dirichlet shard\ntrain_loader / test_loader\nsample_count = weight",
        "#343a40",
        size=27,
    )
    c.text(548, 764, "batches", red, size=25)

    c.rect(68, 1032, 492, 273, gray, gray_fill, stroke_width=3)
    c.text(100, 1061, "Configuration", gray, size=37)
    c.text(
        100,
        1126,
        "Settings.from_env()\nneighbors, rounds, trainer\ntimeouts, seed, partition\nused by runtime + trainer",
        "#343a40",
        size=26,
    )

    c.rect(687, 633, 787, 601, purple, purple_fill, stroke_width=4, stroke_style="dashed")
    c.text(733, 676, "Learning plug-in boundary", purple, size=39)
    c.text(733, 748, "app.training.Trainer protocol", purple, size=31)
    c.text(
        733,
        830,
        "create_model()\nload_data(settings) → DataBundle\ntrain_local(state, loader, settings)\nevaluate(state, test_loader)",
        "#343a40",
        size=28,
    )
    c.rect(759, 1069, 658, 117, purple, "#e5dbff", stroke_width=3)
    c.text(796, 1104, "MNIST / CIFAR-10 / HARTH trainer", purple, size=26)

    c.rect(1582, 705, 675, 474, green, green_fill, stroke_width=3)
    c.text(1613, 736, "Federation – app.federation", green, size=38)
    c.text(
        1613,
        800,
        "CPU-cloned StateDict\nserialize / deserialize\nshape + dtype compatibility\nsample-count-weighted FedAvg\natomic model publication",
        "#343a40",
        size=27,
    )
    c.text(1760, 1216, "persist", teal, size=25)

    c.rect(1582, 1289, 675, 215, teal, teal_fill, stroke_width=3)
    c.text(1614, 1321, "Outputs", teal, size=37)
    c.text(1614, 1381, "• metrics JSONL\n• final .pt snapshot\n• status/errors", "#343a40", size=26)

    c.rect(355, 1337, 1178, 176, "#f08c00", "#fff9db", stroke_width=3)
    c.text(473, 1379, "Round 1…N", "#f08c00", size=34)
    c.text(
        473,
        1438,
        "local train → fetch neighbors → FedAvg → evaluate → publish",
        "#343a40",
        size=25,
    )
    return c


def deployment_diagram() -> Canvas:
    c = Canvas(2418, 1510)
    blue, blue_fill = "#1971c2", "#d0ebff"
    orange, orange_fill = "#e67700", "#fff3bf"
    green, green_fill = "#2b8a3e", "#d3f9d8"
    gray, gray_fill = "#868e96", "#f8f9fa"

    nodes = {
        "client-0": (1025, 9, 255, 178),
        "client-1": (1373, 151, 255, 181),
        "client-2": (1515, 501, 255, 181),
        "client-3": (1373, 849, 255, 182),
        "client-4": (1024, 995, 255, 183),
        "client-5": (669, 849, 255, 182),
        "client-6": (526, 500, 255, 183),
        "client-7": (669, 151, 255, 182),
    }
    centers = {
        name: (x + w / 2, y + h / 2) for name, (x, y, w, h) in nodes.items()
    }
    ring = [
        "client-0",
        "client-1",
        "client-2",
        "client-3",
        "client-4",
        "client-5",
        "client-6",
        "client-7",
    ]
    for index, name in enumerate(ring):
        x1, y1 = centers[name]
        x2, y2 = centers[ring[(index + 1) % len(ring)]]
        c.line(x1, y1, x2, y2, blue, stroke_width=5)

    c.text(102, 83, "Example client details", "#495057", size=29)
    c.rect(69, 139, 521, 312, gray, gray_fill, stroke_width=3)
    c.text(104, 178, "client-3", blue, size=37)
    c.text(
        104,
        249,
        "NEIGHBORS:\nhttp://client-2:9000\nhttp://client-4:9000\nlocal partition + own model",
        "#343a40",
        size=26,
    )

    c.rect(1704, 139, 588, 313, gray, gray_fill, stroke_width=3)
    c.text(1737, 177, "Deployment boundary", "#495057", size=37)
    c.text(
        1737,
        247,
        "8 independent Docker services\ninternal API port 8000\nhost ports 8000…8007\nshared /results volume",
        "#343a40",
        size=26,
    )

    for name, (x, y, w, h) in nodes.items():
        is_head = name == "client-7"
        stroke = orange if is_head else blue
        fill = orange_fill if is_head else blue_fill
        c.ellipse(x, y, w, h, stroke, fill, stroke_width=4)
        c.text(x + 45, y + 35, name, stroke, size=34)
        c.text(x + 60, y + 102, "HEAD" if is_head else ":9000", orange if is_head else "#495057", size=26)

    c.ellipse(938, 431, 420, 320, gray, "#ffffff", stroke_width=3, stroke_style="dashed")
    c.text(987, 499, "Logical ring", "#495057", size=41)
    c.text(989, 578, "no central server", "#495057", size=28)
    c.text(992, 644, "asynchronous FedAvg", "#495057", size=28)

    c.rect(69, 1132, 680, 303, orange, orange_fill, stroke_width=3)
    c.text(101, 1165, "Head: client-7", orange, size=36)
    c.text(
        101,
        1221,
        "HEAD_CLIENT_IDS=7\nwaits for peers to be healthy\nauto POST /start with shared run_id\nneighbors: client-6 and client-0",
        "#343a40",
        size=25,
    )

    c.rect(791, 1209, 714, 189, blue, "#e7f5ff", stroke_width=3)
    c.text(825, 1239, "REST model exchange", blue, size=36)
    c.text(825, 1303, "application/octet-stream StateDict", "#495057", size=25)
    c.text(825, 1352, "headers: ID • version • round • samples", "#495057", size=24)

    c.rect(1528, 1132, 764, 303, green, green_fill, stroke_width=3)
    c.text(1561, 1165, "Every round on every client", green, size=36)
    c.text(
        1561,
        1221,
        "1  train local partition\n2  GET /model from both neighbors\n3  FedAvg by sample_count\n4  publish version + evaluate + record",
        "#343a40",
        size=25,
    )

    c.text(
        174,
        1462,
        "Blue bidirectional edges = each endpoint can POST /start and GET /model from the neighbor URL",
        blue,
        size=23,
    )
    return c


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    architecture_diagram().write("asyncdfl-tool-architecture")
    deployment_diagram().write("asyncdfl-example-deployment")
    print("Created editable Excalidraw files and SVG previews in outputs/.")


if __name__ == "__main__":
    main()
