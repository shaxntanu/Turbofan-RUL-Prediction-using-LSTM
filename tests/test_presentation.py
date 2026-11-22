import json
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "web" / "public" / "presentation"


class PresentationParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.slides = []
        self.tables = {}
        self.table = None
        self.row = None
        self.cell = None
        self.trace = None
        self.images = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "section" and attrs.get("class", "").split()[0:1] == ["slide"]:
            self.slides.append(attrs["id"])
        elif tag == "table":
            self.table = attrs.get("aria-label")
            self.tables[self.table] = []
        elif tag == "tr" and self.table:
            self.row = []
        elif tag in ("th", "td") and self.row is not None:
            self.cell = ""
        elif tag == "polyline":
            self.trace = attrs.get("points")
        elif tag == "img":
            self.images.append(attrs["src"])

    def handle_data(self, data):
        if self.cell is not None:
            self.cell += data

    def handle_endtag(self, tag):
        if tag in ("th", "td") and self.cell is not None:
            self.row.append(self.cell.strip())
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.tables[self.table].append(self.row)
            self.row = None
        elif tag == "table":
            self.table = None


def read_deck():
    parser = PresentationParser()
    parser.feed((DECK / "index.html").read_text(encoding="utf-8"))
    return parser


def test_presentation_has_unique_slides_and_local_images():
    deck = read_deck()
    assert deck.slides == [f"slide-{i}" for i in range(1, 21)]
    assert deck.images and all((DECK / image).is_file() for image in deck.images)


def test_reported_engine_counts_match_bundled_files():
    rows = read_deck().tables["C-MAPSS subset characteristics"][1:]
    for subset, train_count, test_count, *_ in rows:
        for part, expected in (("train", train_count), ("test", test_count)):
            path = ROOT / "Ref" / "CMAPSSData" / f"{part}_{subset}.txt"
            with path.open(encoding="ascii") as source:
                engines = {line.split()[0] for line in source if line.strip()}
            assert len(engines) == int(expected)


def test_result_table_matches_committed_benchmark():
    rows = read_deck().tables["Validation-selected models and test errors by subset"][1:]
    snapshot = json.loads((ROOT / "web" / "public" / "results" / "benchmark.json").read_text())
    for subset, ridge, lstm, selected, test_mae, test_rmse in rows:
        data = snapshot["datasets"][subset]
        assert ridge == f"{data['validation_metrics']['ridge']['mae']:.2f}"
        assert lstm == f"{data['validation_metrics']['lstm']['mae']:.2f}"
        assert selected.lower() == data["selected_model"]
        assert test_mae == f"{data['test_metrics']['mae']:.2f}"
        assert test_rmse == f"{data['test_metrics']['rmse']:.2f}"


def test_sensor_trace_uses_the_sampled_fd001_readings():
    path = ROOT / "Ref" / "CMAPSSData" / "train_FD001.txt"
    with path.open(encoding="ascii") as source:
        rows = [line.split() for line in source if line.startswith("1 ")]
    points = [
        (int(row[1]), float(row[8]))
        for row in rows
        if int(row[1]) % 4 == 1 or int(row[1]) == 192
    ]
    expected = " ".join(
        f"{70 + (cycle - 1) * 900 / 191:.1f},{390 - (reading - 1390) * 310 / 45:.1f}"
        for cycle, reading in points
    )
    assert read_deck().trace == expected
