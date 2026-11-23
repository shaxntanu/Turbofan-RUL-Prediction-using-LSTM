import json
from html.parser import HTMLParser
from pathlib import Path
import struct


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
        self.subset_mae_rows = []

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
        elif tag == "g" and attrs.get("class") == "subset-mae-row":
            self.subset_mae_rows.append(
                {
                    "subset": attrs["data-subset"],
                    "model": attrs["data-model"],
                    "mae": attrs["data-mae"],
                }
            )

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
    assert deck.slides == [f"slide-{i}" for i in range(1, 25)]
    assert deck.images and all((DECK / image).is_file() for image in deck.images)


def test_landscape_prediction_figure_is_wide_and_keeps_the_recorded_result():
    path = DECK / "figures" / "fd001-predicted-vs-actual-landscape.png"
    width, height = struct.unpack(">II", path.read_bytes()[16:24])
    assert width > height

    snapshot = json.loads((ROOT / "web" / "public" / "results" / "benchmark.json").read_text())
    assert snapshot["datasets"]["FD001"]["test_metrics"]["n_samples"] == 10196
    assert f"{snapshot['datasets']['FD001']['test_metrics']['mae']:.2f}" == "10.31"


def test_subset_mae_chart_matches_the_committed_snapshot():
    deck = read_deck()
    snapshot = json.loads((ROOT / "web" / "public" / "results" / "benchmark.json").read_text())
    assert [row["subset"] for row in deck.subset_mae_rows] == ["FD001", "FD002", "FD003", "FD004"]
    for row in deck.subset_mae_rows:
        data = snapshot["datasets"][row["subset"]]
        assert row["model"] == data["selected_model"]
        assert row["mae"] == f"{data['test_metrics']['mae']:.2f}"


def test_sdg_slide_stays_within_the_documented_alignment():
    html = (DECK / "index.html").read_text(encoding="utf-8")
    assessment = (ROOT / "docs" / "sdg-alignment.md").read_text(encoding="utf-8")
    assert "Target 9.5" in html
    assert "No other SDG impact is claimed" in html
    assert "no other SDG alignment is sufficiently direct to claim" in assessment


def test_presentation_deliverables_do_not_contain_em_dash():
    for path in (DECK / "index.html", DECK / "deck.css", DECK / "README.md"):
        assert "\u2014" not in path.read_text(encoding="utf-8")


def test_github_pages_publishes_the_deck_and_badge_targets_it():
    workflow = (ROOT / ".github" / "workflows" / "presentation-pages.yml").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    pages_url = "https://shaxntanu.github.io/Turbofan-RUL-Prediction-using-LSTM/presentation/"

    assert "web/public/presentation/. _site/presentation/" in workflow
    assert "web/public/results/fd001-prediction-intervals.png _site/results/" in workflow
    assert 'url=presentation/' in workflow
    assert "actions/upload-pages-artifact@v3" in workflow
    assert "actions/deploy-pages@v4" in workflow
    assert pages_url in workflow
    assert f"]({pages_url})" in readme
    assert "GitHub Pages presentation" in readme


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
