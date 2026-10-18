from __future__ import annotations

from pathlib import Path

import manimgx as m


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Ref" / "CMAPSSData" / "train_FD001.txt"
BACKGROUND = "#10171B"
FOREGROUND = "#F1F5F3"
MUTED = "#A8B7B6"
TEAL = "#49D6C5"
ORANGE = "#FFB45B"
BLUE = "#87BDF5"
GREEN = "#A6DB8A"


def load_engine_trace() -> tuple[list[int], list[float]]:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Expected the bundled FD001 training data at {DATA_PATH}")

    cycles: list[int] = []
    sensor_4: list[float] = []
    for line in DATA_PATH.read_text(encoding="ascii").splitlines():
        fields = line.split()
        if fields and int(fields[0]) == 1:
            cycles.append(int(fields[1]))
            sensor_4.append(float(fields[8]))
    if not cycles:
        raise ValueError("FD001 training data contains no rows for engine 1")
    return cycles, sensor_4


class RULMethodScene(m.Scene):
    def construct(self) -> None:
        self.camera.background_color = BACKGROUND
        cycles, readings = load_engine_trace()
        lifetime = max(cycles)
        target_cycle = 163
        window_size = 30
        if target_cycle not in cycles or lifetime < target_cycle:
            raise ValueError("The illustrated target cycle must exist in engine 1's training trace")

        title = m.Text(
            "From engine cycles to a remaining-life label",
            font_size=34,
            color=FOREGROUND,
        ).move_to([0, 3.28, 0])
        subtitle = m.Text(
            "NASA C-MAPSS FD001 | engine 1 | measured sensor_4",
            font_size=19,
            color=MUTED,
        ).move_to([0, 2.72, 0])
        self.play(m.FadeIn(title), m.FadeIn(subtitle), run_time=0.8)

        x_left, x_right = -5.65, 5.65
        y_bottom, y_top = -1.48, 1.48
        reading_min, reading_max = min(readings), max(readings)

        def point(cycle: int, reading: float) -> m.np.ndarray:
            x = x_left + (cycle - cycles[0]) / (lifetime - cycles[0]) * (x_right - x_left)
            y = y_bottom + (reading - reading_min) / max(reading_max - reading_min, 1e-9) * (y_top - y_bottom)
            return m.np.array([x, y, 0.0])

        axis_y = -1.72
        axes = m.VGroup(
            m.Line([x_left, axis_y, 0], [x_right, axis_y, 0], color=MUTED, stroke_width=1.5),
            m.Line([x_left, y_bottom, 0], [x_left, y_top, 0], color=MUTED, stroke_width=1.5),
        )
        ticks = m.VGroup()
        for cycle in (1, 96, lifetime):
            x = x_left + (cycle - 1) / (lifetime - 1) * (x_right - x_left)
            ticks.add(m.Line([x, axis_y - 0.05, 0], [x, axis_y + 0.05, 0], color=MUTED))
            ticks.add(m.Text(str(cycle), font_size=23, color=MUTED).move_to([x, -2.04, 0]))
        chart_label = m.Text(
            "sensor_4 (raw values; vertical axis scaled for display)",
            font_size=16,
            color=MUTED,
        ).move_to([-2.85, 1.83, 0])
        trace = m.VGroup(
            *[
                m.Line(point(cycles[i], readings[i]), point(cycles[i + 1], readings[i + 1]), color=TEAL, stroke_width=3)
                for i in range(len(cycles) - 1)
            ]
        )
        axes_group = m.VGroup(axes, ticks, chart_label)
        self.play(m.Create(axes_group), m.Create(trace), run_time=1.6)

        def window_center(end_cycle: int) -> float:
            start_cycle = end_cycle - window_size + 1
            return (point(start_cycle, reading_min)[0] + point(end_cycle, reading_min)[0]) / 2

        window_width = (x_right - x_left) * (window_size - 1) / (lifetime - 1)
        selection = m.Rectangle(
            width=window_width,
            height=y_top - y_bottom + 0.08,
            stroke_color=ORANGE,
            stroke_width=2.5,
            fill_color=ORANGE,
            fill_opacity=0.14,
        ).move_to([window_center(window_size), (y_top + y_bottom) / 2, 0])
        selected_dot = m.Dot(point(window_size, readings[window_size - 1]), radius=0.075, color=ORANGE)
        window_caption = m.Text(
            "Sliding 30-cycle input window",
            font_size=21,
            color=ORANGE,
        ).move_to([0, -2.6, 0])
        self.play(m.FadeIn(selection), m.FadeIn(selected_dot), m.FadeIn(window_caption), run_time=0.6)
        self.play(
            selection.animate.move_to([window_center(target_cycle), (y_top + y_bottom) / 2, 0]),
            selected_dot.animate.move_to(point(target_cycle, readings[target_cycle - 1])),
            run_time=3.2,
        )
        endpoint_caption = m.Text(
            "Selected window ends at cycle 163",
            font_size=21,
            color=ORANGE,
        ).move_to([0, -2.6, 0])
        self.play(m.Transform(window_caption, endpoint_caption), run_time=0.5)
        self.wait(0.8)

        chart_group = m.VGroup(subtitle, axes_group, trace, selection, selected_dot, window_caption)
        next_title = m.Text(
            "How the window becomes an RUL estimate",
            font_size=34,
            color=FOREGROUND,
        ).move_to([0, 3.28, 0])
        self.play(m.FadeOut(chart_group), m.Transform(title, next_title), run_time=0.8)

        box_y = 1.25
        box_width, box_height = 3.05, 1.65
        centers = (-4.35, 0, 4.35)
        box_colors = (TEAL, BLUE, GREEN)
        labels = (
            ("MODEL INPUT", "30 cycles x 18 features", "15 sensors + 3 settings"),
            ("REGRESSOR", "Ridge or LSTM", "schematic model stage"),
            ("ESTIMATE", "RUL + interval", "schematic output"),
        )
        blocks = m.VGroup()
        for center, color, (heading, main, detail) in zip(centers, box_colors, labels):
            outline = m.Rectangle(
                width=box_width,
                height=box_height,
                stroke_color=color,
                stroke_width=2,
                fill_color=BACKGROUND,
                fill_opacity=1,
            ).move_to([center, box_y, 0])
            heading_text = m.Text(heading, font_size=15, color=color).move_to([center, 1.78, 0])
            main_text = m.Text(main, font_size=19, color=FOREGROUND).move_to([center, 1.18, 0])
            detail_text = m.Text(detail, font_size=14, color=MUTED).move_to([center, 0.7, 0])
            blocks.add(m.VGroup(outline, heading_text, main_text, detail_text))

        arrows = m.VGroup(
            m.Arrow([-2.65, box_y, 0], [-1.7, box_y, 0], color=MUTED, buff=0.02, stroke_width=2),
            m.Arrow([1.7, box_y, 0], [2.65, box_y, 0], color=MUTED, buff=0.02, stroke_width=2),
        )
        target = m.Text(
            "Training label at cycle 163:  min(125, 192 - 163) = 29 cycles",
            font_size=23,
            color=ORANGE,
        ).move_to([0, -0.15, 0])
        calibration = m.Text(
            "95% nominal split conformal: q is calibrated from validation residuals",
            font_size=18,
            color=FOREGROUND,
        ).move_to([0, -1.0, 0])
        interval = m.Text(
            "Interval = [max(0, y_hat - q), min(125, y_hat + q)]",
            font_size=20,
            color=TEAL,
        ).move_to([0, -1.58, 0])
        caveat = m.Text(
            "Coverage is descriptive, not guaranteed under engine or operating-condition shift.",
            font_size=15,
            color=MUTED,
        ).move_to([0, -2.55, 0])
        schematic = m.Text(
            "Illustration only: no sample-specific model prediction is shown.",
            font_size=15,
            color=MUTED,
        ).move_to([0, -2.95, 0])
        credit = m.Text("Rendered with ManimGx", font_size=16, color=MUTED).move_to([4.75, -3.62, 0])

        self.play(m.Create(blocks), m.Create(arrows), run_time=1.2)
        self.play(m.FadeIn(target), m.FadeIn(calibration), m.FadeIn(interval), run_time=0.9)
        self.play(m.FadeIn(caveat), m.FadeIn(schematic), m.FadeIn(credit), run_time=0.7)
        self.wait(2)
