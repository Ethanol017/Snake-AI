"""Rescale TensorBoard scalar steps for fair cross-run comparisons."""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator
from tensorboard.compat.proto import event_pb2, summary_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter


def _load_scalar_events(log_dir: Path) -> dict[int, dict[str, tuple[float, float]]]:
    event_files = sorted(
        path for path in log_dir.iterdir() if path.is_file() and path.name.startswith("events.out.tfevents.")
    )
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {log_dir}")

    step_events: dict[int, dict[str, tuple[float, float]]] = defaultdict(dict)
    for event_file in event_files:
        accumulator = event_accumulator.EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        tags = accumulator.Tags().get("scalars", [])
        if not tags:
            continue

        for tag in tags:
            for scalar_event in accumulator.Scalars(tag):
                step = int(scalar_event.step)
                wall_time = float(scalar_event.wall_time)
                value = float(scalar_event.value)
                current = step_events[step].get(tag)
                if current is None or wall_time >= current[0]:
                    step_events[step][tag] = (wall_time, value)

    if not step_events:
        raise ValueError(f"No scalar events were found in {log_dir}")

    return step_events


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if any(output_dir.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    f"Output directory already exists and is not empty: {output_dir}. Use --overwrite to replace it."
                )
            shutil.rmtree(output_dir)
        else:
            shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


def rescale_tensorboard_steps(
    input_dir: Path,
    output_dir: Path,
    step_scale: int,
    step_offset: int = 0,
    overwrite: bool = False,
) -> tuple[int, int]:
    if step_scale <= 0:
        raise ValueError("step_scale must be a positive integer")

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    step_events = _load_scalar_events(input_dir)
    _prepare_output_dir(output_dir, overwrite)

    writer = EventFileWriter(str(output_dir))
    writer.add_event(event_pb2.Event(file_version="brain.Event:2"))

    total_events = 0
    total_steps = 0

    for step in sorted(step_events):
        scaled_step = int(step * step_scale + step_offset)
        for tag in sorted(step_events[step]):
            wall_time, value = step_events[step][tag]
            summary = summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=tag, simple_value=value)])
            writer.add_event(
                event_pb2.Event(
                    wall_time=wall_time,
                    step=scaled_step,
                    summary=summary,
                )
            )
            total_events += 1
        total_steps += 1

    writer.flush()
    writer.close()
    return total_events, total_steps


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="TensorBoard log directory to convert")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for rescaled events")
    parser.add_argument("--step-scale", type=int, required=True, help="Multiplier applied to the original step")
    parser.add_argument("--step-offset", type=int, default=0, help="Offset added after step scaling")
    parser.add_argument("--overwrite", action="store_true", help="Replace the output directory if it exists")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    total_events, total_steps = rescale_tensorboard_steps(
        input_dir=args.input,
        output_dir=args.output,
        step_scale=args.step_scale,
        step_offset=args.step_offset,
        overwrite=args.overwrite,
    )
    print(
        f"Rescaled {total_events} scalar events across {total_steps} steps "
        f"from {args.input} to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
