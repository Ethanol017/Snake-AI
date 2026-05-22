"""Stitch TensorBoard scalar timelines across resumed training sessions.

This script rewrites scalar events into a new log directory while preserving
the wall-time shape inside each event-file segment, then concatenates later
segments after earlier ones so the synthetic timeline looks continuous.
Overlapping steps are kept from the latest segment.
"""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator
from tensorboard.compat.proto import event_pb2, summary_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter


DEFAULT_INPUT_DIR = Path(r"logs\snake_dqn_20260507_215257")
DEFAULT_OUTPUT_DIR = Path(r"logs\snake_dqn_20260507_215257_normalized")
DEFAULT_SEGMENT_GAP_SECONDS = 1.0


@dataclass(frozen=True)
class SegmentData:
	path: Path
	step_events: dict[int, dict[str, tuple[float, float]]]
	start_wall_time: float
	end_wall_time: float


def _representative_step_wall_time(step_events: dict[str, tuple[float, float]]) -> float:
	return max(wall_time for wall_time, _value in step_events.values())


def _load_segments(log_dir: Path) -> list[SegmentData]:
	event_files = sorted(
		path for path in log_dir.iterdir() if path.is_file() and path.name.startswith("events.out.tfevents.")
	)
	if not event_files:
		raise FileNotFoundError(f"No TensorBoard event files found in {log_dir}")

	segments: list[SegmentData] = []
	for event_file in event_files:
		accumulator = event_accumulator.EventAccumulator(str(event_file), size_guidance={"scalars": 0})
		accumulator.Reload()
		tags = accumulator.Tags().get("scalars", [])
		if not tags:
			continue

		step_events: dict[int, dict[str, tuple[float, float]]] = defaultdict(dict)
		for tag in tags:
			for scalar_event in accumulator.Scalars(tag):
				step = int(scalar_event.step)
				wall_time = float(scalar_event.wall_time)
				value = float(scalar_event.value)
				current = step_events[step].get(tag)
				if current is None or wall_time >= current[0]:
					step_events[step][tag] = (wall_time, value)

		if not step_events:
			continue

		step_wall_times = [_representative_step_wall_time(events) for events in step_events.values()]
		segments.append(
			SegmentData(
				path=event_file,
				step_events=dict(step_events),
				start_wall_time=min(step_wall_times),
				end_wall_time=max(step_wall_times),
			)
		)

	if not segments:
		raise ValueError(f"No scalar events were found in {log_dir}")

	segments.sort(key=lambda segment: (segment.start_wall_time, segment.path.name))
	return segments


def _build_latest_step_owner(segments: list[SegmentData]) -> dict[int, int]:
	latest_owner: dict[int, int] = {}
	for segment_index, segment in enumerate(segments):
		for step in segment.step_events:
			latest_owner[step] = segment_index
	return latest_owner


def normalize_tensorboard_log(
	input_dir: Path,
	output_dir: Path,
	segment_gap_seconds: float = DEFAULT_SEGMENT_GAP_SECONDS,
	overwrite: bool = False,
):
	if not input_dir.exists():
		raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

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

	segments = _load_segments(input_dir)
	latest_step_owner = _build_latest_step_owner(segments)

	writer = EventFileWriter(str(output_dir))
	writer.add_event(event_pb2.Event(file_version="brain.Event:2"))

	total_events = 0
	total_steps = 0
	cumulative_offset = 0.0

	for segment_index, segment in enumerate(segments):
		surviving_steps = [step for step in sorted(segment.step_events) if latest_step_owner[step] == segment_index]
		if not surviving_steps:
			continue

		step_wall_times = {
			step: _representative_step_wall_time(segment.step_events[step]) for step in surviving_steps
		}
		segment_start_wall_time = min(step_wall_times.values())
		segment_end_wall_time = max(step_wall_times.values())

		for step in surviving_steps:
			synthetic_wall_time = cumulative_offset + (step_wall_times[step] - segment_start_wall_time)
			for tag in sorted(segment.step_events[step]):
				_wall_time, value = segment.step_events[step][tag]
				summary = summary_pb2.Summary(
					value=[summary_pb2.Summary.Value(tag=tag, simple_value=value)]
				)
				writer.add_event(
					event_pb2.Event(
						wall_time=synthetic_wall_time,
						step=int(step),
						summary=summary,
					)
				)
				total_events += 1

		total_steps += len(surviving_steps)
		cumulative_offset += (segment_end_wall_time - segment_start_wall_time) + segment_gap_seconds

	writer.flush()
	writer.close()
	return total_events, total_steps, len(segments)


def _parse_args():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_DIR, help="TensorBoard log directory to normalize")
	parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for normalized events")
	parser.add_argument(
		"--segment-gap-seconds",
		type=float,
		default=DEFAULT_SEGMENT_GAP_SECONDS,
		help="Synthetic wall-time gap inserted between stitched segments",
	)
	parser.add_argument("--overwrite", action="store_true", help="Replace the output directory if it already exists")
	return parser.parse_args()


def main() -> int:
	args = _parse_args()
	total_events, total_steps, total_segments = normalize_tensorboard_log(
		input_dir=args.input,
		output_dir=args.output,
		segment_gap_seconds=args.segment_gap_seconds,
		overwrite=args.overwrite,
	)
	print(
		f"Stitched {total_events} scalar events across {total_steps} steps from {total_segments} segments "
		f"in {args.input} to {args.output}"
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
