from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from io import BytesIO
from typing import Iterator, cast

import av
import cv2
import numpy as np
from av.container import InputContainer
from PIL import Image, ImageOps


@dataclass(frozen=True)
class VideoMeta:
    width: int
    height: int
    fps: float
    duration: float


def resize_max_side(
    image: np.ndarray,
    max_side: int,
) -> np.ndarray:
    """Resize while preserving aspect ratio."""

    h, w = image.shape[:2]
    longest = max(h, w)

    if longest <= max_side:
        return image

    scale = max_side / float(longest)

    new_w = max(2, int(round(w * scale)))
    new_h = max(2, int(round(h * scale)))

    return cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )


def make_even(image: np.ndarray) -> np.ndarray:
    """
    H.264/yuv420p requires even frame dimensions.
    """

    h, w = image.shape[:2]

    h2 = h - (h % 2)
    w2 = w - (w % 2)

    return image[:h2, :w2]


def probe_video(video_bytes: bytes) -> VideoMeta:
    container = cast(
        InputContainer,
        av.open(
            BytesIO(video_bytes),
            mode="r",
        ),
    )
    with container:
        stream = container.streams.video[0]

        width = int(stream.codec_context.width)
        height = int(stream.codec_context.height)

        fps = float(stream.average_rate) if stream.average_rate is not None else 30.0

        duration = 0.0

        if container.duration is not None:
            duration = float(container.duration / av.time_base)

        elif stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)

        elif stream.frames and fps > 0:
            duration = float(stream.frames / fps)

        return VideoMeta(
            width=width,
            height=height,
            fps=fps,
            duration=duration,
        )


def _frame_timestamp(
    frame: av.VideoFrame,
    stream: av.VideoStream,
) -> float:
    if frame.time is not None:
        return float(frame.time)

    if frame.pts is not None and stream.time_base is not None:
        return float(frame.pts * stream.time_base)

    return 0.0


def iter_video_segment(
    video_bytes: bytes,
    start_seconds: float,
    duration_seconds: float = 30.0,
    target_fps: int = 8,
    max_side: int = 480,
) -> Iterator[tuple[np.ndarray, float]]:
    """
    Decode only a requested time segment and temporally subsample it.

    Each yielded frame is RGB.
    """
    container = cast(
        InputContainer,
        av.open(
            BytesIO(video_bytes),
            mode="r",
        ),
    )
    with container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        if start_seconds > 0:
            container.seek(
                int(start_seconds * av.time_base),
                backward=True,
                any_frame=False,
            )

        end_seconds = start_seconds + duration_seconds
        frame_interval = 1.0 / target_fps
        next_sample_time = start_seconds

        for frame in container.decode(stream):
            timestamp = _frame_timestamp(frame, stream)

            if timestamp + 1e-6 < start_seconds:
                continue

            if timestamp > end_seconds:
                break

            if timestamp + 1e-6 < next_sample_time:
                continue

            rgb = frame.to_ndarray(format="rgb24")
            rgb = resize_max_side(rgb, max_side=max_side)
            rgb = make_even(rgb)

            yield rgb, timestamp

            while next_sample_time <= timestamp:
                next_sample_time += frame_interval


def decode_consecutive_pair(
    video_bytes: bytes,
    target_seconds: float,
    max_side: int = 640,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Decode two consecutive SOURCE frames near target_seconds.

    Unlike the dense-flow visualization, this does not intentionally
    skip frames. It is therefore appropriate for validating the
    two-frame Lucas-Kanade derivation.
    """
    container = cast(
        InputContainer,
        av.open(
            BytesIO(video_bytes),
            mode="r",
        ),
    )
    with container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        if target_seconds > 0:
            container.seek(
                int(target_seconds * av.time_base),
                backward=True,
                any_frame=False,
            )

        selected: list[tuple[np.ndarray, float]] = []

        for frame in container.decode(stream):
            timestamp = _frame_timestamp(frame, stream)

            if timestamp + 1e-6 < target_seconds:
                continue

            rgb = frame.to_ndarray(format="rgb24")
            rgb = resize_max_side(rgb, max_side=max_side)

            selected.append((rgb, timestamp))

            if len(selected) == 2:
                break

        if len(selected) != 2:
            raise ValueError(
                "Could not decode two consecutive frames at the "
                "selected validation time."
            )

        return (
            selected[0][0],
            selected[1][0],
            selected[0][1],
            selected[1][1],
        )


class MemoryMP4Writer:
    """
    Encode RGB numpy frames directly into an in-memory H.264 MP4.
    """

    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        bit_rate: int = 1_200_000,
    ) -> None:
        self.buffer = BytesIO()

        self.container = av.open(
            self.buffer,
            mode="w",
            format="mp4",
        )

        self.stream = self.container.add_stream(
            "libx264",
            rate=Fraction(int(fps), 1),
        )

        self.stream.width = int(width)
        self.stream.height = int(height)
        self.stream.pix_fmt = "yuv420p"
        self.stream.bit_rate = bit_rate

    def write(self, rgb: np.ndarray) -> None:
        rgb = make_even(rgb)

        frame = av.VideoFrame.from_ndarray(
            rgb,
            format="rgb24",
        )

        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> bytes:
        for packet in self.stream.encode():
            self.container.mux(packet)

        self.container.close()

        return self.buffer.getvalue()


def load_image_bytes(
    image_bytes: bytes,
    max_side: int = 1200,
) -> tuple[np.ndarray, tuple[int, int], float]:
    """
    Load RGB or grayscale input.

    Grayscale images are promoted to RGB for display, while downstream
    algorithms can convert them back to grayscale as needed.
    """

    with Image.open(BytesIO(image_bytes)) as pil_image:
        pil_image = ImageOps.exif_transpose(pil_image)

        original_width, original_height = pil_image.size

        pil_image = pil_image.convert("RGB")
        rgb = np.asarray(pil_image)

    resized = resize_max_side(
        rgb,
        max_side=max_side,
    )

    scale = resized.shape[1] / float(original_width)

    return (
        resized,
        (original_width, original_height),
        scale,
    )
