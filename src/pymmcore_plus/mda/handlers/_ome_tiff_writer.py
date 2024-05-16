"""OME.TIFF writer for MDASequences.

Borrowed from the pattern shared by Christoph:
https://forum.image.sc/t/how-to-create-an-image-series-ome-tiff-from-python/42730/7

Note, these are the valid axis keys tifffile:
Supported by OME-XML
    X : width** (image width)
    Y : height** (image length)
    Z : depth** (image depth)
    T : time** (time series)
    C : channel** (acquisition path or emission wavelength)
    Modulo axes:
    S : sample** (color space and extra samples)
    A : angle** (OME)
    P : phase** (OME. In LSM, **P** maps to **position**)
    R : tile** (OME. Region, position, or mosaic)
    H : lifetime** (OME. Histogram)
    E : lambda** (OME. Excitation wavelength)
    Q : other** (OME)
Not Supported by OME-XML
    I : sequence** (generic sequence of images, frames, planes, pages)
    L : exposure** (FluoView)
    V : event** (FluoView)
    M : mosaic** (LSM 6)

Rules:
- all axes must be one of TZCYXSAPRHEQ
- len(axes) must equal len(shape)
- dimensions (order) must end with YX or YXS
- no axis can be repeated
- no more than 8 dimensions (or 9 if 'S' is included)

Non-OME (ImageJ) hyperstack axes MUST be in TZCYXS order
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import ome_types.model as m
from tifffile import tiffcomment

from ._5d_writer_base import _5DWriterBase

if TYPE_CHECKING:
    import useq

IMAGEJ_AXIS_ORDER = "tzcyxs"


class OMETiffWriter(_5DWriterBase[np.memmap]):
    """MDA handler that writes to a 5D OME-TIFF file.

    Positions will be split into different files.

    Data is memory-mapped to disk using numpy.memmap via tifffile.  Tifffile handles
    the OME-TIFF format.

    Parameters
    ----------
    filename : Path | str
        The filename to write to.  Must end with '.ome.tiff' or '.ome.tif'.
    """

    def __init__(self, filename: Path | str) -> None:
        try:
            import tifffile  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "tifffile is required to use this handler. "
                "Please `pip install tifffile`."
            ) from e

        self._filename = str(filename)
        if not self._filename.endswith((".tiff", ".tif")):  # pragma: no cover
            raise ValueError("filename must end with '.tiff' or '.tif'")
        self._is_ome = ".ome.tif" in self._filename

        super().__init__()

    def sequenceStarted(self, seq: useq.MDASequence) -> None:
        super().sequenceStarted(seq)
        # Non-OME (ImageJ) hyperstack axes MUST be in TZCYXS order
        # so we reorder the ordered position_sizes dicts.  This will ensure
        # that the array indices created from event.index are in the correct order.
        if not self._is_ome:
            self._position_sizes = [
                {k: x[k] for k in IMAGEJ_AXIS_ORDER if k.lower() in x}
                for x in self.position_sizes
            ]

    def write_frame(
        self, ary: np.memmap, index: tuple[int, ...], frame: np.ndarray
    ) -> None:
        """Write a frame to the file."""
        super().write_frame(ary, index, frame)
        ary.flush()

    def new_array(
        self, position_key: str, dtype: np.dtype, sizes: dict[str, int]
    ) -> np.memmap:
        """Create a new tifffile file and memmap for this position."""
        from tifffile import imwrite, memmap

        _, shape = zip(*sizes.items())

        # if there are multiple positions, create a folder named as the filename
        # and save each position as a separate file
        if (seq := self.current_sequence) and seq.sizes.get("p", 1) > 1:
            folder_path = Path(self._filename)
            # create the parent directory using the filename
            folder_path.mkdir(parents=True, exist_ok=True)
            # update the position file name to include the position key
            ext = ".ome.tif" if self._is_ome else ".tif"
            name = folder_path.name.replace(ext, f"_{position_key}{ext}")
            # create the full path
            fname = folder_path / name
        # if there is only one position, save the file as the filename
        else:
            fname = Path(self._filename)

        # create parent directories if they don't exist
        # Path(fname).parent.mkdir(parents=True, exist_ok=True)
        # write empty file to disk
        imwrite(
            fname,
            shape=shape,
            dtype=dtype,
            # imagej=not self._is_ome,
            # ome=self._is_ome,
        )

        # memory-mapped NumPy array of image data stored in TIFF file.
        mmap = memmap(fname, dtype=dtype)
        # This line is important, as tifffile.memmap appears to lose singleton dims
        mmap.shape = shape

        # store the few info for this position in th _ome_meta dict for later use
        # in finalize_metadata
        self.frame_metadatas[position_key] = {
            "file": fname,
            "sizes": sizes,
            "shape": shape,
            "dtype": str(dtype),
        }

        return mmap  # type: ignore

    def store_frame_metadata(self, key: str, event: useq.MDAEvent, meta: dict) -> None:
        """Called during each frameReady event to store OME metadata for the frame.

        Parameters
        ----------
        key : str
            The position key for the frame (e.g. "p0" for the first position).
        event : useq.MDAEvent
            The event that triggered the frameReady signal.
        meta : dict
            Metadata associated with the frame.
        """
        # get the OME channel and plane for the event
        channel = self._get_ome_channel(event)
        plane = self._get_ome_plane(event, meta.get("ElapsedTime-ms", 0.0))
        # self.frame_metadatas[key] is added in the 'new_array' method
        # if the 'pixels' key is not present, we need to add the pixel metadata
        if "pixels" not in self.frame_metadatas[key]:
            self._set_ome_pixel(key, meta, channel, plane)
        # otherwise, we only need to append the new channel and plane
        else:
            self._update_ome_pixel(key, channel, plane)
            if channel not in self.frame_metadatas[key]["pixels"]["channels"]:
                self.frame_metadatas[key]["pixels"]["channels"].append(channel)
            self.frame_metadatas[key]["pixels"]["planes"].append(plane)

    def _get_ome_channel(self, event: useq.MDAEvent) -> m.Channel:
        """Return the OME channel from the event."""
        if event.channel is None:
            return m.Channel(id="Channel:0", name="Channel:0", samples_per_pixel=1)

        return m.Channel(
            id=f"Channel:{event.index.get('c', 0)}",
            name=f"{event.channel.group}:{event.channel.config}",
            samples_per_pixel=1,
        )

    def _get_ome_plane(self, event: useq.MDAEvent, elapsed_time_ms: float) -> m.Plane:
        """Return the OME plane from the event."""
        return m.Plane(
            the_c=event.index.get("c", 0),
            the_t=event.index.get("t", 0),
            the_z=event.index.get("z", 0),
            exposure_time=event.exposure,
            exposure_time_unit="ms",
            position_x=event.x_pos,
            position_x_unit="µm",
            position_y=event.y_pos,
            position_y_unit="µm",
            position_z=event.z_pos,
            position_z_unit="µm",
            delta_t=elapsed_time_ms,
            delta_t_unit="ms",
        )

    def _set_ome_pixel(
        self, key: str, meta: dict, channel: m.Channel, plane: m.Plane
    ) -> None:
        # get z step size from the sequence if it is a relative z plan
        z_step = None
        if (
            self.current_sequence is not None
            and self.current_sequence.z_plan
            and hasattr(self.current_sequence.z_plan, "step")
        ):
            z_step = self.current_sequence.z_plan.step

        frame_meta = self.frame_metadatas[key]
        frame_meta["pixels"] = {
            "dimension_order": "XYCZT",  # TODO: should this be dynamic?
            "physical_size_x": meta.get("PixelSizeUm"),
            "physical_size_x_unit": "µm",
            "physical_size_y": meta.get("PixelSizeUm"),
            "physical_size_y_unit": "µm",
            "physical_size_z": z_step,
            "physical_size_z_unit": "µm",
            "size_t": frame_meta["sizes"].get("t", "1"),
            "size_z": frame_meta["sizes"].get("z", "1"),
            "size_c": frame_meta["sizes"].get("c", "1"),
            "size_x": frame_meta["shape"][-2],
            "size_y": frame_meta["shape"][-1],
            "type": frame_meta["dtype"],
            "channels": [channel],
            "planes": [plane],
            # "tiff_data_blocks" will be added in finalize_metadata
        }

    def _update_ome_pixel(self, key: str, channel: m.Channel, plane: m.Plane) -> None:
        if channel not in self.frame_metadatas[key]["pixels"]["channels"]:
            self.frame_metadatas[key]["pixels"]["channels"].append(channel)
        self.frame_metadatas[key]["pixels"]["planes"].append(plane)

    def finalize_metadata(self) -> None:
        """Called during sequenceFinished before clearing sequence metadata.

        Subclasses may override this method to flush any accumulated frame metadata to
        disk at the end of the sequence.
        """
        # add the ome metadata to each position file
        for key, data in self.frame_metadatas.items():
            file = data["file"]
            pixels = data["pixels"]
            pixels["tiff_data_blocks"] = [m.TiffData(plane_count=len(pixels["planes"]))]
            image = m.Image(id=f"Image:{key[1:]}", name=f"Image:{key}", pixels=pixels)
            ome = m.OME(images=[image])
            ome_xml = ome.to_xml()
            tiffcomment(file, ome_xml.encode("utf-8"))
