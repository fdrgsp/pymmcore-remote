from __future__ import annotations

import time
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Iterator,
    Mapping,
    NamedTuple,
    cast,
)

from pyfirmata2 import Arduino
from pyfirmata2.pyfirmata2 import Pin
from rich import print
from useq import HardwareAutofocus, MDAEvent, MDASequence

from pymmcore_plus._logger import logger
from pymmcore_plus._util import retry
from pymmcore_plus.core._constants import PixelType
from pymmcore_plus.core._sequencing import SequencedEvent

from ._protocol import PMDAEngine

if TYPE_CHECKING:
    from typing import TypedDict

    from numpy.typing import NDArray

    from pymmcore_plus.core import CMMCorePlus, Metadata

    from ._protocol import PImagePayload

    # currently matching keys from metadata from AcqEngJ
    SummaryMetadata = TypedDict(
        "SummaryMetadata",
        {
            "DateAndTime": str,
            "PixelType": str,
            "PixelSize_um": float,
            "PixelSizeAffine": str,
            "Core-XYStage": str,
            "Core-Focus": str,
            "Core-Autofocus": str,
            "Core-Camera": str,
            "Core-Galvo": str,
            "Core-ImageProcessor": str,
            "Core-SLM": str,
            "Core-Shutter": str,
            "AffineTransform": str,
        },
    )

NMM_METADATA_KEY = "napari_micromanager"
STIMULATION = "stimulation"


class AutofocusState(NamedTuple):
    """Nametuple to store the state of the autofocus device.

    Attributes
    ----------
    was_engaged : bool
        Whether the autofocus device was engaged at the start of the sequence.
    re_engage : bool
        Whether the autofocus device should be re-engaged after the autofocus action.
        This will be set to `True` or `False` after the autofocus action is executed.
        If it fails, it will be set to `False`.
    """

    was_engaged: bool = False
    re_engage: bool = False

    def __bool__(self) -> bool:
        return bool(self.was_engaged and self.re_engage)


class MDAEngine(PMDAEngine):
    """The default MDAengine that ships with pymmcore-plus.

    This implements the [`PMDAEngine`][pymmcore_plus.mda.PMDAEngine] protocol, and
    uses a [`CMMCorePlus`][pymmcore_plus.CMMCorePlus] instance to control the hardware.

    Attributes
    ----------
    mmcore: CMMCorePlus
        The `CMMCorePlus` instance to use for hardware control.
    use_hardware_sequencing : bool
        Whether to use hardware sequencing if possible. If `True`, the engine will
        attempt to combine MDAEvents into a single `SequencedEvent` if
        [`core.canSequenceEvents()`][pymmcore_plus.CMMCorePlus.canSequenceEvents]
        reports that the events can be sequenced. This can be set after instantiation.
        By default, this is `False`, in order to avoid unexpected behavior, particularly
        in testing and demo scenarios.  But in many "real world" scenarios, this can be
        set to `True` to improve performance.
    arduino_board : Arduino | None
        The Arduino board to use for LED stimulation. If `None`, no LED stimulation will
        be performed.
    arduino_led_pin : Pin | None
        The pin on the Arduino board to use for LED stimulation. If `None`, no LED
        stimulation will be performed.
    """

    def __init__(
        self,
        mmc: CMMCorePlus,
        use_hardware_sequencing: bool = True,
        arduino_board: Arduino | None = None,
        arduino_led_pin: Pin | None = None,
    ) -> None:
        self._mmc = mmc
        self.use_hardware_sequencing = use_hardware_sequencing

        # used to check if the hardware autofocus is engaged when the sequence begins.
        # if it is, we will re-engage it after the autofocus action (if successful).
        self._af_was_engaged: bool = False
        # used to store the success of the last _execute_autofocus call
        self._af_succeeded: bool = False

        # used for one_shot autofocus to store the z correction for each position index.
        # map of {position_index: z_correction}
        self._z_correction: dict[int | None, float] = {}

        # This is used to determine whether we need to re-enable autoshutter after
        # the sequence is done (assuming a event.keep_shutter_open was requested)
        # Note: getAutoShutter() is True when no config is loaded at all
        self._autoshutter_was_set: bool = self._mmc.getAutoShutter()

        # for LED stimulation
        self._arduino_board = arduino_board
        self._arduino_led_pin = arduino_led_pin
        self._exec_stimulation: dict[int, tuple[int, int]] = {}

    @property
    def mmcore(self) -> CMMCorePlus:
        """The `CMMCorePlus` instance to use for hardware control."""
        return self._mmc

    # ===================== Arduino Properties =========================
    def setArduinoBoard(self, arduino_board: Arduino | None) -> None:
        """Set the Arduino board to use for LED stimulation."""
        self._arduino_board = arduino_board

    def setArduinoLedPin(self, arduino_led_pin: Pin | None) -> None:
        """Set the pin on the Arduino board to use for LED stimulation."""
        self._arduino_led_pin = arduino_led_pin

    # ===================== Protocol Implementation =====================

    def setup_sequence(self, sequence: MDASequence) -> Mapping[str, Any]:
        """Setup the hardware for the entire sequence."""
        # clear z_correction for new sequence
        self._z_correction.clear()

        if not self._mmc:  # pragma: no cover
            from pymmcore_plus.core import CMMCorePlus

            self._mmc = CMMCorePlus.instance()

        # get if the autofocus is engaged at the start of the sequence
        self._af_was_engaged = self._mmc.isContinuousFocusLocked()

        if px_size := self._mmc.getPixelSizeUm():
            self._update_grid_fov_sizes(px_size, sequence)

        self._autoshutter_was_set = self._mmc.getAutoShutter()

        # Arduino LED Setup________________________________________________
        self._exec_stimulation.clear()
        if self._arduino_board is not None and self._arduino_led_pin is not None:
            self._setup_stimulation_events(sequence)

        print()
        print("_______________________")
        print("Arduino LED Setup")
        print(self._exec_stimulation)
        print("_______________________")
        # _________________________________________________________________

        return self.get_summary_metadata()

    def _setup_stimulation_events(self, sequence: MDASequence) -> None:
        # switch off the LED if it was on
        self._arduino_led_pin = cast(Pin, self._arduino_led_pin)
        self._arduino_led_pin.write(0.0)
        # get metadata from the sequence and store it in the _exec_stimulation
        meta = cast(dict, sequence.metadata.get(NMM_METADATA_KEY, {}))
        stim_meta = cast(dict, meta.get(STIMULATION, {}))
        pulse_on_frame = stim_meta.get("pulse_on_frame", None)
        led_pulse_duration = stim_meta.get("led_pulse_duration", None)
        if pulse_on_frame is not None and led_pulse_duration is not None:
            # ctrate the _exec_stimulation dict with info about when to pulse the
            # LED, for how long and with what power
            # e.g. {frame: led_power, led_pulse_duration}
            pulse_on_frame = cast(dict, pulse_on_frame)
            for k, v in pulse_on_frame.items():
                self._exec_stimulation[k] = (v, led_pulse_duration)

    def get_summary_metadata(self) -> SummaryMetadata:
        """Get the summary metadata for the sequence."""
        pt = PixelType.for_bytes(
            self._mmc.getBytesPerPixel(), self._mmc.getNumberOfComponents()
        )
        affine = self._mmc.getPixelSizeAffine(True)  # true == cached

        return {
            "DateAndTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
            "PixelType": str(pt),
            "PixelSize_um": self._mmc.getPixelSizeUm(),
            "PixelSizeAffine": ";".join(str(x) for x in affine),
            "Core-XYStage": self._mmc.getXYStageDevice(),
            "Core-Focus": self._mmc.getFocusDevice(),
            "Core-Autofocus": self._mmc.getAutoFocusDevice(),
            "Core-Camera": self._mmc.getCameraDevice(),
            "Core-Galvo": self._mmc.getGalvoDevice(),
            "Core-ImageProcessor": self._mmc.getImageProcessorDevice(),
            "Core-SLM": self._mmc.getSLMDevice(),
            "Core-Shutter": self._mmc.getShutterDevice(),
            "AffineTransform": "Undefined",
        }

    def _update_grid_fov_sizes(self, px_size: float, sequence: MDASequence) -> None:
        *_, x_size, y_size = self._mmc.getROI()
        fov_width = x_size * px_size
        fov_height = y_size * px_size

        if sequence.grid_plan:
            sequence.grid_plan.fov_width = fov_width
            sequence.grid_plan.fov_height = fov_height

        # set fov to any stage positions sequences
        for p in sequence.stage_positions:
            if p.sequence and p.sequence.grid_plan:
                p.sequence.grid_plan.fov_height = fov_height
                p.sequence.grid_plan.fov_width = fov_width

    def setup_event(self, event: MDAEvent) -> None:
        """Set the system hardware (XY, Z, channel, exposure) as defined in the event.

        Parameters
        ----------
        event : MDAEvent
            The event to use for the Hardware config
        """
        if isinstance(event, SequencedEvent):
            self.setup_sequenced_event(event)
        else:
            self.setup_single_event(event)
        self._mmc.waitForSystem()

    def exec_event(self, event: MDAEvent) -> Iterable[PImagePayload]:
        """Execute an individual event and return the image data."""
        print()

        action = getattr(event, "action", None)
        if isinstance(action, HardwareAutofocus):
            print(f"***Autofocus Event: {event.index}, action: {action}***")
            # skip if no autofocus device is found
            if not self._mmc.getAutoFocusDevice():
                logger.warning("No autofocus device found. Cannot execute autofocus.")
                return ()

            try:
                # execute hardware autofocus
                new_correction = self._execute_autofocus(action)
                self._af_succeeded = True
            except RuntimeError as e:
                logger.warning("Hardware autofocus failed. %s", e)
                self._af_succeeded = False
            else:
                # store correction for this position index
                p_idx = event.index.get("p", None)
                self._z_correction[p_idx] = new_correction + self._z_correction.get(
                    p_idx, 0.0
                )
            return ()

        # if the autofocus was engaged at the start of the sequence AND autofocus action
        # did not fail, re-engage it. NOTE: we need to do that AFTER the runner calls
        # `setup_event`, so we can't do it inside the exec_event autofocus action above.
        if self._af_was_engaged and self._af_succeeded:
            self._mmc.enableContinuousFocus(True)

        elif isinstance(event, SequencedEvent):
            yield from self.exec_sequenced_event(event)
        else:
            yield from self.exec_single_event(event)

    def event_iterator(self, events: Iterable[MDAEvent]) -> Iterator[MDAEvent]:
        """Event iterator that merges events for hardware sequencing if possible.

        This wraps `for event in events: ...` inside `MDARunner.run()` and combines
        sequenceable events into an instance of `SequencedEvent` if
        `self.use_hardware_sequencing` is `True`.
        """
        if not self.use_hardware_sequencing:
            yield from events
            return

        seq: list[MDAEvent] = []
        for event in events:
            # if the sequence is empty or the current event can be sequenced with the
            # previous event, add it to the sequence
            if not seq or self._mmc.canSequenceEvents(seq[-1], event, len(seq)):
                seq.append(event)
            else:
                # otherwise, yield a SequencedEvent if the sequence has accumulated
                # more than one event, otherwise yield the single event
                yield seq[0] if len(seq) == 1 else SequencedEvent.create(seq)
                # add this current event and start a new sequence
                seq = [event]
        # yield any remaining events
        if seq:
            yield seq[0] if len(seq) == 1 else SequencedEvent.create(seq)

    # ===================== Regular Events =====================

    def setup_single_event(self, event: MDAEvent) -> None:
        """Setup hardware for a single (non-sequenced) event.

        This method is not part of the PMDAEngine protocol (it is called by
        `setup_event`, which *is* part of the protocol), but it is made public
        in case a user wants to subclass this engine and override this method.
        """
        if event.keep_shutter_open:
            ...

        if event.x_pos is not None or event.y_pos is not None:
            self._set_event_position(event)
        if event.z_pos is not None:
            self._set_event_z(event)

        if event.channel is not None:
            try:
                self._mmc.setConfig(event.channel.group, event.channel.config)
            except Exception as e:
                logger.warning("Failed to set channel. %s", e)
        if event.exposure is not None:
            try:
                self._mmc.setExposure(event.exposure)
            except Exception as e:
                logger.warning("Failed to set exposure. %s", e)

        if (
            # (if autoshutter wasn't set at the beginning of the sequence
            # then it never matters...)
            self._autoshutter_was_set
            # if we want to leave the shutter open after this event, and autoshutter
            # is currently enabled...
            and event.keep_shutter_open
            and self._mmc.getAutoShutter()
        ):
            # we have to disable autoshutter and open the shutter
            self._mmc.setAutoShutter(False)
            self._mmc.setShutterOpen(True)

    def exec_single_event(self, event: MDAEvent) -> Iterator[PImagePayload]:
        """Execute a single (non-triggered) event and return the image data.

        This method is not part of the PMDAEngine protocol (it is called by
        `exec_event`, which *is* part of the protocol), but it is made public
        in case a user wants to subclass this engine and override this method.
        """
        # execute stimulation if the event if it is in the sequence metadata
        # if self._arduino_board is not None and self._arduino_led_pin is not None:
        if t_index := event.index.get("t", None):
            if t_index in self._exec_stimulation:
                self._exec_led_stimulation(t_index, event)

        print(f"***Snap Event: {event.index}***\n")

        try:
            self._mmc.snapImage()
        except Exception as e:
            logger.warning("Failed to snap image. %s", e)
            return ()
        if not event.keep_shutter_open:
            self._mmc.setShutterOpen(False)
        yield ImagePayload(self._mmc.getImage(), event, self.get_frame_metadata())

    def get_frame_metadata(
        self, meta: Metadata | None = None, channel_index: int | None = None
    ) -> dict[str, Any]:
        # TODO:

        # this is not a very fast method, and it is called for every frame.
        # Nico Stuurman has suggested that it was a mistake for MM to pull so much
        # metadata for every frame.  So we'll begin with a more conservative approach.

        # while users can now simply re-implement this method,
        # consider coming up with a user-configurable way to specify needed metadata

        # rather than using self._mmc.getTags (which mimics MM) we pull a smaller
        # amount of metadata.
        # If you need more than this, either override or open an issue.

        tags = dict(meta) if meta else {}
        for dev, label, val in self._mmc.getSystemStateCache():
            tags[f"{dev}-{label}"] = val

        # these are added by AcqEngJ
        # yyyy-MM-dd HH:mm:ss.mmmmmm  # NOTE AcqEngJ omits microseconds
        tags["Time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        # used by Runner
        tags["PerfCounter"] = time.perf_counter()
        return tags

    def teardown_event(self, event: MDAEvent) -> None:
        """Teardown state of system (hardware, etc.) after `event`."""
        # autoshutter was set at the beginning of the sequence, and this event
        # doesn't want to leave the shutter open.  Re-enable autoshutter.
        if not event.keep_shutter_open and self._autoshutter_was_set:
            self._mmc.setAutoShutter(True)

    def teardown_sequence(self, sequence: MDASequence) -> None:
        """Perform any teardown required after the sequence has been executed."""
        # close the current shutter at the end of the sequence
        if self._mmc.getShutterDevice():
            self._mmc.setShutterOpen(False)

    # ===================== Sequenced Events =====================

    def setup_sequenced_event(self, event: SequencedEvent) -> None:
        """Setup hardware for a sequenced (triggered) event.

        This method is not part of the PMDAEngine protocol (it is called by
        `setup_event`, which *is* part of the protocol), but it is made public
        in case a user wants to subclass this engine and override this method.
        """
        core = self._mmc
        cam_device = self._mmc.getCameraDevice()

        if event.exposure_sequence:
            core.loadExposureSequence(cam_device, event.exposure_sequence)
        if event.x_sequence:  # y_sequence is implied and will be the same length
            stage = core.getXYStageDevice()
            core.loadXYStageSequence(stage, event.x_sequence, event.y_sequence)
        if event.z_sequence:
            # these notes are from Nico Stuurman in AcqEngJ
            # https://github.com/micro-manager/AcqEngJ/pull/108
            # at least some zStages freak out (in this case, NIDAQ board) when you
            # try to load a sequence while the sequence is still running.  Nothing in
            # the engine stops a stage sequence if all goes well.
            # Stopping a sequence if it is not running hopefully will not harm anyone.
            zstage = core.getFocusDevice()
            core.stopStageSequence(zstage)
            core.loadStageSequence(zstage, event.z_sequence)
        if prop_seqs := event.property_sequences(core):
            for (dev, prop), value_sequence in prop_seqs.items():
                core.loadPropertySequence(dev, prop, value_sequence)

        # TODO: SLM

        # preparing a Sequence while another is running is dangerous.
        if core.isSequenceRunning():
            self._await_sequence_acquisition()
        core.prepareSequenceAcquisition(cam_device)

        # start sequences or set non-sequenced values
        if event.x_sequence:
            core.startXYStageSequence(stage)
        elif event.x_pos is not None or event.y_pos is not None:
            self._set_event_position(event)

        if event.z_sequence:
            core.startStageSequence(zstage)
        elif event.z_pos is not None:
            self._set_event_z(event)

        if event.exposure_sequence:
            core.startExposureSequence(cam_device)
        elif event.exposure is not None:
            core.setExposure(event.exposure)

        if prop_seqs:
            for dev, prop in prop_seqs:
                core.startPropertySequence(dev, prop)
        elif event.channel is not None:
            core.setConfig(event.channel.group, event.channel.config)

    def _await_sequence_acquisition(
        self, timeout: float = 5.0, poll_interval: float = 0.2
    ) -> None:
        tot = 0.0
        self._mmc.stopSequenceAcquisition()
        while self._mmc.isSequenceRunning():
            time.sleep(poll_interval)
            tot += poll_interval
            if tot >= timeout:
                raise TimeoutError("Failed to stop running sequence")

    def post_sequence_started(self, event: SequencedEvent) -> None:
        """Perform any actions after startSequenceAcquisition has been called.

        This method is available to subclasses in case they need to perform any
        actions after a hardware-triggered sequence has been started (i.e. after
        core.startSequenceAcquisition has been called).

        The default implementation does nothing.
        """

    def _exec_led_stimulation(self, t_index: int, event: MDAEvent) -> None:
        """Execute LED stimulation."""
        self._arduino_board = cast(Arduino, self._arduino_board)
        self._arduino_led_pin = cast(Pin, self._arduino_led_pin)
        led_power = self._exec_stimulation[t_index][0]
        led_pulse_duration = self._exec_stimulation[t_index][1] / 1000  # convert to sec

        print(
            f"***Stimulation Event: {event.index}, "
            f"LED: {self._arduino_led_pin}, "
            f"LED Pulse Duration: {led_pulse_duration * 1000} ms, "
            f"LED Power: {led_power} %***"
        )

        # switch on the LED
        self._arduino_led_pin.write(led_power / 100)
        # wait for the duration of the pulse
        time.sleep(led_pulse_duration)
        # switch off the LED
        self._arduino_led_pin.write(0)

    def exec_sequenced_event(self, event: SequencedEvent) -> Iterable[PImagePayload]:
        """Execute a sequenced (triggered) event and return the image data.

        This method is not part of the PMDAEngine protocol (it is called by
        `exec_event`, which *is* part of the protocol), but it is made public
        in case a user wants to subclass this engine and override this method.
        """
        # TODO: add support for multiple camera devices
        n_events = len(event.events)

        # execute stimulation if the event if it is in the sequence metadata
        # if self._arduino_board is not None and self._arduino_led_pin is not None:
        if t_index := event.index.get("t", None):
            if t_index in self._exec_stimulation:
                self._exec_led_stimulation(t_index, event)

        # Start sequence
        # Note that the overload of startSequenceAcquisition that takes a camera
        # label does NOT automatically initialize a circular buffer.  So if this call
        # is changed to accept the camera in the future, that should be kept in mind.
        self._mmc.startSequenceAcquisition(
            n_events,
            0,  # intervalMS  # TODO: add support for this
            True,  # stopOnOverflow
        )

        self.post_sequence_started(event)

        count = 0
        iter_events = iter(event.events)
        # block until the sequence is done, popping images in the meantime
        while self._mmc.isSequenceRunning():
            if self._mmc.getRemainingImageCount():
                yield self._next_img_payload(next(iter_events))
                count += 1
            else:
                time.sleep(0.001)

        # this is not emitted otherwise
        self._mmc.events.sequenceAcquisitionStopped.emit(self._mmc.getCameraDevice())

        if self._mmc.isBufferOverflowed():  # pragma: no cover
            raise MemoryError("Buffer overflowed")

        while self._mmc.getRemainingImageCount():
            yield self._next_img_payload(next(iter_events))
            count += 1

        if count != n_events:
            logger.warning(
                "Unexpected number of images returned from sequence. "
                "Expected %s, got %s",
                n_events,
                count,
            )

    def _next_img_payload(self, event: MDAEvent) -> PImagePayload:
        """Grab next image from the circular buffer and return it as an ImagePayload."""
        img, meta = self._mmc.popNextImageAndMD()
        tags = self.get_frame_metadata(meta)

        # TEMPORARY SOLUTION
        if self._mmc.mda._wait_until_event(event):  # noqa SLF001
            self._mmc.mda.cancel()
            self._mmc.stopSequenceAcquisition()

        print(f"***Snap Event: {event.index}***\n")

        return ImagePayload(img, event, tags)

    # ===================== EXTRA =====================

    def _execute_autofocus(self, action: HardwareAutofocus) -> float:
        """Perform the hardware autofocus.

        Returns the change in ZPosition that occurred during the autofocus event.
        """
        # switch off autofocus device if it is on
        self._mmc.enableContinuousFocus(False)

        if action.autofocus_motor_offset is not None:
            # set the autofocus device offset
            # if name is given explicitly, use it, otherwise use setAutoFocusOffset
            # (see docs for setAutoFocusOffset for additional details)
            if name := getattr(action, "autofocus_device_name", None):
                self._mmc.setPosition(name, action.autofocus_motor_offset)
            else:
                self._mmc.setAutoFocusOffset(action.autofocus_motor_offset)
            self._mmc.waitForSystem()

        @retry(exceptions=RuntimeError, tries=action.max_retries, logger=logger.warning)
        def _perform_full_focus(previous_z: float) -> float:
            self._mmc.fullFocus()
            self._mmc.waitForSystem()
            return self._mmc.getZPosition() - previous_z

        return _perform_full_focus(self._mmc.getZPosition())

    def _set_event_position(self, event: MDAEvent) -> None:
        # skip if no XY stage device is found
        if not self._mmc.getXYStageDevice():
            logger.warning("No XY stage device found. Cannot set XY position.")
            return

        x = event.x_pos if event.x_pos is not None else self._mmc.getXPosition()
        y = event.y_pos if event.y_pos is not None else self._mmc.getYPosition()
        self._mmc.setXYPosition(x, y)

    def _set_event_z(self, event: MDAEvent) -> None:
        # skip if no Z stage device is found
        if not self._mmc.getFocusDevice():
            logger.warning("No Z stage device found. Cannot set Z position.")
            return

        p_idx = event.index.get("p", None)
        correction = self._z_correction.setdefault(p_idx, 0.0)
        self._mmc.setZPosition(cast("float", event.z_pos) + correction)


class ImagePayload(NamedTuple):
    image: NDArray
    event: MDAEvent
    metadata: dict
