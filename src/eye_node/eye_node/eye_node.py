import math
import random
import threading
import time

import numpy as np

try:
    import lgpio
    import spidev
    from PIL import Image, ImageDraw, ImageChops
    _HW_AVAILABLE = True
    _HW_ERROR = ''
except ImportError as e:
    _HW_AVAILABLE = False
    _HW_ERROR = str(e)

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# ── Hardware constants ────────────────────────────────────────────────────────

GPIO_CHIP    = 4              # Pi 5 RP1 — gpiochip4
DC_PIN       = 24             # BCM
RST_PIN      = 25             # BCM — shared reset for both displays

SPI_BUS      = 0
CS_LEFT      = 0              # CE0 → /dev/spidev0.0
CS_RIGHT     = 1              # CE1 → /dev/spidev0.1
SPI_SPEED_HZ = 10_000_000   # 40MHz fails over cat5 cable run — 10MHz stable
SPI_MODE     = 0

W  = 240
CX = W // 2
CY = W // 2

# ── State → color map (R, G, B) ───────────────────────────────────────────────

STATE_COLORS = {
    'IDLE':       (  0, 140, 255),   # cyborg blue
    'LISTENING':  (  0, 220, 255),   # cyan
    'SPEAKING':   (  0, 255,  80),   # green
    'NAVIGATING': (220, 220, 255),   # white-blue
    'EXPLORING':  (160,   0, 255),   # purple
    'DOCKING':    ( 30, 100, 255),   # blue
    'ERROR':      (255,  20,  20),   # red
}

_DEFAULT_STATE = 'IDLE'

# Dot-matrix style palette — the retro LED panel look is mostly one green, so
# state reads from pulse speed and brightness rather than hue.  Only the states
# that genuinely need to be distinguishable at a glance get their own colour.
MATRIX_COLORS = {
    'IDLE':       (  0, 255,  70),   # LED green
    'LISTENING':  (  0, 255,  70),
    'SPEAKING':   (  0, 255,  70),
    'NAVIGATING': (  0, 255,  70),
    'EXPLORING':  (  0, 255,  70),
    'DOCKING':    (  0, 255,  70),
    'ERROR':      (255,  40,  20),   # red — must not read as "fine"
}

# HAL 9000 glow pulse speed (rad/s) per state — state is conveyed by lens
# colour plus how fast/urgently the core breathes.
_PULSE_SPEED = {
    'IDLE':       1.0,   # slow, calm breathing
    'LISTENING':  1.8,   # attentive
    'SPEAKING':   5.0,   # rapid flutter while talking
    'NAVIGATING': 1.4,
    'EXPLORING':  2.6,
    'DOCKING':    0.9,
    'ERROR':      8.0,   # urgent alarm throb
}


# ── GC9A01 init sequence ──────────────────────────────────────────────────────

def _init_gc9a01(write_cmd, write_data):
    """Send the full GC9A01 init sequence using the provided write helpers."""
    write_cmd(0xEF)
    write_cmd(0xEB); write_data(0x14)
    write_cmd(0xFE)
    write_cmd(0xEF)
    write_cmd(0xEB); write_data(0x14)
    write_cmd(0x84); write_data(0x40)
    write_cmd(0x85); write_data(0xFF)
    write_cmd(0x86); write_data(0xFF)
    write_cmd(0x87); write_data(0xFF)
    write_cmd(0x88); write_data(0x0A)
    write_cmd(0x89); write_data(0x21)
    write_cmd(0x8A); write_data(0x00)
    write_cmd(0x8B); write_data(0x80)
    write_cmd(0x8C); write_data(0x01)
    write_cmd(0x8D); write_data(0x01)
    write_cmd(0x8E); write_data(0xFF)
    write_cmd(0x8F); write_data(0xFF)
    write_cmd(0xB6); write_data([0x00, 0x20])
    write_cmd(0x36); write_data(0x08)
    write_cmd(0x3A); write_data(0x05)
    write_cmd(0x90); write_data([0x08, 0x08, 0x08, 0x08])
    write_cmd(0xBD); write_data(0x06)
    write_cmd(0xBC); write_data(0x00)
    write_cmd(0xFF); write_data([0x60, 0x01, 0x04])
    write_cmd(0xC3); write_data(0x13)
    write_cmd(0xC4); write_data(0x13)
    write_cmd(0xC9); write_data(0x22)
    write_cmd(0xBE); write_data(0x11)
    write_cmd(0xE1); write_data([0x10, 0x0E])
    write_cmd(0xDF); write_data([0x21, 0x0c, 0x02])
    write_cmd(0xF0); write_data([0x45, 0x09, 0x08, 0x08, 0x26, 0x2A])
    write_cmd(0xF1); write_data([0x43, 0x70, 0x72, 0x36, 0x37, 0x6F])
    write_cmd(0xF2); write_data([0x45, 0x09, 0x08, 0x08, 0x26, 0x2A])
    write_cmd(0xF3); write_data([0x43, 0x70, 0x72, 0x36, 0x37, 0x6F])
    write_cmd(0xED); write_data([0x1B, 0x0B])
    write_cmd(0xAE); write_data(0x77)
    write_cmd(0xCD); write_data(0x63)
    write_cmd(0x70); write_data([0x07, 0x07, 0x04, 0x0E, 0x0F, 0x09, 0x07, 0x08, 0x03])
    write_cmd(0xE8); write_data(0x34)
    write_cmd(0x62); write_data([0x18, 0x0D, 0x71, 0xED, 0x70, 0x70,
                                  0x18, 0x0F, 0x71, 0xEF, 0x70, 0x70])
    write_cmd(0x63); write_data([0x18, 0x11, 0x71, 0xF1, 0x70, 0x70,
                                  0x18, 0x13, 0x71, 0xF3, 0x70, 0x70])
    write_cmd(0x64); write_data([0x28, 0x29, 0xF1, 0x01, 0xF1, 0x00, 0x07])
    write_cmd(0x66); write_data([0x3C, 0x00, 0xCD, 0x67, 0x45, 0x45, 0x10, 0x00, 0x00, 0x00])
    write_cmd(0x67); write_data([0x00, 0x3C, 0x00, 0x00, 0x00, 0x01, 0x54, 0x10, 0x32, 0x98])
    write_cmd(0x74); write_data([0x10, 0x85, 0x80, 0x00, 0x00, 0x4E, 0x00])
    write_cmd(0x98); write_data([0x3e, 0x07])
    write_cmd(0x35)
    write_cmd(0x21)
    write_cmd(0x11); time.sleep(0.12)
    write_cmd(0x29); time.sleep(0.02)


# ── HAL 9000 lens renderer ────────────────────────────────────────────────────

def _lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


# Cyborg-eye geometry (radii in px, display is 240 wide → max radius 120)
_R_IRIS  = 116   # outer edge of the glowing iris
_R_PUPIL = 42    # mechanical pupil radius


def _draw_cyborg_eye(now, pulse, color):
    """Render a glowing mechanical 'cyborg' iris: a radial-gradient iris with a
    mid-radius glow ring, slowly rotating fibre striations, concentric tech
    rings, and a dark pupil with a hot sensor core.  Both eyes render
    identically — no left/right mirroring, no saccade, no blink."""
    img  = Image.new('RGB', (W, W), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    cr, cg, cb = color
    dim     = (cr // 6, cg // 6, cb // 6)
    mid     = (cr // 2, cg // 2, cb // 2)
    bright  = (min(255, cr + 70), min(255, cg + 70), min(255, cb + 70))
    breathe = 0.60 + 0.40 * pulse

    # ── Iris body — gradient peaking in a mid-radius glow ring ───────────────
    for r in range(_R_IRIS, 0, -1):
        t   = r / _R_IRIS                          # 1 rim .. 0 centre
        lvl = (0.12 + 0.55 * max(0.0, 1 - abs(t - 0.58) / 0.58) ** 1.3) * breathe
        draw.ellipse([CX - r, CY - r, CX + r, CY + r],
                     fill=(int(cr * lvl), int(cg * lvl), int(cb * lvl)))

    # ── Radial fibre striations — slow rotation for a living mechanism ───────
    rot  = now * 0.15
    fibs = 90
    for i in range(fibs):
        a  = rot + i * (2 * math.pi / fibs)
        x1 = CX + _R_PUPIL * math.cos(a); y1 = CY + _R_PUPIL * math.sin(a)
        x2 = CX + _R_IRIS  * math.cos(a); y2 = CY + _R_IRIS  * math.sin(a)
        c  = bright if i % 2 == 0 else dim
        draw.line([(x1, y1), (x2, y2)], fill=c, width=1)

    # ── Concentric tech rings ────────────────────────────────────────────────
    for rr, wd in ((_R_IRIS - 1, 2), (94, 1), (68, 1)):
        draw.ellipse([CX - rr, CY - rr, CX + rr, CY + rr], outline=mid, width=wd)

    # ── Pupil — dark aperture ────────────────────────────────────────────────
    draw.ellipse([CX - _R_PUPIL, CY - _R_PUPIL, CX + _R_PUPIL, CY + _R_PUPIL],
                 fill=(0, 0, 0))

    # ── Glowing collarette + hot sensor core, composited additively ──────────
    glow = Image.new('RGB', (W, W), (0, 0, 0))
    gd   = ImageDraw.Draw(glow)
    # glow ring whose radius rings in and out around the pupil with the pulse
    ring_c = _R_PUPIL + 4 + 14 * pulse     # expands outward as the pulse swells
    ring_w = 9                              # soft falloff thickness
    for r in range(int(ring_c + ring_w), int(ring_c - ring_w), -1):
        if r < 1:
            continue
        lvl = 1 - abs(r - ring_c) / ring_w
        if lvl <= 0:
            continue
        gd.ellipse([CX - r, CY - r, CX + r, CY + r],
                   outline=(int(bright[0] * lvl), int(bright[1] * lvl),
                            int(bright[2] * lvl)), width=2)
    # hot sensor core at the very centre
    core = 13
    for r in range(core + 14, 0, -1):
        if r > core:
            t   = (r - core) / 14
            lvl = (1 - t) ** 2
            col = (int(bright[0] * lvl), int(bright[1] * lvl), int(bright[2] * lvl))
        else:
            w   = (1 - r / core) ** 1.5
            col = _lerp_color(bright, (255, 255, 255), w)
        gd.ellipse([CX - r, CY - r, CX + r, CY + r], fill=col)
    img = ImageChops.add(img, glow)

    # ── Circular mask — clips to round display ───────────────────────────────
    mask  = Image.new('L', (W, W), 0)
    ImageDraw.Draw(mask).ellipse([2, 2, W - 2, W - 2], fill=255)
    black = Image.new('RGB', (W, W), (0, 0, 0))
    return Image.composite(img, black, mask)


# ── Dot-matrix LED renderer ───────────────────────────────────────────────────
#
# Emulates a discrete LED panel: a square grid of round dots on black, where an
# eye-shaped mask decides which dots light and how brightly.  Unlit cells are
# left near-black so the panel structure only shows where the eye is, matching
# the reference look.

_GRID    = 12                    # dots per axis across the 240px display
_PITCH   = W / _GRID             # 20 px between dot centres
_DOT_R   = 6.4                   # lit-core radius in px (gap between dots reads)
_GLOW_R  = 11.0                  # bloom radius — LED bleed into the black
_EYE_R   = 0.28 * W              # radius of the eye disc — leaves a black
                                 # margin inside the round display bezel


def _make_dot_kernel():
    """Pre-render one LED dot: a hard-ish core with a soft exponential bloom.

    Returns a float32 array of intensity 0..1, and the offset from its centre to
    its top-left corner, so it can be additively blitted at any dot position.
    """
    half = int(math.ceil(_GLOW_R))
    size = 2 * half + 1
    ys, xs = np.mgrid[0:size, 0:size]
    d = np.hypot(xs - half, ys - half).astype(np.float32)

    # Core: flat top, one-pixel soft edge so the dot doesn't alias into a square.
    core = np.clip((_DOT_R + 0.5 - d) / 1.0, 0.0, 1.0)
    # Bloom: falls off exponentially past the core, never fully reaching the core
    # brightness so the dot keeps a defined edge.
    bloom = 0.34 * np.exp(-np.clip(d - _DOT_R, 0.0, None) / 2.6)

    return np.maximum(core, bloom).astype(np.float32), half


_DOT_KERNEL, _DOT_HALF = _make_dot_kernel()

# Grid cell centres, precomputed once.
_CELL_XY = [
    (( gx + 0.5) * _PITCH, (gy + 0.5) * _PITCH)
    for gy in range(_GRID) for gx in range(_GRID)
]


def _blink_openness(now):
    """Eyelid openness 0..1 — mostly 1.0, with a fast blink every few seconds.

    The blink phase is derived from wall-clock time rather than stored state so
    both eyes blink in lockstep without sharing anything.
    """
    period = 4.7                            # seconds between blinks
    phase  = now % period
    if phase > 0.22:                        # open the vast majority of the time
        return 1.0
    # 0.22s blink: closes over the first half, reopens over the second.
    t = phase / 0.22
    return abs(t - 0.5) * 2.0


def _draw_matrix_eye(now, pulse, color, gaze=(0.0, 0.0)):
    """Render the eye as a grid of glowing LED dots.

    ``gaze`` is a (dx, dy) offset in units of the eye radius, so (-0.3, 0) looks
    left.  Both eyes render identically — the mask is centred on the display.
    """
    frame = np.zeros((W, W), dtype=np.float32)

    open_amt = _blink_openness(now)
    if open_amt < 0.02:
        open_amt = 0.02                     # avoid a divide-by-zero on full close

    ecx = CX + gaze[0] * _EYE_R
    ecy = CY + gaze[1] * _EYE_R
    ry  = _EYE_R * open_amt                 # squash vertically to blink

    breathe = 0.72 + 0.28 * pulse

    for cx_, cy_ in _CELL_XY:
        dx = (cx_ - ecx) / _EYE_R
        dy = (cy_ - ecy) / ry
        d  = math.hypot(dx, dy)             # 0 centre .. 1 rim

        if d > 1.18:
            continue                        # well outside the eye — dot stays dark

        # Soft rim so the disc edge dithers across the outer ring of dots rather
        # than cutting off square.
        edge = min(1.0, max(0.0, (1.18 - d) / 0.30))
        # Very slight centre-bright falloff keeps it from looking like flat fill.
        lvl  = edge * (1.0 - 0.18 * d ** 2) * breathe
        if lvl <= 0.01:
            continue

        px, py = int(round(cx_)), int(round(cy_))
        x0, y0 = px - _DOT_HALF, py - _DOT_HALF
        x1, y1 = x0 + _DOT_KERNEL.shape[1], y0 + _DOT_KERNEL.shape[0]

        # Clip the kernel against the frame edges.
        kx0, ky0 = max(0, -x0), max(0, -y0)
        kx1, ky1 = _DOT_KERNEL.shape[1] - max(0, x1 - W), _DOT_KERNEL.shape[0] - max(0, y1 - W)
        if kx1 <= kx0 or ky1 <= ky0:
            continue

        dst = frame[max(0, y0):min(W, y1), max(0, x0):min(W, x1)]
        np.maximum(dst, _DOT_KERNEL[ky0:ky1, kx0:kx1] * lvl, out=dst)

    # Colourise: intensity drives the state colour, and the hottest part of each
    # dot washes toward white the way a saturated LED does to a camera.
    cr, cg, cb = color
    hot = np.clip((frame - 0.75) / 0.25, 0.0, 1.0) * 0.55
    rgb = np.dstack([
        frame * cr + hot * (255 - cr),
        frame * cg + hot * (255 - cg),
        frame * cb + hot * (255 - cb),
    ])
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    img = Image.fromarray(rgb, 'RGB')

    # Circular mask — clips to the round display.
    mask = Image.new('L', (W, W), 0)
    ImageDraw.Draw(mask).ellipse([2, 2, W - 2, W - 2], fill=255)
    return Image.composite(img, Image.new('RGB', (W, W), (0, 0, 0)), mask)


class EyeNode(Node):

    def __init__(self):
        super().__init__('eye_node')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('target_fps', 20.0)
        self._target_fps     = self.get_parameter('target_fps').value
        self._frame_interval = 1.0 / self._target_fps

        # 'matrix' = retro LED dot grid, 'lens' = the original cyborg iris.
        self.declare_parameter('eye_style', 'matrix')
        self._eye_style = self.get_parameter('eye_style').value
        self._palette   = (MATRIX_COLORS if self._eye_style == 'matrix'
                           else STATE_COLORS)

        # ── State ────────────────────────────────────────────────────────────
        self._current_state  = _DEFAULT_STATE
        self._state_lock     = threading.Lock()
        self._stop_event     = threading.Event()

        # Animation variables
        self._blended_color = list(self._palette[_DEFAULT_STATE])
        self._pulse_speed   = _PULSE_SPEED[_DEFAULT_STATE]

        # ── Hardware init ─────────────────────────────────────────────────────
        if _HW_AVAILABLE:
            # GPIO via lgpio (Pi 5 gpiochip4)
            self._gpio = lgpio.gpiochip_open(GPIO_CHIP)
            for _pin in (DC_PIN, RST_PIN):
                try:
                    lgpio.gpio_free(self._gpio, _pin)
                except lgpio.error:
                    pass
            lgpio.gpio_claim_output(self._gpio, DC_PIN,  0)
            lgpio.gpio_claim_output(self._gpio, RST_PIN, 1)

            # Two SpiDev instances — both open simultaneously (never close/reopen)
            self._spi0 = spidev.SpiDev()
            self._spi1 = spidev.SpiDev()
            self._spi0.open(SPI_BUS, CS_LEFT)
            self._spi0.max_speed_hz = SPI_SPEED_HZ
            self._spi0.mode = SPI_MODE
            self._spi1.open(SPI_BUS, CS_RIGHT)
            self._spi1.max_speed_hz = SPI_SPEED_HZ
            self._spi1.mode = SPI_MODE

            # Reset both displays (shared RST line — one pulse resets both)
            lgpio.gpio_write(self._gpio, RST_PIN, 1); time.sleep(0.05)
            lgpio.gpio_write(self._gpio, RST_PIN, 0); time.sleep(0.05)
            lgpio.gpio_write(self._gpio, RST_PIN, 1); time.sleep(0.05)

            # Init left eye (CE0 / spi0)
            self.get_logger().info('Initialising left eye (CE0)...')
            _init_gc9a01(
                lambda cmd:  self._spi_write_cmd(self._spi0, cmd),
                lambda data: self._spi_write_data(self._spi0, data),
            )
            # Init right eye (CE1 / spi1)
            self.get_logger().info('Initialising right eye (CE1)...')
            _init_gc9a01(
                lambda cmd:  self._spi_write_cmd(self._spi1, cmd),
                lambda data: self._spi_write_data(self._spi1, data),
            )
        else:
            self._gpio = None
            self._spi0 = None
            self._spi1 = None
            self.get_logger().error(f'Hardware unavailable: {_HW_ERROR}')

        # ── Subscriber ───────────────────────────────────────────────────────
        self.create_subscription(String, '/robot_state', self._robot_state_cb, 10)

        # ── Animation thread ─────────────────────────────────────────────────
        self._anim_thread = threading.Thread(
            target=self._animation_loop, daemon=True, name='eye_anim'
        )
        self._anim_thread.start()

        self.get_logger().info(
            f'eye_node ready — {self._target_fps:.0f} fps, '
            f'state={_DEFAULT_STATE}, hw={_HW_AVAILABLE}'
        )

    # ── Subscriber ────────────────────────────────────────────────────────────

    def _robot_state_cb(self, msg: String):
        state = msg.data.strip().upper()
        if state in self._palette:
            with self._state_lock:
                self._current_state = state
            self.get_logger().info(f'robot_state → {state}')
        else:
            self.get_logger().warn(
                f'Unknown robot_state: {msg.data!r}',
                throttle_duration_sec=10.0,
            )

    # ── SPI helpers ──────────────────────────────────────────────────────────

    def _spi_write_cmd(self, spi_dev, cmd: int):
        lgpio.gpio_write(self._gpio, DC_PIN, 0)
        spi_dev.writebytes([cmd])

    def _spi_write_data(self, spi_dev, data):
        lgpio.gpio_write(self._gpio, DC_PIN, 1)
        if isinstance(data, int):
            spi_dev.writebytes([data])
        else:
            spi_dev.writebytes(list(data))

    def _send_frame(self, spi_dev, img: 'Image.Image'):
        """Convert PIL image to RGB565 and push to one display."""
        pixels = np.array(img.convert('RGB'), dtype=np.uint8)
        r = pixels[:, :, 0].astype(np.uint16)
        g = pixels[:, :, 1].astype(np.uint16)
        b = pixels[:, :, 2].astype(np.uint16)
        color = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
        data  = np.dstack([(color >> 8) & 0xFF, color & 0xFF]).flatten().astype(np.uint8)

        lgpio.gpio_write(self._gpio, DC_PIN, 0)
        spi_dev.writebytes([0x2A])
        lgpio.gpio_write(self._gpio, DC_PIN, 1)
        spi_dev.writebytes([0x00, 0x00, 0x00, 0xEF])
        lgpio.gpio_write(self._gpio, DC_PIN, 0)
        spi_dev.writebytes([0x2B])
        lgpio.gpio_write(self._gpio, DC_PIN, 1)
        spi_dev.writebytes([0x00, 0x00, 0x00, 0xEF])
        lgpio.gpio_write(self._gpio, DC_PIN, 0)
        spi_dev.writebytes([0x2C])
        lgpio.gpio_write(self._gpio, DC_PIN, 1)
        spi_dev.writebytes2(data.tolist())

    # ── Animation loop ────────────────────────────────────────────────────────

    def _animation_loop(self):
        while not self._stop_event.is_set():
            t0  = time.monotonic()
            now = time.time()

            with self._state_lock:
                state = self._current_state

            target_color = self._palette[state]

            # Smooth color transition between states
            self._blended_color = list(_lerp_color(
                tuple(int(c) for c in self._blended_color),
                target_color, 0.05
            ))
            color = tuple(int(c) for c in self._blended_color)

            # Smoothly ease the pulse speed toward the state's target so the
            # rhythm shifts gracefully rather than jumping on a state change.
            self._pulse_speed += (_PULSE_SPEED[state] - self._pulse_speed) * 0.05

            # Breathing glow + faint flicker for an analogue, "alive" lens
            pulse = math.sin(now * self._pulse_speed) * 0.5 + 0.5
            pulse = max(0.0, min(1.0, pulse + random.uniform(-0.04, 0.04)))

            try:
                if self._eye_style == 'matrix':
                    eye_img = _draw_matrix_eye(now, pulse, color)
                else:
                    eye_img = _draw_cyborg_eye(now, pulse, color)

                if _HW_AVAILABLE and self._spi0 is not None:
                    self._send_frame(self._spi0, eye_img)
                    self._send_frame(self._spi1, eye_img)

            except Exception as e:
                self.get_logger().warn(
                    f'Eye render/send error: {e}',
                    throttle_duration_sec=5.0,
                )

            spent = time.monotonic() - t0
            gap   = self._frame_interval - spent
            if gap > 0.001:
                time.sleep(gap)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def destroy_node(self):
        self._stop_event.set()
        if hasattr(self, '_anim_thread'):
            self._anim_thread.join(timeout=2.0)
        if _HW_AVAILABLE:
            if self._spi0 is not None:
                try:
                    self._spi0.close()
                except Exception:
                    pass
            if self._spi1 is not None:
                try:
                    self._spi1.close()
                except Exception:
                    pass
            if self._gpio is not None:
                try:
                    lgpio.gpiochip_close(self._gpio)
                except Exception:
                    pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = EyeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
