import math
import random
import threading
import time

import numpy as np

try:
    import lgpio
    import spidev
    from PIL import Image, ImageDraw
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
    'IDLE':       (255, 140,   0),   # amber
    'LISTENING':  (  0, 220, 255),   # cyan
    'SPEAKING':   (  0, 255,  80),   # green
    'NAVIGATING': (220, 220, 255),   # white-blue
    'EXPLORING':  (160,   0, 255),   # purple
    'DOCKING':    ( 30, 100, 255),   # blue
    'ERROR':      (255,  20,  20),   # red
}

_DEFAULT_STATE = 'IDLE'

# Look-style parameters per state: (saccade_range, saccade_interval, lerp_alpha)
_LOOK_PARAMS = {
    'IDLE':       (0.5,  (2.0, 4.0), 0.04),
    'LISTENING':  (0.2,  (3.0, 5.0), 0.03),
    'SPEAKING':   (0.6,  (0.4, 0.9), 0.06),
    'NAVIGATING': (0.1,  (3.0, 6.0), 0.03),
    'EXPLORING':  (0.9,  (0.1, 0.4), 0.10),
    'DOCKING':    (0.1,  (2.0, 4.0), 0.03),
    'ERROR':      (0.5,  (0.1, 0.3), 0.12),
}

_BLINK_INTERVALS = {
    'IDLE':       (3.0, 5.0),
    'LISTENING':  (5.0, 8.0),
    'SPEAKING':   (1.5, 3.5),
    'NAVIGATING': (4.0, 7.0),
    'EXPLORING':  (2.5, 4.5),
    'DOCKING':    (5.0, 8.0),
    'ERROR':      (1.0, 2.5),
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


# ── HUD eye renderer (from confirmed working code) ────────────────────────────

def _lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def _draw_eye(side, now, pulse, look_x, look_y, blink, color):
    img  = Image.new('RGB', (W, W), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    cr, cg, cb = color
    dim    = (cr // 6,  cg // 6,  cb // 6)
    mid    = (cr // 3,  cg // 3,  cb // 3)
    full   = (cr,       cg,       cb)
    bright = (min(255, cr + 60), min(255, cg + 60), min(255, cb + 60))

    # Background subtle glow
    for r in range(115, 0, -5):
        t = 1 - r / 115
        v = int(18 * t)
        draw.ellipse([CX - r, CY - r, CX + r, CY + r],
                     fill=(cr * v // 255, cg * v // 255, cb * v // 255))

    # Outer ring — slow rotate, mirrored per side
    outer_r   = 112
    rot_speed = 0.3 if side == 'left' else -0.3
    rot_angle = now * rot_speed
    seg_count = 32
    seg_gap   = 4
    seg_deg   = (360 / seg_count) - seg_gap
    for i in range(seg_count):
        start = rot_angle * (180 / math.pi) + i * (360 / seg_count)
        c = bright if i % 4 == 0 else mid
        draw.arc([CX - outer_r, CY - outer_r, CX + outer_r, CY + outer_r],
                 start=start, end=start + seg_deg, fill=c, width=3)

    # Second ring with tick marks
    ring2_r    = 95
    tick_count = 48
    for i in range(tick_count):
        angle    = math.radians(i * (360 / tick_count))
        tick_len = 8 if i % 6 == 0 else 4
        x1 = CX + int((ring2_r - tick_len) * math.cos(angle))
        y1 = CY + int((ring2_r - tick_len) * math.sin(angle))
        x2 = CX + int(ring2_r * math.cos(angle))
        y2 = CY + int(ring2_r * math.sin(angle))
        c  = full if i % 6 == 0 else dim
        draw.line([(x1, y1), (x2, y2)], fill=c, width=1)

    # Segmented arc ring — counter-rotate, mirrored
    ring3_r    = 80
    rot2       = now * (-0.5 if side == 'left' else 0.5)
    seg2_count = 12
    seg2_gap   = 8
    seg2_deg   = (360 / seg2_count) - seg2_gap
    for i in range(seg2_count):
        start = rot2 * (180 / math.pi) + i * (360 / seg2_count)
        alpha = 1.0 if i % 3 == 0 else 0.4
        c = (int(cr * alpha), int(cg * alpha), int(cb * alpha))
        draw.arc([CX - ring3_r, CY - ring3_r, CX + ring3_r, CY + ring3_r],
                 start=start, end=start + seg2_deg, fill=c, width=4)

    # Inner solid ring
    ring4_r = 62
    draw.ellipse([CX - ring4_r, CY - ring4_r, CX + ring4_r, CY + ring4_r],
                 outline=mid, width=1)

    # Pulse ring (breathes)
    pulse_r = int(68 + 5 * pulse)
    pulse_c = (int(cr * (0.3 + 0.5 * pulse)),
               int(cg * (0.3 + 0.5 * pulse)),
               int(cb * (0.3 + 0.5 * pulse)))
    draw.ellipse([CX - pulse_r, CY - pulse_r, CX + pulse_r, CY + pulse_r],
                 outline=pulse_c, width=2)

    # Targeting brackets — mirrored on right eye
    bracket_r = 52
    blen      = 14
    bthick    = 2
    corners   = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
    if side == 'right':
        corners = [(-x, y) for x, y in corners]
    for dx, dy in corners:
        bx = CX + dx * bracket_r
        by = CY + dy * bracket_r
        draw.line([(bx, by), (bx - dx * blen, by)], fill=full, width=bthick)
        draw.line([(bx, by), (bx, by - dy * blen)], fill=full, width=bthick)

    # Crosshair lines (subtle)
    cross_r = 55
    draw.line([(CX - cross_r, CY), (CX - 10, CY)], fill=dim, width=1)
    draw.line([(CX + 10,      CY), (CX + cross_r, CY)], fill=dim, width=1)
    draw.line([(CX, CY - cross_r), (CX, CY - 10)], fill=dim, width=1)
    draw.line([(CX, CY + 10),      (CX, CY + cross_r)], fill=dim, width=1)

    # Pupil with look offset
    ox      = int(look_x * 20)
    oy      = int(look_y * 20)
    px, py  = CX + ox, CY + oy
    pupil_r = 22
    for pr in range(pupil_r + 10, pupil_r - 1, -2):
        alpha = 0.15 * (1 - (pr - pupil_r) / 10)
        draw.ellipse([px - pr, py - pr, px + pr, py + pr],
                     fill=(int(cr * alpha), int(cg * alpha), int(cb * alpha)))
    draw.ellipse([px - pupil_r, py - pupil_r, px + pupil_r, py + pupil_r],
                 fill=(0, 0, 0))
    draw.ellipse([px - pupil_r, py - pupil_r, px + pupil_r, py + pupil_r],
                 outline=full, width=2)
    draw.ellipse([px - 7, py - 9, px - 1, py - 3], fill=(255, 255, 255))
    draw.ellipse([px + 3, py + 3, px + 7, py + 7],
                 fill=(min(255, cr + 80), min(255, cg + 80), min(255, cb + 80)))

    # Blink eyelids
    if blink > 0.01:
        lid = int((W // 2 + 20) * blink)
        draw.rectangle([0, 0, W, CY - 60 + lid], fill=(0, 0, 0))
        draw.rectangle([0, CY + 60 - lid, W, W],  fill=(0, 0, 0))

    # Circular mask — clips to round display
    mask  = Image.new('L', (W, W), 0)
    ImageDraw.Draw(mask).ellipse([2, 2, W - 2, W - 2], fill=255)
    black = Image.new('RGB', (W, W), (0, 0, 0))
    return Image.composite(img, black, mask)


class EyeNode(Node):

    def __init__(self):
        super().__init__('eye_node')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('target_fps', 20.0)
        self._target_fps     = self.get_parameter('target_fps').value
        self._frame_interval = 1.0 / self._target_fps

        # ── State ────────────────────────────────────────────────────────────
        self._current_state  = _DEFAULT_STATE
        self._state_lock     = threading.Lock()
        self._stop_event     = threading.Event()

        # Animation variables
        self._look_x         = 0.0
        self._look_y         = 0.0
        self._target_x       = 0.0
        self._target_y       = 0.0
        self._next_saccade_t = time.monotonic()
        self._blended_color  = list(STATE_COLORS[_DEFAULT_STATE])

        self._blinking    = False
        self._blink_phase = 0.0
        self._blink       = 0.0
        self._next_blink_t = time.monotonic() + random.uniform(2.0, 4.0)

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
        if state in STATE_COLORS:
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

            target_color = STATE_COLORS[state]
            look_rng, saccade_ivl, lerp_alpha = _LOOK_PARAMS[state]
            blink_lo, blink_hi = _BLINK_INTERVALS[state]

            # Smooth color transition
            self._blended_color = list(_lerp_color(
                tuple(int(c) for c in self._blended_color),
                target_color, 0.05
            ))
            color = tuple(int(c) for c in self._blended_color)

            pulse = math.sin(now * 2.0) * 0.5 + 0.5

            # Saccade
            if t0 >= self._next_saccade_t:
                lo, hi = saccade_ivl
                self._next_saccade_t = t0 + random.uniform(lo, hi)
                if state == 'DOCKING':
                    self._target_x = random.uniform(-0.2, 0.2)
                    self._target_y = random.uniform(0.3, 0.7)
                elif state == 'NAVIGATING':
                    self._target_x = random.gauss(0.0, 0.15)
                    self._target_y = random.gauss(0.1, 0.1)
                else:
                    self._target_x = random.uniform(-look_rng, look_rng)
                    self._target_y = random.uniform(-look_rng * 0.6, look_rng * 0.6)
                self._target_x = max(-1.0, min(1.0, self._target_x))
                self._target_y = max(-1.0, min(1.0, self._target_y))

            self._look_x += (self._target_x - self._look_x) * lerp_alpha
            self._look_y += (self._target_y - self._look_y) * lerp_alpha

            # Blink
            if not self._blinking and t0 >= self._next_blink_t:
                self._blinking    = True
                self._blink_phase = 0.0
            if self._blinking:
                self._blink_phase += 0.12
                self._blink = (
                    math.sin(self._blink_phase * math.pi)
                    if self._blink_phase <= 1.0 else 0.0
                )
                if self._blink_phase > 1.0:
                    self._blinking     = False
                    self._next_blink_t = t0 + random.uniform(blink_lo, blink_hi)
            else:
                self._blink = 0.0

            try:
                left_img  = _draw_eye('left',  now, pulse,
                                      self._look_x, self._look_y,
                                      self._blink, color)
                right_img = _draw_eye('right', now, pulse,
                                      self._look_x, self._look_y,
                                      self._blink, color)

                if _HW_AVAILABLE and self._spi0 is not None:
                    self._send_frame(self._spi0, left_img)
                    self._send_frame(self._spi1, right_img)

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
