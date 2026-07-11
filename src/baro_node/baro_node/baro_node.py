import struct
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import FluidPressure, Temperature
from std_msgs.msg import Float32, Int32

import smbus2


# ---------------------------------------------------------------------------
# BMP280 register map
# ---------------------------------------------------------------------------
REG_CHIP_ID   = 0xD0   # returns 0x58 for a genuine BMP280
REG_RESET     = 0xE0
REG_CTRL_MEAS = 0xF4
REG_CONFIG    = 0xF5
REG_CALIB     = 0x88   # 24 bytes of factory trim (dig_T1..dig_P9)
REG_DATA      = 0xF7   # press[3] + temp[3], burst-readable

CHIP_ID_BMP280 = 0x58

# ctrl_meas: osrs_t x2 (010) | osrs_p x16 (101) | normal mode (11) = 0x57
CTRL_MEAS = 0x57
# config: t_sb 0.5ms (000) | IIR filter x16 (100) | spi3w off (0) = 0x10
CONFIG = 0x10


class BMP280:
    """Minimal, dependency-free BMP280 driver over smbus2 (float compensation)."""

    def __init__(self, bus, address):
        self._bus = bus
        self._addr = address
        chip_id = self._bus.read_byte_data(self._addr, REG_CHIP_ID)
        if chip_id != CHIP_ID_BMP280:
            raise RuntimeError(
                f'BMP280 not found at 0x{self._addr:02x} '
                f'(chip id 0x{chip_id:02x}, expected 0x{CHIP_ID_BMP280:02x})'
            )
        self._read_calibration()
        self._bus.write_byte_data(self._addr, REG_CONFIG, CONFIG)
        self._bus.write_byte_data(self._addr, REG_CTRL_MEAS, CTRL_MEAS)
        self._t_fine = 0.0
        # In normal mode the data registers hold reset values until the first
        # conversion completes (~40 ms at these oversampling settings). Wait,
        # then discard one sample so the caller never sees the garbage reading.
        time.sleep(0.1)
        self.read()

    def _read_calibration(self):
        cal = self._bus.read_i2c_block_data(self._addr, REG_CALIB, 24)
        # dig_T1, P1 unsigned; the rest signed. Little-endian on the wire.
        (self.dig_T1, self.dig_T2, self.dig_T3,
         self.dig_P1, self.dig_P2, self.dig_P3, self.dig_P4, self.dig_P5,
         self.dig_P6, self.dig_P7, self.dig_P8, self.dig_P9) = struct.unpack(
            '<HhhHhhhhhhhh', bytes(cal))

    def read(self):
        """Return (temperature_C, pressure_Pa)."""
        d = self._bus.read_i2c_block_data(self._addr, REG_DATA, 6)
        adc_p = (d[0] << 12) | (d[1] << 4) | (d[2] >> 4)
        adc_t = (d[3] << 12) | (d[4] << 4) | (d[5] >> 4)

        # Temperature compensation (datasheet float formula) -> sets t_fine
        var1 = (adc_t / 16384.0 - self.dig_T1 / 1024.0) * self.dig_T2
        var2 = ((adc_t / 131072.0 - self.dig_T1 / 8192.0) ** 2) * self.dig_T3
        self._t_fine = var1 + var2
        temperature = self._t_fine / 5120.0

        # Pressure compensation
        var1 = self._t_fine / 2.0 - 64000.0
        var2 = var1 * var1 * self.dig_P6 / 32768.0
        var2 = var2 + var1 * self.dig_P5 * 2.0
        var2 = var2 / 4.0 + self.dig_P4 * 65536.0
        var1 = (self.dig_P3 * var1 * var1 / 524288.0 + self.dig_P2 * var1) / 524288.0
        var1 = (1.0 + var1 / 32768.0) * self.dig_P1
        if var1 == 0.0:
            return temperature, float('nan')
        p = 1048576.0 - adc_p
        p = (p - var2 / 4096.0) * 6250.0 / var1
        var1 = self.dig_P9 * p * p / 2147483648.0
        var2 = p * self.dig_P8 / 32768.0
        pressure = p + (var1 + var2 + self.dig_P7) / 16.0
        return temperature, pressure


def pressure_to_altitude(pressure_pa, sea_level_pa):
    """Barometric formula -> altitude in metres above the sea-level reference."""
    return 44330.0 * (1.0 - (pressure_pa / sea_level_pa) ** (1.0 / 5.255))


class BaroNode(Node):
    def __init__(self):
        super().__init__('baro_node')

        self.declare_parameter('i2c_bus', 1)
        self.declare_parameter('i2c_address', 0x77)
        self.declare_parameter('publish_rate', 10.0)
        # Sea-level pressure only affects the *absolute* altitude reading; floor
        # detection below is relative and does not depend on it.
        self.declare_parameter('sea_level_pressure', 101325.0)
        # EMA smoothing on raw pressure. Baro is noisy (~few Pa); a floor is tens
        # of Pa, so we filter hard. 0..1, higher = smoother/slower.
        self.declare_parameter('filter_alpha', 0.9)
        # Metres of vertical travel per building floor.
        self.declare_parameter('floor_height_m', 3.0)
        # Starting floor assumed at boot (overwritten by /baro/set_floor).
        self.declare_parameter('start_floor', 0)

        self.bus_num = self.get_parameter('i2c_bus').value
        self.address = self.get_parameter('i2c_address').value
        rate = self.get_parameter('publish_rate').value
        self.sea_level = self.get_parameter('sea_level_pressure').value
        self.alpha = self.get_parameter('filter_alpha').value
        self.floor_height = self.get_parameter('floor_height_m').value

        self.pub_pressure = self.create_publisher(FluidPressure, '/baro/pressure', 10)
        self.pub_temperature = self.create_publisher(Temperature, '/baro/temperature', 10)
        self.pub_altitude = self.create_publisher(Float32, '/baro/altitude', 10)
        self.pub_floor = self.create_publisher(Int32, '/baro/floor', 10)

        # Docking/AprilTag layer calls this with the known floor to re-anchor
        # the barometric reference and cancel weather drift.
        self.sub_set_floor = self.create_subscription(
            Int32, '/baro/set_floor', self.on_set_floor, 10)

        self.bus = smbus2.SMBus(self.bus_num)
        self.sensor = BMP280(self.bus, self.address)
        self.get_logger().info(
            f'BMP280 online on i2c-{self.bus_num} @ 0x{self.address:02x}')

        self.filtered_pressure = None
        # Reference: at ref_pressure we were on ref_floor.
        self.ref_floor = self.get_parameter('start_floor').value
        self.ref_pressure = None  # set on first reading
        self.current_floor = self.ref_floor

        self.timer = self.create_timer(1.0 / rate, self.tick)

    def on_set_floor(self, msg):
        """Re-anchor: 'you are on floor N right now' (e.g. from a dock tag)."""
        if self.filtered_pressure is None:
            self.get_logger().warn('set_floor ignored: no pressure reading yet')
            return
        self.ref_floor = int(msg.data)
        self.ref_pressure = self.filtered_pressure
        self.current_floor = self.ref_floor
        self.get_logger().info(
            f'Reference set: floor {self.ref_floor} @ {self.ref_pressure:.1f} Pa')

    def tick(self):
        try:
            temperature, pressure = self.sensor.read()
        except OSError as e:
            self.get_logger().warn(f'BMP280 read failed: {e}')
            return

        # EMA filter
        if self.filtered_pressure is None:
            self.filtered_pressure = pressure
            self.ref_pressure = pressure  # anchor floor reference on first sample
        else:
            self.filtered_pressure = (
                self.alpha * self.filtered_pressure + (1.0 - self.alpha) * pressure)

        now = self.get_clock().now().to_msg()

        fp = FluidPressure()
        fp.header.stamp = now
        fp.header.frame_id = 'baro_link'
        fp.fluid_pressure = float(self.filtered_pressure)
        fp.variance = 0.0
        self.pub_pressure.publish(fp)

        t = Temperature()
        t.header.stamp = now
        t.header.frame_id = 'baro_link'
        t.temperature = float(temperature)
        t.variance = 0.0
        self.pub_temperature.publish(t)

        altitude = pressure_to_altitude(self.filtered_pressure, self.sea_level)
        self.pub_altitude.publish(Float32(data=float(altitude)))

        # Floor estimate: height climbed since the reference, in floor units.
        # Pressure falls ~12 Pa per metre, so higher floor => lower pressure.
        delta_m = pressure_to_altitude(self.filtered_pressure, self.sea_level) \
            - pressure_to_altitude(self.ref_pressure, self.sea_level)
        floor = self.ref_floor + int(round(delta_m / self.floor_height))
        if floor != self.current_floor:
            self.get_logger().info(
                f'Floor change: {self.current_floor} -> {floor} '
                f'(Δ{delta_m:+.1f} m)')
            self.current_floor = floor
        self.pub_floor.publish(Int32(data=int(self.current_floor)))

    def destroy_node(self):
        try:
            self.bus.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BaroNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
