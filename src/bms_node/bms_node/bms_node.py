import asyncio
import math
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Bool, Float32

from bleak import BleakClient

ADDRESS = '41:1A:04:01:05:83'
WRITE_CHAR = '0000fff2-0000-1000-8000-00805f9b34fb'
NOTIFY_CHAR = '0000fff1-0000-1000-8000-00805f9b34fb'
CMD = bytes.fromhex('8103000000405bfa')

LOW_BATTERY_THRESHOLD = 20.0


class BMSNode(Node):
    def __init__(self):
        super().__init__('bms_node')
        self.pub_voltage = self.create_publisher(Float32, '/bms/voltage', 10)
        self.pub_soc = self.create_publisher(Float32, '/bms/soc', 10)
        self.pub_cell1 = self.create_publisher(Float32, '/bms/cell1', 10)
        self.pub_cell2 = self.create_publisher(Float32, '/bms/cell2', 10)
        self.pub_cell3 = self.create_publisher(Float32, '/bms/cell3', 10)
        self.pub_low_battery    = self.create_publisher(Bool,         '/bms/low_battery',    10)
        self.pub_battery_state  = self.create_publisher(BatteryState, '/battery/status',     10)
        self.pub_battery_voltage = self.create_publisher(Float32,     '/bms/battery_voltage', 10)
        self.get_logger().info('BMS node started')

        self.ble_thread = threading.Thread(target=self.run_ble, daemon=True)
        self.ble_thread.start()

    def parse(self, data):
        if len(data) < 122 or data[0] != 0x51:
            return
        if data[1] == 0x03 and data[2] == 0x80:
            c1 = int.from_bytes(data[3:5], 'big') / 1000
            c2 = int.from_bytes(data[5:7], 'big') / 1000
            c3 = int.from_bytes(data[7:9], 'big') / 1000
            voltage = int.from_bytes(data[115:117], 'big') / 10
            soc = int.from_bytes(data[119:121], 'big') / 10

            self.pub_voltage.publish(Float32(data=voltage))
            self.pub_battery_voltage.publish(Float32(data=voltage))
            self.pub_soc.publish(Float32(data=soc))
            self.pub_cell1.publish(Float32(data=c1))
            self.pub_cell2.publish(Float32(data=c2))
            self.pub_cell3.publish(Float32(data=c3))

            bat = BatteryState()
            bat.header.stamp = self.get_clock().now().to_msg()
            bat.voltage = float(voltage)
            bat.percentage = float(soc / 100.0)
            bat.cell_voltage = [float(c1), float(c2), float(c3)]
            bat.present = True
            bat.power_supply_technology = BatteryState.POWER_SUPPLY_TECHNOLOGY_LION
            bat.temperature = math.nan
            bat.current = math.nan
            bat.charge = math.nan
            bat.capacity = math.nan
            bat.design_capacity = math.nan
            self.pub_battery_state.publish(bat)

            if soc < LOW_BATTERY_THRESHOLD:
                self.pub_low_battery.publish(Bool(data=True))
                self.get_logger().warn(f'Low battery: {soc}%')

            self.get_logger().info(
                f'Pack: {voltage}V | SOC: {soc}% | '
                f'Cells: {c1}V {c2}V {c3}V'
            )

    def _on_notify(self, sender, data):
        self.parse(data)

    def run_ble(self):
        asyncio.run(self.ble_loop())

    async def ble_loop(self):
        while rclpy.ok():
            try:
                async with BleakClient(ADDRESS) as client:
                    await client.start_notify(NOTIFY_CHAR, self._on_notify)
                    self.get_logger().info('BMS connected')
                    while rclpy.ok():
                        await client.write_gatt_char(WRITE_CHAR, CMD, response=False)
                        await asyncio.sleep(2)
            except Exception as e:
                self.get_logger().warn(f'BMS disconnected: {e}, retrying in 5s')
                await asyncio.sleep(5)


def main(args=None):
    rclpy.init(args=args)
    node = BMSNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
