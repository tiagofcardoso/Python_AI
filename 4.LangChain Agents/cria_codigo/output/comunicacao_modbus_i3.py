
import pymodbus
from pymodbus.client import ModbusTcpClient

class ModbusReader:
    def __init__(self, ip_address='192.168.1.100', port=1700):
        self.ip_address = ip_address
        self.port = port
        self.client = None

    def connect(self):
        try:
            self.client = ModbusTcpClient(self.ip_address, self.port)
            self.client.connect()
        except Exception as e:
            print(f"Error connecting to {self.ip_address}: {e}")

    def read_clip_count(self):
        if not self.client:
            return None
        result = self.client.read_holding_registers(0x0000, 2)
        self.client.close()

        if result.isError():
            return None
        else:
            clip_count = (result.getRegister(0) << 16) | result.getRegister(1)
            return clip_count

    def __del__(self):
        if self.client:
            self.client.close()


if __name__ == "__main__":
    reader = ModbusReader()
    reader.connect()
    count = reader.read_clip_count()
    print(count)
