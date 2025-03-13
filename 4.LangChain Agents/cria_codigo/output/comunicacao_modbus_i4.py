
import pymodbus
from pymodbus.client import ModbusTcpClient
from enum import Enum

class ModbusReader:
    class ModbusError(Enum):
        SUCCESS = 0
        CONNECTION_ERROR = 1
        READ_ERROR = 2

    def __init__(self, ip_address='192.168.1.100', port=1700):
        self.ip_address = ip_address
        self.port = port
        self.client = None

    def connect(self) -> ModbusReader.ModbusError:
        try:
            self.client = ModbusTcpClient(self.ip_address, self.port)
            if not self.client.connect():
                return ModbusReader.ModbusError.CONNECTION_ERROR
            return ModbusReader.ModbusError.SUCCESS
        except Exception as e:
            print(f"Error connecting to {self.ip_address}: {e}")
            return ModbusReader.ModbusError.READ_ERROR

    def read_clip_count(self) -> int:
        if not self.client:
            return None
        try:
            result = self.client.read_holding_registers(0x0000, 2)
            if result.isError():
                return None
            self.client.close()
            clip_count = (result.getRegister(0) << 16) | result.getRegister(1)
            return clip_count
        except Exception as e:
            print(f"Error reading from {self.ip_address}: {e}")
            self.client.close()
            return None

    def __del__(self):
        if self.client:
            self.client.close()


if __name__ == "__main__":
    reader = ModbusReader()
    result = reader.connect()
    if result == ModbusReader.ModbusError.SUCCESS:
        count = reader.read_clip_count()
        print(count)
