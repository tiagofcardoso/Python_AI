
import pymodbus
from pymodbus.client import ModbusTcpClient

def read_clip_count(ip_address='192.168.1.100', port=1700):
    try:
        client = ModbusTcpClient(ip_address, port)
        client.connect()
        result = client.read_holding_registers(0x0000, 2)
        client.close()

        if result.isError():
            return None
        else:
            clip_count = (result.getRegister(0) << 16) | result.getRegister(1)
            return clip_count
    except Exception as e:
        print(f"Error: {e}")
        return None

print(read_clip_count())
