
import pymodbus
from pymodbus.client import ModbusTcpClient

client = ModbusTcpClient('192.168.1.100', 1700)

def read_clip_count():
    client.connect()
    result = client.read_holding_registers(0x0000, 2)
    client.close()

    if result.isError():
        return None
    else:
        clip_count = (result.getRegister(0) << 16) | result.getRegister(1)
        return clip_count

print(read_clip_count())
