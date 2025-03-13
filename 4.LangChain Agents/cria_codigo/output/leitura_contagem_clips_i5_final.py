
import serial
import time

SERIAL_PORT = 'COM3'  # ou '/dev/ttyUSB0' para Linux/Mac
BAUDRATE = 9600

class ClipCounter:
    def __init__(self):
        self.ser = None

    def read_clip_count(self):
        try:
            if not self.ser:
                self.ser = serial.Serial(SERIAL_PORT, BAUDRATE)
                self.ser.reset_input_buffer()
            
            while True:
                data = self.ser.readline().decode().strip()
                
                if data:
                    print(data)
                
                time.sleep(0.1)  # Verificar se é necessário
            
        except KeyboardInterrupt:
            if self.ser:
                self.ser.close()
        finally:
            if self.ser:
                self.ser.close()

if __name__ == "__main__":
    cc = ClipCounter()
    cc.read_clip_count()
