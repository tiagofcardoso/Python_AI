
import serial
import time

SERIAL_PORT = 'COM3'  # ou '/dev/ttyUSB0' para Linux/Mac
BAUDRATE = 9600

def read_clip_count():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE)
        ser.reset_input_buffer()
        
        while True:
            data = ser.readline().decode().strip()
            
            if data:
                print(data)
                
            time.sleep(0.1)  # Verificar se é necessário
            
    except KeyboardInterrupt:
        ser.close()
    finally:
        ser.close()

read_clip_count()
