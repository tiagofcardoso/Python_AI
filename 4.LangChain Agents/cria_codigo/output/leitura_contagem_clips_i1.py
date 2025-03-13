
import serial
import time

# Configuração da comunicação serial
port = 'COM3'  # ou '/dev/ttyUSB0' para Linux/Mac
baudrate = 9600

# Abre a conexão serial
ser = serial.Serial(port, baudrate)

try:
    while True:
        # Lê os dados do buffer serial
        data = ser.readline().decode().strip()
        
        if data:
            print(data)
        
        time.sleep(0.1)  # Verificar se é necessário
    
except KeyboardInterrupt:
    ser.close()

ser.close()