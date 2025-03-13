
import logging
import datetime

# Configurando o logger
logging.basicConfig(filename='system_logs.log', level=logging.INFO)

def monitor_system():
    while True:
        # Monitorar sistema...
        logging.info(f'Sistema em funcionamento às {datetime.datetime.now()}')

if __name__ == '__main__':
    monitor_system()
