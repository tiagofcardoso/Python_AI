
import logging
from datetime import datetime

logging.basicConfig(filename='system_logs.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def monitor_system():
    while True:
        try:
            # Monitorar sistema...
            logging.info(f'Sistema em funcionamento às {datetime.now()}')
        except Exception as e:
            logging.error(f'Erro ao monitorar sistema: {e}')

if __name__ == '__main__':
    monitor_system()
