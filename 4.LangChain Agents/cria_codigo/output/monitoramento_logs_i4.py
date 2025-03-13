
import logging
import time
from datetime import datetime

logging.basicConfig(filename='system_logs.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SystemMonitor:
    def __init__(self):
        self.logger = logging.getLogger('SystemMonitor')
        self.logger.setLevel(logging.INFO)

    def monitor(self):
        while True:
            try:
                # Monitorar sistema...
                self.logger.info(f'Sistema em funcionamento às {datetime.now()}')
                time.sleep(60)  # Aguardar 1 minuto antes de reavaliar
            except Exception as e:
                self.logger.error(f'Erro ao monitorar sistema: {e}')

    def sleep(self, seconds):
        time.sleep(seconds)

if __name__ == '__main__':
    SystemMonitor().monitor()
