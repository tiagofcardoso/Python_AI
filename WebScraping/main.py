import requests
from bs4 import BeautifulSoup
import csv
import os
import logging

# Configurando o logger
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def collect_data_from_url(url):
    logging.debug(f'Iniciando coleta de dados da URL: {url}')
    response = requests.get(url)
    logging.debug('Requisição HTTP feita com sucesso')
    
    soup = BeautifulSoup(response.content, 'html.parser')
    logging.debug('Conteúdo HTML analisado com BeautifulSoup')
    
    # Coletando dados de texto
    text_data = soup.get_text()
    logging.debug('Dados de texto coletados')
    
    # Salvando dados de texto em um arquivo
    with open('text_data.txt', 'w', encoding='utf-8') as file:
        file.write(text_data)
    logging.debug('Dados de texto salvos em text_data.txt')
    
    # Coletando URLs de imagens
    image_data = []
    for img in soup.find_all('img'):
        image_url = img.get('src')
        if image_url:
            image_data.append(image_url)
    logging.debug(f'{len(image_data)} URLs de imagens coletadas')
    
    # Salvando URLs de imagens em um arquivo CSV
    with open('image_data.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Image URL'])
        for image_url in image_data:
            writer.writerow([image_url])
    logging.debug('URLs de imagens salvas em image_data.csv')


# Exemplo de uso
url = 'https://pt.wikipedia.org/wiki/Vin%C3%ADcius_J%C3%BAnior'
collect_data_from_url(url)
