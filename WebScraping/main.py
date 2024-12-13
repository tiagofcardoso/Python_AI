import requests
from bs4 import BeautifulSoup
import csv
import os

def collect_data_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Coletando dados de texto
    text_data = soup.get_text()
    
    # Coletando URLs de imagens
    image_data = []
    for img in soup.find_all('img'):
        image_url = img.get('src')
        if image_url:
            image_data.append(image_url)
    
    return text_data, image_data

# URL da página web que deseja coletar dados
url = 'https://moodle.livetraining.pt/course/view.php?id=11660'

# Coletando dados da página principal
main_text_data, main_image_data = collect_data_from_url(url)

# Encontrando e coletando dados dos sublinks
sublinks_data = []
soup = BeautifulSoup(requests.get(url).content, 'html.parser')
for link in soup.find_all('a', href=True):
    sublink_url = link['href']
    if sublink_url.startswith('http'):
        sublink_text_data, sublink_image_data = collect_data_from_url(sublink_url)
        sublinks_data.append((sublink_url, sublink_text_data, sublink_image_data))

# Exportando dados coletados para um arquivo CSV
csv_file = 'data_exported.csv'
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file, delimiter=';')
    writer.writerow(['URL', 'Text', 'Image URLs'])
    writer.writerow([url, main_text_data, ', '.join(main_image_data)])
    for sublink_url, sublink_text_data, sublink_image_data in sublinks_data:
        writer.writerow([sublink_url, sublink_text_data, ', '.join(sublink_image_data)])

print(f'Dados exportados para {csv_file}')