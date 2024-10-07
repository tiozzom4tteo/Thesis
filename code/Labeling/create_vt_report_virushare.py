import os
import requests
import json
import hashlib
import time
from tqdm import tqdm

from dotenv import load_dotenv

load_dotenv('variables.env')

API_KEY = os.getenv('YOUR_API_KEY')
MALWARE_DATASET_DIR = '../Malware/malware_dataset_virushare_2024/'
REPORT_DIR = 'report_virushare'
REQUEST_DELAY = 20  
INITIAL_DELAY = 30  
MAX_RETRIES = 3  
MAX_REPORTS = 500  

def get_file_hash(file_path):
    hasher = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
    except Exception as e:
        print(f"Errore: Impossibile leggere il file {file_path}.")
        print(e)
        return None
    return hasher.hexdigest()

def upload_file_to_virus_total(file_path):
    url = 'https://www.virustotal.com/vtapi/v2/file/scan'
    params = {'apikey': API_KEY}
    with open(file_path, 'rb') as f:
        file_data = f.read()
    files = {'file': (os.path.basename(file_path), file_data)}
    total_size = len(file_data)
    with tqdm(total=total_size, unit='B', unit_scale=True, desc='Caricamento') as tqdm_bar:
        try:
            response = requests.post(url, files=files, params=params, timeout=20)
            tqdm_bar.update(total_size)
            if response.status_code == 200:
                print(response.json())
                return response.json().get('resource')
            else:
                print(f"Errore nel caricamento del file {file_path}: {response.status_code}")
                print(f"Contenuto della risposta: {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Errore nel caricamento del file {file_path}: {e}")
            return None

def check_existing_report(file_hash):
    url = 'https://www.virustotal.com/vtapi/v2/file/report'
    params = {'apikey': API_KEY, 'resource': file_hash}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data.get('response_code') == 1:  
            return data
    return None

def generate_reports():
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)
    
    report_count = 0
    for file_name in os.listdir(MALWARE_DATASET_DIR):
        if report_count >= MAX_REPORTS:
            break
        
        file_path = os.path.join(MALWARE_DATASET_DIR, file_name)
        if os.path.isfile(file_path):
            report_path = os.path.join(REPORT_DIR, file_name.replace(" ", "") + ".json")
            if os.path.exists(report_path):
                print(f"Report già esistente per {file_name}. Skipping...")
                continue
            
            file_hash = get_file_hash(file_path)
            if file_hash:
                report = check_existing_report(file_hash)
                if report:
                    with open(report_path, 'w') as report_file:
                        json.dump(report, report_file, indent=4)
                    report_count += 1
                    continue
                
                resource = upload_file_to_virus_total(file_path)
                if resource:
                    time.sleep(INITIAL_DELAY)  
                    report = check_existing_report(resource)
                    if report:
                        with open(report_path, 'w') as report_file:
                            json.dump(report, report_file, indent=4)
                        report_count += 1
                        print(f"Report generato per {file_name}")

if __name__ == '__main__':
    generate_reports()