import os
import requests
import json
import hashlib
import time
import pyzipper

API_KEY = 'f409659284cfc9be60d123bddb5ed3cd13236ecf3ec55ae271147eadaac9c617'
MALWARE_DATASET_DIR = '../MalwareBazaar/malware_dataset_bazaar/'
REPORT_DIR = 'report_bazaar'
REQUEST_DELAY = 10  # Delay in seconds between requests to avoid rate limiting
MAX_RETRIES = 3  # Maximum number of retries for a request
PASSWORD = 'infected'  # Replace with the actual password

def get_file_hash(file_path):
    hasher = hashlib.md5()
    try:
        with pyzipper.AESZipFile(file_path) as zf:
            zf.pwd = PASSWORD.encode()
            for file_info in zf.infolist():
                with zf.open(file_info) as f:
                    buf = f.read()
                    hasher.update(buf)
    except pyzipper.zipfile.BadZipFile:
        print(f"Errore: Il file {file_path} non è un file zip valido.")
        return None
    return hasher.hexdigest()

def get_virus_total_report(file_hash):
    url = 'https://www.virustotal.com/vtapi/v2/file/report'
    params = {
        'apikey': API_KEY,
        'resource': file_hash
    }
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                try:
                    return response.json()
                except json.JSONDecodeError:
                    print(f"Errore nel decodificare la risposta JSON per il file hash {file_hash}")
                    print(f"Contenuto della risposta: {response.text}")
                    return None
            elif response.status_code == 204:
                print(f"Nessun contenuto per il file hash {file_hash}. Verifica l'API key o il file hash.")
                return None
            else:
                print(f"Errore nella richiesta per il file hash {file_hash}: {response.status_code}")
                print(f"Contenuto della risposta: {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Errore nella richiesta per il file hash {file_hash}: {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"Retrying... ({attempt + 1}/{MAX_RETRIES})")
                time.sleep(REQUEST_DELAY)
            else:
                print("Max retries reached. Skipping this file.")
                return None

def generate_reports():
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)
    
    for file_name in os.listdir(MALWARE_DATASET_DIR):
        file_path = os.path.join(MALWARE_DATASET_DIR, file_name)
        if os.path.isfile(file_path):
            file_hash = get_file_hash(file_path)
            if file_hash:
                report = get_virus_total_report(file_hash)
                if report:
                    sanitized_file_name = file_name.replace(" ", "").replace(".bin", ".json")
                    report_path = os.path.join(REPORT_DIR, sanitized_file_name)
                    # Scrivi il report JSON senza spazi
                    with open(report_path, 'w') as report_file:
                        json.dump(report, report_file, separators=(',', ':'))
            time.sleep(REQUEST_DELAY)  # Delay to avoid rate limiting

if __name__ == '__main__':
    generate_reports()