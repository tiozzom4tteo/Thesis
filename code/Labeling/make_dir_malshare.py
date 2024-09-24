import os
import json
from collections import Counter
import shutil

def create_dir(superfamily, subfamily):
    superfamily_dir = os.path.join("labels", superfamily)
    subfamily_dir = os.path.join(superfamily_dir, subfamily)
    if not os.path.exists(subfamily_dir):
        os.makedirs(subfamily_dir)

def detect_malware_type(results):
    malware_types = {
        "Trojan": ["Trojan", "Trj"],
        "Worm": ["Worm", "Wrm"],
        "Virus": ["Virus", "Vir"],
        "Ransomware": ["Ransomware", "Ransom", "Rsnw", "WannaCry"],
        "Adware": ["Adware", "Adv"],
        "Spyware": ["Spyware", "Spy"],
        "Backdoor": ["Backdoor", "Bdoor"],
        "Rootkit": ["Rootkit", "Rkit"],
        "Botnet": ["Botnet", "Bot"],
        "Cryptojacking": ["Cryptojacking", "Crypto"],
        "Exploit Kit": ["Exploit Kit", "Exploit"],
        "Scareware": ["Scareware", "Scare"],
        "Downloader/Dropper": ["Downloader", "Dropper"],
        "Malvertising": ["Malvertising", "Malad"],
        "Fileless Malware": ["Fileless", "Filels"],
        "Hybrid Malware": ["Hybrid"],
        "Polymorphic Malware": ["Polymorphic", "Poly"],
        "APT": ["APT", "Advanced Persistent Threat"]
    }

    detected_types = set()

    for result in results:
        for malware_type, keywords in malware_types.items():
            for keyword in keywords:
                if keyword.lower() in result.lower():
                    detected_types.add(malware_type)

    return list(detected_types)

def process_json_file(json_filepath):
    try:
        with open(json_filepath, 'r') as json_file:
            data = json.load(json_file)
        
        results = []
        if isinstance(data, dict):
            for key, value in data.items():
                if key == "result":
                    results.append(value)
                elif isinstance(value, dict):
                    results.extend(search_in_dict(value))

        results = [res for res in results if res is not None and res != "None"]
        
        if results:
            detected_types = detect_malware_type(results)
            if detected_types:
                superfamily = detected_types[0]
                print(f"Tipo di malware rilevato nel file: {json_filepath} - {superfamily}")
                return superfamily

    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Errore nella lettura del file JSON {json_filepath}: {e}")
    
    return None

def search_in_dict(d):
    results = []
    for key, value in d.items():
        if key == "result":
            results.append(value)
        elif isinstance(value, dict):
            results.extend(search_in_dict(value))
    return results

def read_txt_file(txt_filepath, superfamily):
    try:
        with open(txt_filepath, 'r') as file:
            content = file.read().split("\t")[1].split("\n")[0]
            subfamily = content if content else "generic"
            return subfamily
                
    except (IndexError, FileNotFoundError) as e:
        print(f"Errore nella lettura del file TXT {txt_filepath}: {e}")
        return "generic"

def move_malware_file(malware_filepath, superfamily, subfamily):
    create_dir(superfamily, subfamily)
    destination = os.path.join("labels", superfamily, subfamily, os.path.basename(malware_filepath))
    shutil.move(malware_filepath, destination)

def move_unrecognized_malware_file(malware_filepath):
    destination_dir = os.path.join("labels", "unrecognised")
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    destination = os.path.join(destination_dir, os.path.basename(malware_filepath))
    shutil.move(malware_filepath, destination)

def process_files(json_dir, txt_dir, malware_dir):
    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            json_filepath = os.path.join(json_dir, filename)
            superfamily = process_json_file(json_filepath)

            if superfamily:
                txt_filename = filename.replace(".json", ".txt")
                txt_filepath = os.path.join(txt_dir, txt_filename)
                if os.path.exists(txt_filepath):
                    subfamily = read_txt_file(txt_filepath, superfamily)
                    malware_filepath = os.path.join(malware_dir, filename.replace(".json", ""))
                    if os.path.exists(malware_filepath):
                        move_malware_file(malware_filepath, superfamily, subfamily)
                    else:
                        print(f"File non trovato per {filename}")
                else:
                    print(f"File TXT non trovato per {filename}")
            else:
                malware_filepath = os.path.join(malware_dir, filename.replace(".json", ""))
                if os.path.exists(malware_filepath):
                    move_unrecognized_malware_file(malware_filepath)
                else:
                    print(f"File non trovato per {filename}")

            os.remove(json_filepath)
            txt_filepath = os.path.join(txt_dir, filename.replace(".json", ".txt"))
            if os.path.exists(txt_filepath):
                os.remove(txt_filepath)

if __name__ == "__main__":
    json_directory = "report_malshare/"
    txt_directory = "avclass_malshare/"
    malware_directory = "../Malware/malware_dataset_malshare/"
    process_files(json_directory, txt_directory, malware_directory)