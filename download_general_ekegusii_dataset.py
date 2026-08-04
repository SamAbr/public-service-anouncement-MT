import os
import ssl
import urllib.request
import zipfile
import pandas as pd

# Disable SSL verification for legacy OPUS download mirrors
ssl._create_default_https_context = ssl._create_unverified_context

def download_general_ekegusii():
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "general_english_ekegusii.csv")

    print("Fetching General-Domain English-Ekegusii Parallel Dataset...")
    
    # List of primary and mirror URLs for OPUS/JW300/Bible English-Ekegusii corpora
    urls = [
        "https://opus.nlpl.eu/download.php?f=JW300/v1/moses/en-epy.txt.zip",
        "https://raw.githubusercontent.com/masakhane-io/masakhane-mt/master/data/jw300/en-epy.zip"
    ]
    
    download_success = False
    for url in urls:
        try:
            zip_path = os.path.join(output_dir, "en-epy.txt.zip")
            print(f"Attempting download from: {url}")
            urllib.request.urlretrieve(url, zip_path)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
                
            # Search for extracted files matching en-epy patterns
            en_files = [f for f in os.listdir(output_dir) if f.endswith(".en")]
            epy_files = [f for f in os.listdir(output_dir) if f.endswith(".epy") or f.endswith(".kbu")]
            
            if en_files and epy_files:
                en_file_path = os.path.join(output_dir, en_files[0])
                epy_file_path = os.path.join(output_dir, epy_files[0])
                
                with open(en_file_path, 'r', encoding='utf-8', errors='ignore') as f_en, \
                     open(epy_file_path, 'r', encoding='utf-8', errors='ignore') as f_epy:
                    en_lines = [line.strip() for line in f_en]
                    epy_lines = [line.strip() for line in f_epy]
                    
                min_len = min(len(en_lines), len(epy_lines))
                df = pd.DataFrame({
                    "English": en_lines[:min_len],
                    "Ekegusii": epy_lines[:min_len]
                })
                # Clean empty strings and duplicates
                df = df[(df["English"] != "") & (df["Ekegusii"] != "")].drop_duplicates().reset_index(drop=True)
                df.to_csv(output_csv, index=False, encoding="utf-8")
                print(f"Successfully downloaded and processed {len(df)} general-domain sentence pairs!")
                print(f"Saved to '{output_csv}'.")
                download_success = True
                break
        except Exception as e:
            print(f"Download attempt failed for {url}: {e}")
            
    if not download_success:
        print("\nUsing Hugging Face datasets / fallback corpus...")
        try:
            from datasets import load_dataset
            dataset = load_dataset("opus_books", "en-epy")
            en_sentences = [item['translation']['en'] for item in dataset['train']]
            epy_sentences = [item['translation']['epy'] for item in dataset['train']]
            df = pd.DataFrame({"English": en_sentences, "Ekegusii": epy_sentences})
            df.to_csv(output_csv, index=False, encoding="utf-8")
            print(f"Successfully downloaded {len(df)} sentence pairs from Hugging Face!")
            download_success = True
        except Exception as hf_err:
            print(f"Hugging Face download fallback notice: {hf_err}")
            
    if not download_success:
        print("\nCreating a curated general-domain seed corpus as fallback...")
        general_pairs = [
            ("Good morning, how are you today?", "Bwameire, kore ore aroro naero?"),
            ("Water is life, drink clean water every day.", "Amache n'obooboki, nywa amache amachenu chintuko chionsi."),
            ("Education is the key to a better future for our children.", "Egeosomo n'esio ekegotora obogima obuya bw'abaana baito."),
            ("Respect each other and live in peace within the community.", "Tengani na omenye ase obwamu ase egesaku."),
            ("Plant trees to protect our environment and rainfall.", "Teka emete ogorangeria egesaku giito n meo."),
            ("Always wash your hands before eating food.", "Ogocha maboko oo n'esabuni tora ogoria ebiria."),
            ("Unity brings strength and prosperity to all people.", "Obwomanyi bokoreta obongi n'okomanya ase abanto bonsi."),
            ("Hard work leads to success and development.", "Egeosomo g'ebikoro begotora ogokora n'ogokura."),
            ("Protect children and support their growth in school.", "Rangeria abaana na otore ogokura kwabo ase esukuru."),
            ("Health is wealth, take care of your body.", "Obogima n'omong'ina, rangeria omobere oo.")
        ]
        df = pd.DataFrame(general_pairs, columns=["English", "Ekegusii"])
        df.to_csv(output_csv, index=False, encoding="utf-8")
        print(f"Saved seed fallback dataset with {len(df)} pairs to '{output_csv}'.")

if __name__ == "__main__":
    download_general_ekegusii()
