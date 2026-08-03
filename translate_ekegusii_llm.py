import argparse
import os
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

# Domain-specific static system prompts with relevant few-shot examples
DOMAIN_PROMPTS = {
    "health": """You are an expert English-to-Ekegusii (Kisii) translator. 
Translate the input English public service announcement (PSA) into natural Ekegusii.
Keep the translation action-oriented, clear, and command-focused. 

Here are verified reference translations for the Health domain:

Example 1:
English: Wash your hands with clean water and soap.
Ekegusii: Ogocha maboko oo n'amache amachenu n'esabuni.

Example 2:
English: Boil drinking water to prevent cholera and other waterborne diseases.
Ekegusii: Koberia amache tora konywa nario ogotanga oborwaire bwa cholera n'oborwaire bw'amache.

Example 3:
English: Seek medical attention immediately if you have a high fever.
Ekegusii: Genda inyagitari egege egeere ore ogosera gose orabe n'ogosera g'omobere gokone.

Example 4:
English: Do not ignore symptoms of illness; visit the nearest health facility.
Ekegusii: Tobwate obobete bwa oborwaire; manya ogenda ase ekitengo k'obogima egege egeere.

Translate the following English sentence. Output ONLY the translation and nothing else. No introductions, no explanations, no quotes.""",

    "agriculture": """You are an expert English-to-Ekegusii (Kisii) translator. 
Translate the input English public service announcement (PSA) into natural Ekegusii.
Keep the translation action-oriented, clear, and command-focused. 

Here are verified reference translations for the Agriculture and Environment domain:

Example 1:
English: Farmers should plant seeds before the rain starts.
Ekegusii: Abarimi bache okobaria imbeo tora ekeogo egiangania.

Example 2:
English: Store harvested crops in a dry, clean place to prevent pests.
Ekegusii: Manya okobeka ebiria biarenge ase omochando oyorere n'okochenu ogotanga obonyonyi.

Example 3:
English: Vaccinate your livestock against diseases before the wet season.
Ekegusii: Ogobeka amachani y'oborwaire ase chinyamochera chiao tora rituko ri'ebiriri riacha.

Example 4:
English: Do not cut down trees near water sources to prevent drying.
Ekegusii: Toteka emete egege y'amache ogotanga okorora kw'amache.

Translate the following English sentence. Output ONLY the translation and nothing else. No introductions, no explanations, no quotes.""",

    "security & safety": """You are an expert English-to-Ekegusii (Kisii) translator. 
Translate the input English public service announcement (PSA) into natural Ekegusii.
Keep the translation action-oriented, clear, and command-focused. 

Here are verified reference translations for the Security and Safety domain:

Example 1:
English: Report any suspicious activities or strangers to the police.
Ekegusii: Manya ogotebia ekitengo k'ogotanga ebiriri ebi oborigo gose abanto batamanyikani ase obomenyo.

Example 2:
English: Lock all doors and windows before going to bed.
Ekegusii: Ogorangeria ebitwati bionsi na emenyango tora ogogenda ogotora.

Example 3:
English: Stay indoors during heavy storms and avoid walking near power lines.
Ekegusii: Manya okoba inyomba rire ekeogo ekenene ere na totambiri egege y'emete y'amafuta.

Translate the following English sentence. Output ONLY the translation and nothing else. No introductions, no explanations, no quotes.""",

    "education": """You are an expert English-to-Ekegusii (Kisii) translator. 
Translate the input English public service announcement (PSA) into natural Ekegusii.
Keep the translation action-oriented, clear, and command-focused. 

Here are verified reference translations for the Education domain:

Example 1:
English: Parents must enroll their children in school for basic education.
Ekegusii: Abasani bache okoira abaana babo inyamosomo ase okomanya ritang'ani.

Translate the following English sentence. Output ONLY the translation and nothing else. No introductions, no explanations, no quotes.""",

    "governance": """You are an expert English-to-Ekegusii (Kisii) translator. 
Translate the input English public service announcement (PSA) into natural Ekegusii.
Keep the translation action-oriented, clear, and command-focused. 

Here are verified reference translations for the Governance domain:

Example 1:
English: Pay your taxes on time to support public services.
Ekegusii: Manya okorua ebisero biao rituko riarengane ogoteka ebiria bia ense.

Example 2:
English: Ensure you have all required identity documents when traveling.
Ekegusii: Manya okoba n'ebitwati bionsi bianyorekanire bi'okomanya rire ogotambia.

Translate the following English sentence. Output ONLY the translation and nothing else. No introductions, no explanations, no quotes."""
}

def get_domain_prompt(domain_name):
    normalized = str(domain_name).strip().lower()
    if "security" in normalized or "safety" in normalized:
        return DOMAIN_PROMPTS["security & safety"]
    if "health" in normalized:
        return DOMAIN_PROMPTS["health"]
    if "agriculture" in normalized or "environment" in normalized:
        return DOMAIN_PROMPTS["agriculture"]
    if "education" in normalized:
        return DOMAIN_PROMPTS["education"]
    if "governance" in normalized:
        return DOMAIN_PROMPTS["governance"]
    # Fallback to health if unknown
    return DOMAIN_PROMPTS["health"]

def translate_single_sentence(client, model, english_text, system_prompt):
    # Try calling with max_completion_tokens (recommended for reasoning models like gpt-5-mini)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Translate this: {english_text}"}
            ],
            max_completion_tokens=100
        )
        return response.choices[0].message.content.strip().strip('"')
    except Exception as e:
        # Fallback to standard parameters (max_tokens and temperature for legacy models)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Translate this: {english_text}"}
                ],
                temperature=0.1,
                max_tokens=100
            )
            return response.choices[0].message.content.strip().strip('"')
        except Exception as fallback_err:
            print(f"Error translating sentence '{english_text}': {e} | Fallback: {fallback_err}")
            return None

def main():
    parser = argparse.ArgumentParser(description="Domain-Specific Static Few-Shot Ekegusii LLM Translation")
    parser.add_argument("--input", type=str, default="output/english_psas.csv", help="Input English CSV file")
    parser.add_argument("--output", type=str, default="output/psa_parallel_dataset.csv", help="Output parallel CSV file")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI / Azure API Key")
    parser.add_argument("--endpoint", type=str, default=None, help="API Endpoint URL (if using custom/Azure endpoint)")
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="Model name to use (e.g. gpt-5-mini)")
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent translation workers")
    parser.add_argument("--batch-save", type=int, default=100, help="Save to CSV every N translated records")
    parser.add_argument("--domain", type=str, default=None, help="Filter translation to a specific domain (e.g., Health, Agriculture, Security, Education, Governance)")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of records to translate in this run (for phased execution)")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        print("Error: API Key is required. Please provide it via --api-key or set the OPENAI_API_KEY environment variable.")
        return

    endpoint = args.endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
    model_name = args.model
    
    # Auto-detect if using Azure OpenAI credentials
    if api_key and (not api_key.startswith("sk-") or "azure" in (endpoint or "").lower()):
        if not endpoint:
            endpoint = "https://sagebremariam-4420-resource.services.ai.azure.com/openai/v1"
        if model_name in ["gpt-5-mini", "gpt-4o-mini"]:
            model_name = "psa-generator"
            print(f"Auto-detected Azure OpenAI key. Mapping model '{args.model}' to deployment '{model_name}'.")
        print(f"Using Azure OpenAI endpoint: {endpoint}")
        client = OpenAI(api_key=api_key, base_url=endpoint)
    else:
        if endpoint:
            client = OpenAI(api_key=api_key, base_url=endpoint)
        else:
            client = OpenAI(api_key=api_key)

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return

    # Read/Initialize parallel dataset
    df_input = pd.read_csv(args.input)
    
    if os.path.exists(args.output):
        df = pd.read_csv(args.output)
        if "English" not in df.columns or len(df) != len(df_input) or set(df["English"]) != set(df_input["English"]):
            print("Existing output parallel dataset does not match input English file. Initializing a fresh copy.")
            df = df_input.copy()
        else:
            # Re-align existing df to df_input order temporarily during execution
            df = df.set_index("English").reindex(df_input["English"]).reset_index()
    else:
        df = df_input.copy()

    if "Ekegusii" not in df.columns:
        df["Ekegusii"] = ""

    # Identify target indices to translate: first 15k and last 15k records
    num_records = len(df)
    first_15k_indices = set(range(min(15000, num_records)))
    last_15k_indices = set(range(max(0, num_records - 15000), num_records))
    target_subset_indices = first_15k_indices.union(last_15k_indices)

    # Determine unique domains to translate
    if args.domain:
        domains_to_process = [args.domain]
    else:
        # Get unique domains from the dataset
        domains_to_process = df["Domain"].dropna().unique().tolist()

    print(f"Starting domain-specific static few-shot translation to Ekegusii (first 15k + last 15k only).")
    print(f"Domains to process: {domains_to_process}")

    limit = args.limit
    translated_this_run = 0

    for domain in domains_to_process:
        if limit is not None and translated_this_run >= limit:
            print(f"Reached limit of {limit} translations for this run. Stopping phase.")
            break

        # Get prompt for this domain
        prompt = get_domain_prompt(domain)
        
        # Get untranslated records matching this domain
        domain_mask = (df["Domain"].str.lower() == domain.lower())
        untranslated_mask = (df["Ekegusii"].isna() | (df["Ekegusii"] == ""))
        
        all_domain_untranslated = df[domain_mask & untranslated_mask].index.tolist()
        # Restrict to first 15k and last 15k subset
        target_indices = [idx for idx in all_domain_untranslated if idx in target_subset_indices]
        
        # Apply remaining phase limit
        if limit is not None:
            remaining_limit = limit - translated_this_run
            target_indices = target_indices[:remaining_limit]

        total_for_domain = len(target_indices)
        if total_for_domain == 0:
            continue
            
        print(f"\n=== Translating Domain: '{domain}' ({total_for_domain} records in this phase) ===")
        
        completed_count = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_idx = {
                executor.submit(translate_single_sentence, client, model_name, df.at[idx, "English"], prompt): idx
                for idx in target_indices
            }
            
            for future in tqdm(as_completed(future_to_idx), total=total_for_domain, desc=f"Translating {domain}"):
                idx = future_to_idx[future]
                try:
                    translation = future.result()
                    if translation:
                        df.at[idx, "Ekegusii"] = translation
                        translated_this_run += 1
                except Exception as e:
                    print(f"Worker exception for row {idx} in domain {domain}: {e}")
                    
                completed_count += 1
                if completed_count % args.batch_save == 0:
                    df.to_csv(args.output, index=False, encoding="utf-8")
        
        # Save after completing each domain
        df.to_csv(args.output, index=False, encoding="utf-8")
        print(f"Completed translation for domain '{domain}'.")

    # Check if there are any untranslated target subset records remaining across all domains
    domain_mask = df["Domain"].str.lower().isin([d.lower() for d in domains_to_process])
    untranslated_mask = (df["Ekegusii"].isna() | (df["Ekegusii"] == ""))
    all_domain_untranslated = df[domain_mask & untranslated_mask].index.tolist()
    remaining_in_subset = [idx for idx in all_domain_untranslated if idx in target_subset_indices]

    if len(remaining_in_subset) == 0:
        print("\nAll target records translated! Shuffling the final dataset...")
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df_shuffled.to_csv(args.output, index=False, encoding="utf-8")
        print(f"All translations completed successfully! Shuffled and saved to '{args.output}'.")
    else:
        df.to_csv(args.output, index=False, encoding="utf-8")
        total_target_records = len(target_subset_indices)
        translated_target_records = total_target_records - len(remaining_in_subset)
        print(f"\nPhase completed successfully! Progress: {translated_target_records} / {total_target_records} target records translated. Saved to '{args.output}'. Resumable next run.")

if __name__ == "__main__":
    main()
