import argparse
import json
import random
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
Do NOT translate or alter official Kenyan institution names, acronyms, or administrative programs (e.g. NDMA, HELB, NTSA, KUCCPS, Huduma, KNEC, KRA, SHA, NSSF, NHIF, NC4, etc.). Keep them exactly as written in English in your translation.

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
Do NOT translate or alter official Kenyan institution names, acronyms, or administrative programs (e.g. NDMA, HELB, NTSA, KUCCPS, Huduma, KNEC, KRA, SHA, NSSF, NHIF, NC4, etc.). Keep them exactly as written in English in your translation.

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
Do NOT translate or alter official Kenyan institution names, acronyms, or administrative programs (e.g. NDMA, HELB, NTSA, KUCCPS, Huduma, KNEC, KRA, SHA, NSSF, NHIF, NC4, etc.). Keep them exactly as written in English in your translation.

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
Do NOT translate or alter official Kenyan institution names, acronyms, or administrative programs (e.g. NDMA, HELB, NTSA, KUCCPS, Huduma, KNEC, KRA, SHA, NSSF, NHIF, NC4, etc.). Keep them exactly as written in English in your translation.

Here are verified reference translations for the Education domain:

Example 1:
English: Parents must enroll their children in school for basic education.
Ekegusii: Abasani bache okoira abaana babo inyamosomo ase okomanya ritang'ani.

Translate the following English sentence. Output ONLY the translation and nothing else. No introductions, no explanations, no quotes.""",

    "governance": """You are an expert English-to-Ekegusii (Kisii) translator. 
Translate the input English public service announcement (PSA) into natural Ekegusii.
Keep the translation action-oriented, clear, and command-focused. 
Do NOT translate or alter official Kenyan institution names, acronyms, or administrative programs (e.g. NDMA, HELB, NTSA, KUCCPS, Huduma, KNEC, KRA, SHA, NSSF, NHIF, NC4, etc.). Keep them exactly as written in English in your translation.

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

def get_batch_prompt(domain_name):
    base_prompt = get_domain_prompt(domain_name)
    batch_instruction = """Translate the list of English sentences provided by the user into Ekegusii.
Output your translations as a raw JSON array of strings in the exact same order: ["Translation 1", "Translation 2", ...]. 
Output ONLY the JSON array, no explanation, no quotes, no markdown formatting (do not wrap in ```json)."""
    
    lines = base_prompt.split("\n")
    filtered_lines = [l for l in lines if "Translate the following English sentence" not in l and "Output ONLY the translation" not in l]
    return "\n".join(filtered_lines) + "\n\n" + batch_instruction

def translate_batch(client, model, english_texts, system_prompt):
    max_retries = 3
    backoff_factor = 2
    user_content = json.dumps(english_texts, indent=2)
    
    for attempt in range(max_retries):
        try:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Translate these sentences:\n{user_content}"}
                    ],
                    max_completion_tokens=4000,
                    reasoning_effort="low"
                )
            except Exception:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Translate these sentences:\n{user_content}"}
                    ],
                    max_completion_tokens=4000
                )
            content = response.choices[0].message.content.strip()
            
            # Clean markdown formatting if present
            if content.startswith("```"):
                lines = content.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines).strip()
                
            translations = json.loads(content)
            if isinstance(translations, list) and len(translations) == len(english_texts):
                return [str(t).strip() for t in translations]
        except Exception as e:
            err_msg = str(e)
            if "rate_limit" in err_msg or "429" in err_msg or "too_many_requests" in err_msg:
                sleep_time = (backoff_factor ** attempt) + random.uniform(0.5, 1.5)
                time.sleep(sleep_time)
                continue
            # For content safety or JSON parsing errors, break early to trigger individual fallback
            break
    return None

def translate_single_sentence(client, model, english_text, system_prompt):
    max_retries = 5
    backoff_factor = 2
    
    for attempt in range(max_retries):
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
            err_msg = str(e)
            
            # Handle Azure Content Filter violation
            if "content_filter" in err_msg or "ResponsibleAIPolicyViolation" in err_msg:
                print(f"\nRow skipped due to Azure Content Filter violation: '{english_text}'")
                return "[Content Filtered]"
                
            # Handle Rate Limits (429) or Server Errors (5xx)
            if "rate_limit" in err_msg or "429" in err_msg or "too_many_requests" in err_msg or "500" in err_msg or "503" in err_msg:
                sleep_time = (backoff_factor ** attempt) + random.uniform(0.5, 1.5)
                print(f"\nRate limit hit (429). Retrying in {sleep_time:.2f} seconds (Attempt {attempt+1}/{max_retries}) for: '{english_text[:50]}...'")
                time.sleep(sleep_time)
                continue
                
            # For other exceptions or on last attempt, try the legacy parameter fallback
            if attempt == max_retries - 1:
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
                    fallback_err_msg = str(fallback_err)
                    if "content_filter" in fallback_err_msg or "ResponsibleAIPolicyViolation" in fallback_err_msg:
                        print(f"\nRow skipped due to Azure Content Filter violation in fallback: '{english_text}'")
                        return "[Content Filtered]"
                    print(f"\nError translating sentence '{english_text}': {e} | Fallback Error: {fallback_err}")
                    return None
            
            # Wait before retry
            sleep_time = (backoff_factor ** attempt) + random.uniform(0.5, 1.5)
            time.sleep(sleep_time)
            
    return None

def main():
    parser = argparse.ArgumentParser(description="Domain-Specific Static Few-Shot Ekegusii LLM Translation")
    parser.add_argument("--input", type=str, default="output/psa_parallel_dataset.csv", help="Input English CSV file")
    parser.add_argument("--output", type=str, default="output/psa_parallel_dataset.csv", help="Output parallel CSV file")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI / Azure API Key")
    parser.add_argument("--endpoint", type=str, default=None, help="API Endpoint URL (if using custom/Azure endpoint)")
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="Model name to use (e.g. gpt-5-mini)")
    parser.add_argument("--workers", type=int, default=5, help="Number of concurrent translation workers")
    parser.add_argument("--batch-save", type=int, default=100, help="Save to CSV every N translated records")
    parser.add_argument("--domain", type=str, default=None, help="Filter translation to a specific domain (e.g., Health, Agriculture, Security, Education, Governance)")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of records to translate in this run (for phased execution)")
    parser.add_argument("--batch-size", type=int, default=50, help="Number of sentences to translate in a single LLM API call")
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

    # Priority list for loading dataset:
    # 1. Output file on Drive (args.output)
    # 2. Local workspace parallel dataset (output/psa_parallel_dataset.csv)
    # 3. Input file argument (args.input)
    target_file = None
    for candidate in [args.output, "output/psa_parallel_dataset.csv", args.input]:
        if candidate and os.path.exists(candidate):
            target_file = candidate
            break
            
    if not target_file:
        print(f"Error: Target dataset file not found.")
        return
        
    print(f"Loading parallel dataset from '{target_file}'...")
    df = pd.read_csv(target_file, low_memory=False)

    # Restore missing language columns (Kiswahili, Somali, Luo) from local workspace if missing in loaded file
    if os.path.exists("output/psa_parallel_dataset.csv") and target_file != "output/psa_parallel_dataset.csv":
        try:
            df_workspace = pd.read_csv("output/psa_parallel_dataset.csv", low_memory=False)
            for lang_col in ["Kiswahili", "Somali", "Luo"]:
                if lang_col in df_workspace.columns and (lang_col not in df.columns or df[lang_col].isna().all()):
                    print(f"Merging missing language column '{lang_col}' from workspace parallel dataset...")
                    df[lang_col] = df_workspace[lang_col]
        except Exception as e:
            print(f"Warning: Could not merge language columns from workspace: {e}")

    # Rename "Ekegussi" to "Ekegusii" if present from previous runs
    if "Ekegussi" in df.columns:
        df = df.rename(columns={"Ekegussi": "Ekegusii"})
        
    col_name = "Ekegusii"
    if col_name not in df.columns:
        df[col_name] = ""
    df[col_name] = df[col_name].fillna("").astype(str)

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
        untranslated_mask = (df[col_name].isna() | (df[col_name] == ""))
        
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
        
        batch_size = args.batch_size
        batches = [target_indices[i : i + batch_size] for i in range(0, len(target_indices), batch_size)]
        
        batch_prompt = get_batch_prompt(domain)
        single_prompt = get_domain_prompt(domain)
        
        completed_count = 0
        
        def process_batch(idx_list):
            texts = [df.at[idx, "English"] for idx in idx_list]
            # Try batch translation first
            results = translate_batch(client, model_name, texts, batch_prompt)
            if results is not None:
                return list(zip(idx_list, results))
                
            # Fallback: translate individually
            fallback_results = []
            for idx in idx_list:
                res = translate_single_sentence(client, model_name, df.at[idx, "English"], single_prompt)
                fallback_results.append((idx, res))
            return fallback_results
            
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_batch = {
                executor.submit(process_batch, batch): batch
                for batch in batches
            }
            
            for future in tqdm(as_completed(future_to_batch), total=len(batches), desc=f"Translating {domain}"):
                try:
                    batch_results = future.result()
                    if batch_results:
                        for idx, translation in batch_results:
                            if translation:
                                df.at[idx, col_name] = translation
                                translated_this_run += 1
                except Exception as e:
                    print(f"Worker exception for batch: {e}")
                    
                completed_count += 1
                if completed_count % (args.batch_save // batch_size + 1) == 0:
                    df.to_csv(args.output, index=False, encoding="utf-8")
        
        # Save after completing each domain
        df.to_csv(args.output, index=False, encoding="utf-8")
        print(f"Completed translation for domain '{domain}'.")

    # Check if there are any untranslated target subset records remaining across all domains
    domain_mask = df["Domain"].str.lower().isin([d.lower() for d in domains_to_process])
    untranslated_mask = (df[col_name].isna() | (df[col_name] == ""))
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
