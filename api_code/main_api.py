# main_api.py
import argparse
import importlib
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # for vmmu_dataset.py

from vmmu_dataset import SPLITS, load_split

PROVIDERS = ['openai', 'claude', 'gemini', 'aya', 'openrouter']
KEY_PLACEHOLDER = "Input in your key"


def read_api_key(key_path, env_var):
    """Key from api_key/<provider>_key.txt, or from the provider's environment variable."""
    if os.path.exists(key_path):
        with open(key_path, 'r') as f:
            key = f.read().strip()
        if key and key != KEY_PLACEHOLDER:
            return key
    key = os.environ.get(env_var)
    if key:
        return key
    raise FileNotFoundError(f"No API key found. Put your key in {key_path} or set the {env_var} environment variable.")

def ensure_dir_exists(file_path):
    directory = os.path.dirname(file_path)
    if directory:
        Path(directory).mkdir(parents=True, exist_ok=True)

def load_existing_results(output_file):
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: {output_file} exists but is not valid JSON. Starting fresh.")
    return []

def get_processed_ids(results):
    return {item['ID'] for item in results if 'ID' in item}

def get_output_filename(data_name, base_output_dir, model_folder, prompt_lang):
    return os.path.join(base_output_dir, model_folder, f"{data_name}_{prompt_lang}.json")

def save_results(results, output_file, lock):
    with lock:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

def print_summary(output_file, model, item_ids):
    """Score this run's items and print per-subject extraction rate and accuracy (same scoring as src/result.py)."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from result import calculate_rates_by_subject, load_and_process_json, order_subjects_custom

    scored = [r for r in load_and_process_json(output_file, model.replace("/", "_")) if r['ID'] in item_ids]
    if not scored:
        return
    rates = calculate_rates_by_subject(scored)
    subjects = order_subjects_custom(set(rates))

    print(f"\n{model} | {os.path.basename(output_file)} | {len(scored)} questions")
    print(f"|{'Subject':<12}|{'N':>5}|{'Extraction':>11}|{'Accuracy':>9}|")
    print(f"|{'-' * 12}|{'-' * 5}|{'-' * 11}|{'-' * 9}|")
    for subject in subjects:
        r = rates[subject]
        print(f"|{subject:<12}|{r['total_questions']:>5}|{r['extraction_rate'] * 100:>10.2f}%|{r['accuracy_rate'] * 100:>8.2f}%|")
    overall_extraction = sum(r['has_extraction'] for r in scored) / len(scored) * 100
    overall_accuracy = sum(r['is_correct'] for r in scored) / len(scored) * 100
    print(f"|{'Overall':<12}|{len(scored):>5}|{overall_extraction:>10.2f}%|{overall_accuracy:>8.2f}%|")
    # "Mean" = average over subjects, the number src/result.py's tables report
    mean_extraction = sum(rates[s]['extraction_rate'] for s in subjects) / len(subjects) * 100
    mean_accuracy = sum(rates[s]['accuracy_rate'] for s in subjects) / len(subjects) * 100
    print(f"|{'Mean':<12}|{'':>5}|{mean_extraction:>10.2f}%|{mean_accuracy:>8.2f}%|")

def get_model_provider(model_name):
    """Determines the API provider based on the model name."""
    model_lower = model_name.lower()
    if 'gpt' in model_lower or 'o3' in model_lower:
        return 'openai'
    elif 'claude' in model_lower:
        return 'claude'
    elif 'gemini' in model_lower or 'gemma' in model_lower or model_lower.startswith('google/'):
        return 'openrouter'
    elif 'aya' in model_lower or 'command-a-vision' in model_lower:
        return 'aya'
    elif 'qwen' in model_lower or 'deepseek' in model_lower or 'mistral' in model_lower or 'llama' in model_lower:
        return 'openrouter'
    else:
        # Default to OpenRouter, since it serves many models
        print(f"Warning: Could not determine provider for '{model_name}'. Defaulting to 'openrouter'.")
        return 'openrouter'

# ==============================================================================
# MAIN FUNCTION
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Unified API caller for Vision-Language Models.")
    # Common arguments
    parser.add_argument("--model", type=str, required=True, help="Model to use (e.g., 'gpt-4.1-2025-04-14', 'claude-sonnet-4-20250514', 'qwen/qwen2.5-vl-72b-instruct')")
    parser.add_argument("--provider", type=str, default=None, choices=PROVIDERS, help="API provider. Default: detected from the model name (e.g. use --provider gemini to call Google's API directly instead of OpenRouter).")
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument("--split", type=str, default=None, choices=SPLITS, help="VMMU split to download from Hugging Face (anvo25/vmmu). Default: full_vqa.")
    data_group.add_argument("--input-file", type=str, default=None, help="Local metadata JSON instead of --split (e.g. dataset/metadata/random_subset_vqa.json); its image_path files must exist.")
    parser.add_argument("--output-file", type=str, default=None, help="Optional: Manually specify output file path.")
    parser.add_argument("--output-dir", type=str, default=None, help="Base directory for saving result files. Default: results/ in the repo root.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for the model")
    parser.add_argument("--test", type=int, default=None, metavar="N", help="Test mode with N samples.")
    parser.add_argument("--concurrency", type=int, default=300, help="Number of concurrent API calls")
    parser.add_argument("--prompt_language", type=str, default="vn", choices=['vn', 'en'], help="Choose prompt language ('vn' for Vietnamese or 'en' for English)")
    parser.add_argument("--reasoning-effort", type=str, default="low", choices=['low', 'medium', 'high'], help="Reasoning effort for o3 / gpt-5 models and for claude-opus-4-6 extended thinking")
    parser.add_argument(
        "--openrouter-ignore-providers",
        type=str,
        default=None,
        help="For OpenRouter only: Comma-separated list of providers to ignore (e.g., 'anthropic,google')"
    )

    args = parser.parse_args()

    # Paths given on the command line are relative to where the command was run; everything else
    # (api_key/, dataset/, results/) is relative to the repo root.
    for attr in ("input_file", "output_file", "output_dir"):
        if getattr(args, attr):
            setattr(args, attr, os.path.abspath(getattr(args, attr)))
    os.chdir(REPO_ROOT)
    args.output_dir = args.output_dir or "results"

    # 1. Determine the provider and corresponding handler
    provider = args.provider or get_model_provider(args.model)
    handler = importlib.import_module(f"api_handlers.{provider}_handler")

    print(f"Model provider: {provider.upper()}")

    if provider == 'openrouter' and args.openrouter_ignore_providers:
        print(f"OpenRouter: Ignoring providers -> {args.openrouter_ignore_providers}")

    # 2. Get API key path and read the key
    api_key = read_api_key(handler.get_api_key_path(), handler.API_KEY_ENV)

    # 3. Load data
    if args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if args.test is not None:
            data = data[:args.test]
        data_name = os.path.splitext(os.path.basename(args.input_file))[0]
        missing = [item['image_path'] for item in data if item.get('image_path') and not os.path.exists(item['image_path'])]
        if missing:
            raise FileNotFoundError(f"{len(missing)} image(s) listed in {args.input_file} are missing (e.g. {missing[0]}). "
                                    f"The repo only ships a few sample images; use --split to download the dataset from Hugging Face.")
    else:
        split = args.split or "full_vqa"
        data = load_split(split, limit=args.test)
        data_name = split

    if args.test is not None:
        print(f"--- Running in TEST MODE with {args.test} samples ---")

    prompt_key = f"{args.prompt_language}_prompt"
    if not any(item.get(prompt_key) for item in data):
        raise ValueError(f"'{data_name}' has no {prompt_key}. The cropped_* splits only have Vietnamese prompts: use --prompt_language vn.")

    # 4. Create the output file path
    model_folder_name = args.model.replace("/", "_")  # '/' is not valid in a folder name
    output_file = args.output_file or get_output_filename(data_name, args.output_dir, model_folder_name, args.prompt_language)
    ensure_dir_exists(output_file)

    results = load_existing_results(output_file)
    processed_ids = get_processed_ids(results)

    print(f"Found {len(results)} existing results. Continuing from where we left off.")
    print(f"Using model: {args.model}")
    print(f"Output file: {output_file}")

    results_lock = threading.Lock()
    save_lock = threading.Lock()

    items_to_process = [item for item in data if item['ID'] not in processed_ids]
    print(f"Found {len(items_to_process)} new items to process out of {len(data)} total.")

    # 5. Main processing loop
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_to_item = {}
        for item in items_to_process:
            process_kwargs = {
                'item': item,
                'api_key': api_key,
                'model': args.model,
                'temperature': args.temperature,
                'prompt_lang': args.prompt_language
            }

            # reasoning_effort: OpenAI reasoning models, and extended thinking for claude-opus-4-6
            if provider == 'openai':
                process_kwargs['reasoning_effort'] = args.reasoning_effort
            elif args.model == 'claude-opus-4-6' and args.reasoning_effort:
                process_kwargs['reasoning_effort'] = args.reasoning_effort

            # If provider is 'openrouter', add 'ignore_providers'
            if provider == 'openrouter':
                process_kwargs['ignore_providers'] = args.openrouter_ignore_providers

            future = executor.submit(handler.process_item, **process_kwargs)
            future_to_item[future] = item

        for i, future in enumerate(as_completed(future_to_item)):
            item = future_to_item[future]
            try:
                result = future.result()
                if result:
                    with results_lock:
                        results.append(result)

                    if (i + 1) % 5 == 0 or i == len(items_to_process) - 1:
                        save_results(results, output_file, save_lock)
                        print(f"  → Saved progress ({len(results)}/{len(data)} items)")
            except Exception as exc:
                print(f"Item {item['ID']} generated an exception: {exc}")

    save_results(results, output_file, save_lock)
    failed = len(data) - len(get_processed_ids(results) & {item['ID'] for item in data})
    print(f"All done! Processed {len(items_to_process)} new items. Total results: {len(results)}. Saved to {output_file}")
    if failed:
        print(f"{failed} item(s) failed. Run the same command again to retry them.")

    if "ocr" in data_name:
        print("OCR run: responses are transcriptions, not answers, so no accuracy table is printed.")
    else:
        print_summary(output_file, args.model, {item['ID'] for item in data})

if __name__ == "__main__":
    main()
