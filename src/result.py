"""Extraction-rate and accuracy tables, per subject, for every result file under results/.

Result files are found automatically: results/<model>/<dataset>_<prompt_language>.json, as written
by api_code/main_api.py (a trailing _YYYYMMDD_HHMMSS timestamp, as in older OpenAI Batch API
results, is ignored). One set of tables is produced per <dataset>_<prompt_language>.

    python src/result.py                      # every result file
    python src/result.py --file full_vqa_vn   # only full_vqa with Vietnamese prompts
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMERCIAL_MODEL_PATTERN = re.compile(r"^(gpt|o\d|claude|gemini|google_gemini)", re.IGNORECASE)


def extract_answer_from_text(text, is_multiple_question=None, model_name=None):
    """
    Extract answer from curly braces in text with filtering logic

    Args:
        text: The text to extract answer from
        is_multiple_question: Boolean flag from the data
            - If False: Filter out {} with more than 1 character inside
            - If True or None: Take the last match without filtering
        model_name: Model name, for the \\boxed{} formats of claude-opus-4-6 and gemini-2.5-flash

    Returns:
        Extracted answer string or None
    """
    if not text:
        return None

    # Check \boxed{} LaTeX format for Claude Opus (claude uses \boxed{\{B\}}, \boxed{{D}}, etc.)
    if model_name == 'claude-opus-4-6':
        boxed_matches = re.findall(r'\\boxed\{[^A-D]*([A-D])', text)
        if boxed_matches:
            return boxed_matches[-1]
        # Handle standalone \{X\} with backslash-escaped braces
        escaped_matches = re.findall(r'\\?\{([A-D])\\?\}', text)
        if escaped_matches:
            return escaped_matches[-1]

    # Check LaTeX format FIRST for Gemini: it contains nested braces
    if model_name == 'gemini-2.5-flash':
        # Try LaTeX format: $\boxed{\text{C}}$ or $\boxed{C}$
        boxed_matches = re.findall(r'\$\\boxed\{(?:\\text\{)?([^}]+)\}?\}\$', text)
        if boxed_matches:
            answer = boxed_matches[-1].strip()
            # Remove any remaining \text{} wrapper
            answer = re.sub(r'\\text\{([^}]+)\}', r'\1', answer)
            return answer.strip()

    # Find all matches with curly braces
    matches = re.findall(r'\{([^}]*)\}', text)
    if not matches:
        return None

    if is_multiple_question is False:
        # Only keep matches with exactly 1 character (drops {12}, {AB}, etc.)
        filtered_matches = [m.strip() for m in matches if len(m.strip()) == 1]
        return filtered_matches[-1] if filtered_matches else None
    # If multiple_question is True or None, take the last match without filtering
    return matches[-1].strip()


def normalize_answer(answer):
    """Normalize answer by removing spaces, commas, and other punctuation"""
    if not answer:
        return ""
    normalized = re.sub(r'[,\s\.\-\(\)\[\]]+', '', str(answer))
    return normalized.upper()


def get_response_text(api_response):
    """The model's answer text, from any saved api_response shape."""
    if not isinstance(api_response, dict):
        return ""
    # OpenAI Batch API wraps the completion: api_response -> body -> choices
    if 'body' in api_response:
        api_response = api_response.get('body') or {}
    # OpenAI / OpenRouter: api_response -> choices -> message -> content
    choices = api_response.get('choices')
    if choices:
        return (choices[0].get('message') or {}).get('content') or ""
    # Claude / Gemini / Aya handlers: api_response -> content -> [{type: text, text}]
    content_list = api_response.get('content')
    if isinstance(content_list, list) and content_list:
        # Claude with extended thinking has several blocks (thinking + text)
        text_block = next((b for b in content_list if isinstance(b, dict) and b.get('type') == 'text'), None)
        if text_block is None and isinstance(content_list[0], dict) and 'text' in content_list[0]:
            text_block = content_list[0]
        if text_block:
            return text_block.get('text', '')
    return ""


def load_and_process_json(file_path, model_name):
    """Load a result file and score every item"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error decoding JSON: {file_path}")
        return []

    results = []
    for item in data:
        try:
            ground_truth = str(item.get('ground_truth', '')).strip()
            text = get_response_text(item.get('api_response', {}))
            extracted_answer = extract_answer_from_text(text, item.get('multiple_question'), model_name)
            results.append({
                'ID': item.get('ID', ''),
                'subject': item.get('subject', ''),
                'ground_truth': ground_truth,
                'extracted_answer': extracted_answer,
                'has_extraction': extracted_answer is not None,
                'is_correct': extracted_answer is not None and normalize_answer(extracted_answer) == normalize_answer(ground_truth)
            })
        except Exception as e:
            print(f"Error processing item in {model_name} (ID: {item.get('ID')}): {e}")
    return results


def calculate_rates_by_subject(results):
    """Calculate extraction and accuracy rates by subject"""
    subject_stats = defaultdict(lambda: {'total': 0, 'extracted': 0, 'correct': 0})

    for result in results:
        subject = result['subject']
        subject_stats[subject]['total'] += 1
        if result['has_extraction']:
            subject_stats[subject]['extracted'] += 1
        if result['is_correct']:
            subject_stats[subject]['correct'] += 1

    rates = {}
    for subject, stats in subject_stats.items():
        rates[subject] = {
            'extraction_rate': stats['extracted'] / stats['total'] if stats['total'] > 0 else 0,
            'accuracy_rate': stats['correct'] / stats['total'] if stats['total'] > 0 else 0,
            'total_questions': stats['total']
        }
    return rates


def order_subjects_custom(all_subjects_set):
    """Order subjects according to specified order"""
    desired_order = ['math', 'physics', 'chemistry', 'bio', 'geography', 'driving test', 'IQ Test']
    subject_mapping = {}
    for subject in all_subjects_set:
        subject_lower = subject.lower()
        for desired in desired_order:
            if desired.lower() in subject_lower or subject_lower in desired.lower():
                subject_mapping[subject] = desired
                break
        if subject not in subject_mapping:
            subject_mapping[subject] = subject

    ordered_subjects = []
    for desired in desired_order:
        for original_subject in all_subjects_set:
            if subject_mapping.get(original_subject) == desired and original_subject not in ordered_subjects:
                ordered_subjects.append(original_subject)

    for subject in sorted(all_subjects_set):
        if subject not in ordered_subjects:
            ordered_subjects.append(subject)

    return ordered_subjects


def discover_result_files(results_dir):
    """{result name (e.g. full_vqa_vn): {model: file path}}, newest file per model."""
    groups = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(results_dir, '**', '*.json'), recursive=True)):
        model = os.path.relpath(os.path.dirname(path), results_dir)
        if model == '.':
            continue
        name = re.sub(r'_\d{8}_\d{6}$', '', os.path.splitext(os.path.basename(path))[0])
        if 'ocr' in name:
            continue  # OCR transcriptions are not multiple-choice answers
        previous = groups[name].get(model)
        if previous is None or os.path.getmtime(path) > os.path.getmtime(previous):
            groups[name][model] = path
    return groups


def create_table_data(model_list, all_model_results, all_subjects, rate_type):
    """Create table data for given models and rate type"""
    table_data = []
    for model in model_list:
        row = {'Model': model}
        rates = []
        for subject in all_subjects:
            rate = all_model_results[model].get(subject, {}).get(rate_type, 0)
            row[subject] = round(rate * 100, 2)
            rates.append(rate)
        row['Mean'] = round(np.mean(rates) * 100, 2) if rates else 0
        table_data.append(row)

    if table_data:
        mean_row = {'Model': 'Mean'}
        for subject in all_subjects:
            subject_rates = [all_model_results[model].get(subject, {}).get(rate_type, 0) for model in model_list]
            mean_row[subject] = round(np.mean(subject_rates) * 100, 2) if subject_rates else 0
        all_rates = [all_model_results[model].get(s, {}).get(rate_type, 0) for model in model_list for s in all_subjects]
        mean_row['Mean'] = round(np.mean(all_rates) * 100, 2) if all_rates else 0
        table_data.append(mean_row)
    return table_data


def report(name, model_files, results_dir):
    print(f"\n{'=' * 80}\n{name}\n{'=' * 80}")
    all_model_results = {}
    all_subjects = set()
    for model, file_path in model_files.items():
        results = load_and_process_json(file_path, model)
        if not results:
            print(f"  No results found for {model}")
            continue
        rates = calculate_rates_by_subject(results)
        all_model_results[model] = rates
        all_subjects.update(rates.keys())
        print(f"  {model}: {len(results)} questions ({os.path.relpath(file_path, results_dir)})")

    if not all_model_results:
        return
    all_subjects = order_subjects_custom(all_subjects)

    commercial_models = [m for m in all_model_results if COMMERCIAL_MODEL_PATTERN.match(os.path.basename(m))]
    open_source_models = [m for m in all_model_results if m not in commercial_models]

    sections = [
        ('EXTRACTION RATE - COMMERCIAL MODELS', commercial_models, 'extraction_rate'),
        ('EXTRACTION RATE - OPEN SOURCE MODELS', open_source_models, 'extraction_rate'),
        ('ACCURACY RATE - COMMERCIAL MODELS', commercial_models, 'accuracy_rate'),
        ('ACCURACY RATE - OPEN SOURCE MODELS', open_source_models, 'accuracy_rate'),
    ]
    columns = ['Model'] + all_subjects + ['Mean']
    combined_parts = []
    for title, models, rate_type in sections:
        if not models:
            continue
        df = pd.DataFrame(create_table_data(models, all_model_results, all_subjects, rate_type), columns=columns)
        print(f"\n{title}:")
        print(df.to_string(index=False))
        combined_parts += [pd.DataFrame([[title] + [''] * (len(columns) - 1)], columns=columns), df,
                           pd.DataFrame([[''] * len(columns)], columns=columns)]

    csv_path = os.path.join(results_dir, f"analysis_{name}.csv")
    pd.concat(combined_parts, ignore_index=True).to_csv(csv_path, index=False)
    print(f"\nFile saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Extraction-rate and accuracy tables for VMMU results.")
    parser.add_argument("--results-dir", default=os.path.join(REPO_ROOT, "results"), help="Directory with <model>/<dataset>_<lang>.json result files.")
    parser.add_argument("--file", default=None, help="Only report this result name, e.g. full_vqa_vn.")
    args = parser.parse_args()

    groups = discover_result_files(args.results_dir)
    if args.file:
        groups = {name: files for name, files in groups.items() if name == args.file}
    if not groups:
        print(f"No result files found in {args.results_dir}")
        return
    for name in sorted(groups):
        report(name, groups[name], args.results_dir)


if __name__ == "__main__":
    main()
