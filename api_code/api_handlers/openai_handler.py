# api_handlers/openai_handler.py
import base64
import time
import threading
from openai import OpenAI

thread_local = {}

API_KEY_ENV = "OPENAI_API_KEY"

def get_api_key_path():
    """Returns the default path for the OpenAI API key."""
    return "api_key/openai_key.txt"

def get_client(api_key):
    """Creates or retrieves the OpenAI client for the current thread."""
    thread_id = str(threading.get_ident())
    if thread_id not in thread_local:
        thread_local[thread_id] = OpenAI(api_key=api_key)
    return thread_local[thread_id]

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def make_api_call(api_key, model, prompt, image_path=None, temperature=0, reasoning_effort='low'):
    client = get_client(api_key)

    # Build content based on whether image_path is provided
    content = [{"type": "text", "text": prompt}]

    if image_path:
        base64_image = encode_image_to_base64(image_path)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })

    for attempt in range(3):
        try:
            # Models that support reasoning_effort: o3 series and gpt-5.x
            if 'o3' in model.lower() or 'gpt-5' in model.lower():
                response = client.chat.completions.create(
                    model=model,
                    reasoning_effort=reasoning_effort,
                    messages=[
                        {
                            "role": "user",
                            "content": content
                        }
                    ]
                )
            else:
                # Standard models (GPT-4, GPT-4.1, etc.) use temperature
                response = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=[
                        {
                            "role": "user",
                            "content": content
                        }
                    ]
                )
            return response.model_dump()
        except Exception as e:
            error_location = image_path if image_path else "text-only request"
            print(f"Attempt {attempt+1} failed for {error_location}: {e}")
            if attempt < 2:
                print(f"Retrying in 30 seconds...")
                time.sleep(30)
            else:
                raise

def process_item(item, api_key, model, temperature, prompt_lang, reasoning_effort='low'):
    item_id = item['ID']
    image_path = item.get('image_path')  # Use .get() to allow None

    prompt_key = f"{prompt_lang}_prompt"
    if prompt_key not in item:
        print(f"ERROR: Item {item_id} is missing '{prompt_key}'. Skipping.")
        return None

    prompt_text = item[prompt_key]

    try:
        response_dict = make_api_call(api_key, model, prompt_text, image_path, temperature, reasoning_effort)
        item_with_response = item.copy()
        item_with_response['api_response'] = response_dict
        print(f"Successfully processed {item_id}")
        return item_with_response
    except Exception as e:
        print(f"Failed to process {item_id}: {e}")
        return None
