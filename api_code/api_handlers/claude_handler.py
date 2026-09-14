# api_handlers/claude_handler.py
import base64
import time
import os
from pathlib import Path
from anthropic import Anthropic
import threading
from PIL import Image
import io

thread_local = {}

API_KEY_ENV = "ANTHROPIC_API_KEY"

def get_api_key_path():
    """Returns the default path for the Claude API key."""
    return "api_key/claude_key.txt"

def get_client(api_key):
    """Creates or retrieves the Anthropic client for the current thread."""
    thread_id = str(threading.get_ident())
    if thread_id not in thread_local:
        thread_local[thread_id] = Anthropic(api_key=api_key)
    return thread_local[thread_id]

def resize_image_if_needed(image_path, max_size_mb=3.7):
    """Resize image if it exceeds max_size_mb. Returns base64 encoded string.
    Note: max_size_mb accounts for base64 encoding overhead (~33% increase).
    Claude API limit is 5MB for base64, so we use 3.7MB raw to stay under 5MB encoded.
    """
    file_size_mb = os.path.getsize(image_path) / (1024 * 1024)

    # Always process through PIL to ensure proper compression
    img = Image.open(image_path)
    buffer = io.BytesIO()

    if file_size_mb <= max_size_mb:
        # Image is within limit, but still compress it
        img_format = img.format or 'PNG'
        if img_format == 'PNG':
            img.save(buffer, format='PNG', optimize=True)
        elif img_format in ['JPEG', 'JPG']:
            img.save(buffer, format='JPEG', quality=90, optimize=True)
        else:
            img.save(buffer, format=img_format)

        buffer.seek(0)
        compressed_size_mb = len(buffer.getvalue()) / (1024 * 1024)

        # Check if compression alone is enough
        if compressed_size_mb <= max_size_mb:
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

        # If still too large after compression, need to resize
        file_size_mb = compressed_size_mb
        buffer = io.BytesIO()

    # Image is too large, resize it
    print(f"  → Image {os.path.basename(image_path)} is {file_size_mb:.1f}MB, resizing...")

    # Calculate new dimensions (reduce by scale factor)
    # Use 0.85 safety margin to account for base64 overhead
    scale_factor = (max_size_mb * 0.85 / file_size_mb) ** 0.5
    new_width = int(img.width * scale_factor)
    new_height = int(img.height * scale_factor)

    # Resize image
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Save to bytes with compression
    img_format = img.format or 'PNG'
    if img_format == 'PNG':
        img_resized.save(buffer, format='PNG', optimize=True)
    elif img_format in ['JPEG', 'JPG']:
        img_resized.save(buffer, format='JPEG', quality=85, optimize=True)
    else:
        img_resized.save(buffer, format=img_format)

    buffer.seek(0)
    resized_size_mb = len(buffer.getvalue()) / (1024 * 1024)
    print(f"  → Resized to {new_width}x{new_height} ({resized_size_mb:.1f}MB raw, ~{resized_size_mb*1.33:.1f}MB base64)")

    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def encode_image_to_base64(image_path):
    return resize_image_if_needed(image_path, max_size_mb=3.7)

def get_image_media_type(image_path):
    ext = Path(image_path).suffix.lower()
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    elif ext == ".png":
        return "image/png"
    elif ext == ".webp":
        return "image/webp"
    elif ext == ".gif":
        return "image/gif"
    else:
        print(f"Warning: Unknown image type for {image_path}, defaulting to image/jpeg.")
        return "image/jpeg"

REASONING_EFFORT_TO_BUDGET = {
    "low": 2000,
    "medium": 10000,
    "high": 20000,
}

def make_api_call(api_key, model, prompt, image_path=None, temperature=0, reasoning_effort=None):
    client = get_client(api_key)

    # Build content based on whether image_path is provided
    content = []

    if image_path:
        base64_image = encode_image_to_base64(image_path)
        media_type = get_image_media_type(image_path)
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": base64_image,
            },
        })

    content.append({"type": "text", "text": prompt})

    use_thinking = reasoning_effort is not None
    budget_tokens = REASONING_EFFORT_TO_BUDGET.get(reasoning_effort, 10000) if use_thinking else None

    for attempt in range(3):
        try:
            kwargs = dict(
                model=model,
                max_tokens=(budget_tokens + 4096) if use_thinking else 2048,
                messages=[{"role": "user", "content": content}]
            )
            if use_thinking:
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget_tokens}
                # Extended thinking requires temperature=1
                kwargs["temperature"] = 1
            else:
                kwargs["temperature"] = temperature

            response = client.messages.create(**kwargs)
            return response.model_dump()
        except Exception as e:
            error_location = image_path if image_path else "text-only request"
            print(f"Attempt {attempt+1} failed for {error_location}: {e}")
            if attempt < 2:
                print(f"Retrying in 30 seconds...")
                time.sleep(30)
            else:
                raise

def process_item(item, api_key, model, temperature, prompt_lang, reasoning_effort=None):
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
