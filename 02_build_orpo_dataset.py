import os
import sys
import json
import glob
import random
import time
from openai import OpenAI
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
# Use a cheap API to generate the dataset automatically.
# GPT-4o-mini is extremely cheap ($0.15 per 1M input tokens).
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("[ERROR] OPENAI_API_KEY environment variable is not set.")
    print("  Run: export OPENAI_API_KEY='your-key-here'")
    sys.exit(1)
client = OpenAI(api_key=api_key)

RAW_NOVELS_DIR = "raw_novels"
OUTPUT_JSONL = "orpo_dataset.jsonl"
WORDS_PER_CHUNK = 1000 # Roughly a scene/chapter
TARGET_SAMPLES = 2000  # Number of pairs to generate

def chunk_text(text, word_count):
    """
    Split text into chunks at paragraph boundaries (double newlines),
    grouping paragraphs until we reach the target word count.
    This avoids cutting mid-sentence and keeps scenes coherent.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_words = []
    current_count = 0

    for para in paragraphs:
        para_words = para.split()
        if current_count + len(para_words) > word_count and current_count > 0:
            chunks.append(" ".join(current_words))
            current_words = para_words
            current_count = len(para_words)
        else:
            current_words.extend(para_words)
            current_count += len(para_words)

    if current_words:
        chunks.append(" ".join(current_words))
    return chunks

def generate_orpo_pair(human_text, max_retries=4):
    """Uses a cheap API to reverse-engineer an outline, then generate 'bad' AI prose.
    Includes exponential backoff on API errors.
    """
    for attempt in range(1, max_retries + 1):
        try:
            # Step 1: Reverse Engineer the Outline
            outline_res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert literary analyst."},
                    {"role": "user", "content": f"Extract a concise, bulleted plot outline and a brief list of character emotional states from the following text. Do not include any prose.\n\nTEXT:\n{human_text}"}
                ],
                max_tokens=300
            )
            outline = outline_res.choices[0].message.content

            # Step 2: Generate \"Bad\" AI Slop from that Outline
            bad_ai_res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a generic, robotic AI assistant trying to write a novel. Use clichés, 'purple prose', and words like 'delve', 'testament', and 'tapestry'."},
                    {"role": "user", "content": f"Write a novel chapter based exactly on this outline:\n\n{outline}"}
                ],
                max_tokens=1500
            )
            bad_prose = bad_ai_res.choices[0].message.content

            return outline, bad_prose

        except Exception as e:
            wait = 2 ** attempt  # Exponential backoff: 2, 4, 8, 16 seconds
            print(f"  [API Error] Attempt {attempt}/{max_retries}: {e}. Retrying in {wait}s...")
            time.sleep(wait)

    print("  [FAIL] Skipping this chunk after all retries.")
    return None, None

# 1. Load all human text
all_chunks = []
for file in glob.glob(f"{RAW_NOVELS_DIR}/*.txt"):
    with open(file, 'r', encoding='utf-8') as f:
        chunks = chunk_text(f.read(), WORDS_PER_CHUNK)
        # Filter out chunks that are too short
        all_chunks.extend([c for c in chunks if len(c.split()) > WORDS_PER_CHUNK * 0.8])

# 2. Randomly sample to reach target
random.shuffle(all_chunks)
selected_chunks = all_chunks[:TARGET_SAMPLES]

# 3. Generate dataset
print(f"Generating {len(selected_chunks)} ORPO pairs via API...")
with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for chunk in tqdm(selected_chunks):
        outline, bad_prose = generate_orpo_pair(chunk)
        if outline and bad_prose:
            record = {
                "prompt": f"Write a chapter based on this outline:\n\n{outline}\n\nChapter Prose:\n",
                "chosen": chunk,         # The beautiful human writing
                "rejected": bad_prose    # The generic AI writing
            }
            f.write(json.dumps(record) + "\n")

print(f"Dataset saved to {OUTPUT_JSONL}")
