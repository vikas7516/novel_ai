# Novel AI: Writer's Soul Workflow

Fine-tunes a 3B parameter model (Zamba2-2.7B) to write high-quality fiction
without generic "AI slop" tone. Uses two stages: CPT style soak → ORPO alignment.

---

## 0. Setup

```bash
pip install -r requirements.txt
playwright install chromium   # For the interactive web scraper
```

---

## 1. Scrape Novels (Interactive)

Run the interactive scraper and paste novelfire.net URLs when prompted.
Each novel is saved as a separate `.txt` file in `raw_novels/`.

```bash
python 01_scrape_novels.py
```

A browser window will open automatically to handle Cloudflare. Paste one or
more novel URLs (one per line), press ENTER twice, and watch it go.

**You need ~800–1000 novels (250M tokens) for a meaningful CPT dataset.**
Progress is auto-saved in `scrape_progress.json` — re-run to resume after crashes.

---

## 2. Build ORPO Dataset (Good vs. Bad Prose Pairs)

Requires an OpenAI API key. Costs ~$1–2 for 2000 pairs using gpt-4o-mini.

```bash
export OPENAI_API_KEY="your-key-here"
python 02_build_orpo_dataset.py
```

Outputs `orpo_dataset.jsonl` with `{prompt, chosen, rejected}` pairs.
- `chosen` = authentic human prose (from your scraped novels)
- `rejected` = generic AI-generated prose from the same outline

---

## 3. Stage 1: Continuous Pre-Training (T4 — ~50 hours)

Upload `raw_novels/` and the scripts to Lightning AI (T4 instance).

```bash
python 03_stage1_cpt.py
```

Outputs LoRA adapter to `./novel_model_stage1_final/`.

---

## 3b. Merge Stage 1 Adapter ← DO NOT SKIP

Merge the Stage 1 adapter into the base model weights.
**If you skip this, Stage 2 will train on the original base model and ignore all Stage 1 work.**

```bash
python 00_merge_stage1.py
```

Outputs `./novel_model_stage1_merged/` — used as input for Stage 2.

---

## 4. Stage 2: ORPO Alignment (H100 — ~3 hours)

Switch to an H100 instance on Lightning AI.

```bash
python 04_stage2_orpo.py
```

Outputs final weights to `./novel_model_final_weights/`.

---

## 5. Local Inference (RTX 2050 / 4GB VRAM)

Download `./novel_model_final_weights/` to your laptop.
Quantize with llama.cpp or LM Studio (4-bit GGUF) for ~30 tok/s on your GPU.

---

## File Structure

```
novel_ai/
├── 00_merge_stage1.py       # Bridge: merge Stage 1 adapter → base model
├── 01_scrape_novels.py      # Interactive novelfire.net scraper (Playwright)
├── 02_build_orpo_dataset.py # Generate good/bad prose pairs via GPT-4o-mini
├── 03_stage1_cpt.py         # Stage 1: Continuous Pre-Training (T4)
├── 04_stage2_orpo.py        # Stage 2: ORPO Alignment (H100)
├── raw_novels/              # One .txt file per novel (auto-created)
├── scrape_progress.json     # Resume state for the scraper
└── orpo_dataset.jsonl       # Training pairs for Stage 2
```
