# Table 1 Offline Data Runbook

This setup keeps the original SkillOpt config structure:

- Full runs use each dataset's original `default.yaml`.
- Smoke runs use the added `smoke.yaml` files.
- The active model provider is configured in `configs\_base_\default.yaml`.

## 1. Install

Use Python 3.11-3.13. Avoid Python 3.14 if dependency constraints reject it.

```bat
cd /d D:\Agent\RelayAgentDev\SkillOpt
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
pip install -e .
pip install pandas pyarrow pillow
```

If you use Anaconda instead of venv, run the same `pip install -e .` inside that environment.

## 2. Raw Data Layout

The offline raw data root is:

```text
D:\Agent\RelayAgentDev\benchmark
```

Expected subdirectories:

```text
benchmark\searchqa_raw\*.parquet
benchmark\livemathematicianbench_raw\qa_*_final.json
benchmark\spreadsheetbench_raw\spreadsheetbench_verified_400.tar.gz
benchmark\docvqa_raw\*.parquet
benchmark\officeqa_hf\officeqa_full.csv
benchmark\officeqa_hf\treasury_bulletins_parsed\...
```

## 3. Materialize Full Splits

```bat
cd /d D:\Agent\RelayAgentDev\SkillOpt
python scripts\materialize_table1_datasets.py --raw-root D:\Agent\RelayAgentDev\benchmark --force
```

Generated full splits:

```text
data\searchqa_split                         train=400 val=200 test=1400
data\livemathematicianbench_split           train=35  val=18  test=124
data\spreadsheetbench_split                 train=80  val=40  test=280
data\spreadsheetbench_verified_400          spreadsheet files
data\docvqa\splits                          train=107 val=53  test=374
data\docvqa_images                          extracted page images
data\officeqa_split                         train=50  val=24  test=172
data\officeqa_docs_official                 offline OfficeQA documents
```

## 4. Create Smoke Splits

```bat
python scripts\make_table1_smoke_splits.py --force
```

Generated smoke splits:

```text
data\table1_smoke\searchqa                  train=5 val=3 test=5
data\table1_smoke\livemathematicianbench    train=5 val=3 test=5
data\table1_smoke\spreadsheetbench          train=5 val=3 test=5
data\table1_smoke\docvqa                    train=5 val=3 test=5
data\table1_smoke\officeqa                  train=5 val=3 test=5
```

## 5. Configure The Model

Edit the `model` section in:

```text
configs\_base_\default.yaml
```

For the current DeepSeek-compatible run, the important fields are:

```yaml
model:
  backend: qwen_chat
  optimizer: deepseek-v4-flash
  target: deepseek-v4-flash
  optimizer_backend: qwen_chat
  target_backend: qwen_chat
  qwen_chat_base_url: https://api.deepseek.com
  qwen_chat_api_key: YOUR_PROVIDER_API_KEY
```

For another OpenAI-compatible provider, change the model names, base URL, and API key.

## 6. Smoke Test Commands

Run one first:

```bat
python scripts\train.py --config configs\searchqa\smoke.yaml
```

Then run the remaining four:

```bat
python scripts\train.py --config configs\livemathematicianbench\smoke.yaml
python scripts\train.py --config configs\spreadsheetbench\smoke.yaml
python scripts\train.py --config configs\docvqa\smoke.yaml
python scripts\train.py --config configs\officeqa\smoke.yaml
```

Smoke tests are only for pipeline validation.

## 7. Full Runs

Full runs use the original dataset configs:

```bat
python scripts\train.py --config configs\searchqa\default.yaml
python scripts\train.py --config configs\livemathematicianbench\default.yaml
python scripts\train.py --config configs\spreadsheetbench\default.yaml
python scripts\train.py --config configs\docvqa\default.yaml
python scripts\train.py --config configs\officeqa\default.yaml
```

## 8. Notes

- `data\spreadsheetbench_verified_400` is the SpreadsheetBench data root used by the original dataset config.
- DocVQA is image-based. The split and images are prepared, but model success depends on whether the selected chat provider accepts the image message format used by SkillOpt.
