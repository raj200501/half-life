# Third-party data

Rule 9 of the build brief: the source, license, and retrieval date of every piece of third-party data this project uses, and what its terms allow. Add a section here before using any new third-party data.

## Model weights (E1 pilot candidates)

Five instruct models from five families, each at most 4B parameters, used only for local inference on this project's own synthetic coding tasks (the family pilot and the experiments in section 5 of the brief). All five files are Q4_K_M GGUF conversions published by the quantizer bartowski on Hugging Face; each GGUF repository's model card declares the same license as its upstream model.

How they are pinned and kept:

- `src/halflife/models.py` pins each file by repository, revision commit, file name, quantization, size, and SHA-256. `tools/fetch_models.py` downloads from the pinned revision (never a branch), verifies size and SHA-256, and writes `artifacts/models/pins.json` after checking every pin against the Hugging Face API (`/api/models/<repo>` and `/api/models/<repo>/tree/<revision>`). The SHA-256 of the downloaded file is authoritative.
- The weights live outside the repository in `~/.cache/halflife/models` (or `$HALFLIFE_MODELS_DIR`). `.gitignore` excludes `*.gguf`. This project never commits, uploads, or otherwise redistributes the weights.
- Retrieval: every file was downloaded on 2026-11-04 (UTC); the exact time is in the `.provenance.json` file next to each download and in `artifacts/models/pins.json`. The licenses and model metadata below were checked on 2026-11-04 from the Hugging Face API and the license files in the upstream repositories.

| Key | Upstream model | GGUF repository and revision | License |
|---|---|---|---|
| `qwen2.5-3b-instruct-q4km` | [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | `bartowski/Qwen2.5-3B-Instruct-GGUF` at `f302c64a2269a69fb27b2f9473b362f5bb8e78d8` | Qwen Research License Agreement (`qwen-research`) |
| `llama-3.2-3b-instruct-q4km` | [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | `bartowski/Llama-3.2-3B-Instruct-GGUF` at `5ab33fa94d1d04e903623ae72c95d1696f09f9e8` | Llama 3.2 Community License Agreement (`llama3.2`) |
| `phi-3.5-mini-instruct-q4km` | [microsoft/Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) | `bartowski/Phi-3.5-mini-instruct-GGUF` at `6d70da17e749a471ccb62ade694486011a75cda3` | MIT |
| `smollm2-1.7b-instruct-q4km` | [HuggingFaceTB/SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) | `bartowski/SmolLM2-1.7B-Instruct-GGUF` at `1f03464768bfcc0319fc50da8ff5fb20b6417ba2` | Apache-2.0 |
| `granite-3.3-2b-instruct-q4km` | [ibm-granite/granite-3.3-2b-instruct](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct) | `bartowski/ibm-granite_granite-3.3-2b-instruct-GGUF` at `4e06dce39d4c5fcf65ca1cc69b92962e411d38e1` | Apache-2.0 |

File names, sizes, SHA-256 digests, and parameter counts are in `src/halflife/models.py` and `artifacts/models/pins.json`.

### Terms that matter here

**Qwen2.5-3B-Instruct, Qwen Research License Agreement** (release date September 19, 2024). License file: `LICENSE` in `Qwen/Qwen2.5-3B-Instruct` at revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`, SHA-256 `ef52482bb785733093dc9a2e8edd8e764c77d12d8e9d8f10a80c9b547d32d0f9`. The license grants use, reproduction, distribution, and modification for non-commercial purposes only, defined as "for research or evaluation purposes only"; commercial use needs a separate license from Alibaba Cloud. Redistribution must include a copy of the agreement and a Notice file with the stated attribution. Anyone who uses the materials or their outputs to create, train, fine-tune, or improve an AI model that is distributed must display "Built with Qwen" or "Improved using Qwen". This project's use is research evaluation, it does not redistribute the weights, and it trains no model. This is the only candidate whose license is not permissive.

**Llama-3.2-3B-Instruct, Llama 3.2 Community License Agreement** (version release date September 25, 2024). License file: `LICENSE.txt` in `meta-llama/Llama-3.2-3B-Instruct` at revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`, SHA-256 `0b4284c1f87029e67654c7953afa16279961632cf73dcfe33374c4c2f298fa35`. The license states that using or distributing any part of the Llama materials means agreeing to it. It grants a royalty-free license to use, reproduce, distribute, and modify the materials. Use must follow applicable law and Meta's Acceptable Use Policy (incorporated by reference, published at https://www.llama.com/llama3_2/use-policy). Redistribution requires a copy of the agreement, a "Built with Llama" display, and a Notice file with the stated attribution; a model trained or improved with Llama outputs and distributed must have a name starting with "Llama". Licensees whose products had more than 700 million monthly active users on the release date need a separate license. The upstream repository is gated (manual approval); the GGUF repository is not. The `USE_POLICY.md` file in the upstream repository did not open without approval (HTTP 401), so the Acceptable Use Policy text was not retrieved for this record. The tasks here (small synthetic coding sessions in a sandbox) are benign; this project does not redistribute the weights and trains no model.

**Phi-3.5-mini-instruct, MIT License**, Copyright (c) Microsoft Corporation. License file: `LICENSE` in `microsoft/Phi-3.5-mini-instruct` at revision `2fe192450127e6a83f7441aef6e3ca586c338b77`, SHA-256 `fa8235e5b48faca34e3ca98cf4f694ef08bd216d28b58071a1f85b1d50cb814d`. The repository's `NOTICE.md` (SHA-256 `a3386cc7125ec75ed34d4f0a340c47782982095e251d322e01d8ecab22d87ba4`) lists third-party components of its code (flash-attention, BSD 3-Clause), which this project does not use.

**SmolLM2-1.7B-Instruct, Apache-2.0.** Declared in the model card metadata of `HuggingFaceTB/SmolLM2-1.7B-Instruct` (revision `31b70e2e869a7173562077fd711b654946d38674` on the check date); the repository has no separate license file.

**granite-3.3-2b-instruct, Apache-2.0.** Declared in the model card metadata of `ibm-granite/granite-3.3-2b-instruct` (revision `707f574c62054322f6b5b04b6d075f0a8f05e0f0` on the check date); the repository has no separate license file.

## Model outputs kept in the repository

Replies from these models are research data: they appear in banked run artifacts under `artifacts/`, in `artifacts/models/benchmark.json` (the first 400 characters of three replies per model to fixed synthetic prompts), and in the replay fixtures under `tests/fixtures/replay/`, which hold three calls recorded from SmolLM2-1.7B-Instruct (Apache-2.0) by `tools/bench_models.py` (the model key is in each fixture file; the raw results keep the model file name, not its local path). None of these files contains model weights. The prompts that produced them are this project's own synthetic material.

## Software used to run the models

llama-cpp-python 0.3.36 (MIT, per its package metadata), installed from the project's CPU wheel index as the optional `local` extra (`make setup-local`; pinned in `uv.lock`). It is a dependency, not data, and is listed here only because it reads the files above.
