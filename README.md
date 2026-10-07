# BharatVaani: A Multilingual Indic Corpus and Latent Text-Guided Knowledge Distillation for Direct Speech-to-Speech Translation

The main contributions of this paper are as follows:

- We release [BharatVaani](https://huggingface.co/datasets/latentvector/BharatVaani), a multilingual parallel speech corpus comprising approximately **732 hours** of aligned speech across **35 Indic languages**.

<img width="753" height="660" alt="final-1new_mkbalignment1" src="https://github.com/user-attachments/assets/67d1c4a3-15c0-4b34-bf99-f8a7081af413" />

- We introduce a scalable pipeline for constructing multilingual parallel speech corpora and propose an **Adaptive Sliding Window Alignment Strategy** for accurate cross-lingual speech alignment.
  
<img width="659" height="905" alt="1final-new-sliding_window_v6" src="https://github.com/user-attachments/assets/96138a28-1946-4cee-8d99-aeee370dbe8c" />

- We propose **Latent Text-Guided Knowledge Distillation (LTKD)**, a framework for fully textless **direct speech-to-speech translation (DS2ST)** that distills latent linguistic knowledge from a multilingual teacher into an **S2UT** student.

<img width="1312" height="422" alt="final-ltkd_framework_v7" src="https://github.com/user-attachments/assets/1c81da23-22d0-4a81-be41-d0f3de222283" />

- We conduct a large-scale **DS2ST evaluation across 34 Indic languages to English**, including **20 languages not represented in benchmarks such as FLEURS**, showing that multilingual LTKD narrows the gap with text-supervised baselines and generalizes to unseen languages.

---

## Table of Contents

1. [Data Pipeline](#data-pipeline)
   - [Step 1 — Alignment](#step-1--alignment)
   - [Step 2 — MMS Forced Alignment](#step-2--mms-forced-alignment)
   - [Step 3 — TSV with Audio Units](#step-3--tsvs-with-audio-units)
   - [Step 4 — Merge Across Episodes](#step-4--merge-across-episodes)
   - [Step 5 — Build Train/Valid/Test Manifests](#step-5--build-trainvalidtest-manifests)
2. [Training (LTKD / S2UT)](#training-ltkd--s2ut)
3. [Pretrained Checkpoints](#pretrained-checkpoints)

---

## Data Pipeline

### Step 1 — Alignment

Run the alignment pipeline. This produces per-episode outputs:
- **Excel**: `all_pairs_before_threshold`, `aligned`, `sliding_aligned`, `topk_only`, `both_candidates`
- **TSV**: `en_aligned`, `all_aligned`, `en_all_candidates`, `all_all_candidates`
- **Text + ID sidecars**: `split_sentences`, `aligned_audio`, `unaligned_audio`, `full_paragraph`

```bash
python align_pipeline_gpu_parallel.py \
    --text_dir /path/to/Maan_ki_Baat_all_36_languages \
    --output_dir /path/to/aligned_output \
    --debug \
    --threshold 0.65 \
    --top_k 1 \
    --max_usage 1 \
    --run_mode parallel \
    --n_gpus 1 \
    --workers_per_gpu 8 \
    --save_all_candidates
```

---

### Step 2 — MMS Forced Alignment

> **Prerequisites — set up the MMS forced-alignment environment first.**
>
> Clone and configure the environment as described in the `mms_forced_alignment` README of the following repo:
>
> ```bash
> git clone https://github.com/karynaur/MMS-forced-align.git
> cd MMS-forced-align
> # Follow the README inside this repo to create the conda/pip environment
> # and download the required MMS model weights before running the step below.
> ```

Once the environment is active, run forced alignment. This splits audio using the `split_sentences` + `full_paragraph` sidecars from Step 1 and produces a `manifest.json` (with `sentence_id`) per episode:

```bash
python force_aligned_over_folders_parallel.py \
    --base_dir /path/to/aligned_output \
    --uroman_path /path/to/uroman/bin \
    --output_dir /path/to/forced_align_output \
    --num_workers 4 \
    --gpu_ids 1
```

---

### Step 3 — TSVs with Audio Units

Enrich each of the four per-episode TSVs with audio path + discrete units (in-place, joined on `sentence_id` against `manifest.json`):

```bash
python create_tsv_over_multiple_folders_parallel.py \
    --input_dir /path/to/aligned_output \
    --gpus 0 \
    --workers_per_gpu 1 \
    --suffixes en_aligned,all_aligned,en_all_candidates,all_all_candidates
```

---

### Step 4 — Merge Across Episodes

Merge across episodes, per language. The TSV side already has paths + units embedded; the Excel/JSON side contains the diagnostic + combined wide files:

```bash
python merge_tsv_along_different_folders_v2.py \
    --input_dir /path/to/aligned_output \
    --output_dir /path/to/merged_output \
    --also_save_excel

python merge_excel_along_folders_v2.py \
    --input_dir /path/to/aligned_output \
    --output_dir /path/to/merged_output
```

---

### Step 5 — Build Train/Valid/Test Manifests

Build train/valid/test manifests per language pair, across all four TSV combinations (`en_aligned` / `all_aligned` / `en_all_candidates` / `all_all_candidates`):

```bash
python create_json_over_all_lang_combinations_v2.py \
    --tsv_dir /path/to/merged_output \
    --output_dir /path/to/lang_pair_combinations
```

---

## Training (LTKD / S2UT)

After the JSON manifests are prepared (Steps 5–7), we use **[Seamlesscommunication](https://github.com/facebookresearch/seamless_communication)** for training the S2UT model with LTKD.

Clone and set up the repository:

```bash
git clone https://github.com/facebookresearch/seamless_communication.git
cd seamless_communication
pip install -e .
```

Training follows the **direct S2ST with discrete units** recipe from fairseq. Refer to the step-by-step guide at:

```
fairseq/examples/speech_to_speech/docs/direct_s2st_discrete_units.md
```

This document describes:
- How to prepare the speech-to-unit (S2UT) training data
- The fairseq `speech_to_speech` task configuration
- Multi-decoder architecture setup
- How to run unit-based vocoder fine-tuning for final waveform synthesis

Point your fairseq training config at the JSON manifests produced by the pipeline above as your `--train-manifest` / `--valid-manifest` inputs.

---

## Pretrained Checkpoints

The dataset ([latentvector/BharatVaani](https://huggingface.co/datasets/latentvector/BharatVaani)) and the fine-tuned checkpoints ([latentvector/BharatVaani-s2ut](https://huggingface.co/latentvector/BharatVaani-s2ut)) are hosted separately on the Hugging Face Hub.

We release **two multilingual S2UT checkpoints**, both covering 34 Indic → English language directions, trained on different data:

| Checkpoint | Training Data | Filename |
|---|---|---|
| BharatVaani | BharatVaani (Maan Ki Baat) | `s2st_multilingual_bv.pt` |
| Bhashaanuvaad | Bhashaanuvaad ([AI4Bharat](https://ai4bharat.iitm.ac.in/)) | `s2st_multilingual_ba.pt` |

### Downloading the checkpoints

```python
from huggingface_hub import hf_hub_download

# BharatVaani-trained checkpoint
bv_ckpt = hf_hub_download(
    repo_id="latentvector/BharatVaani-s2ut",
    filename="s2st_multilingual_bv.pt",
)

# Bhashaanuvaad-trained checkpoint
ba_ckpt = hf_hub_download(
    repo_id="latentvector/BharatVaani-s2ut",
    filename="s2st_multilingual_ba.pt",
)
```

Or via the CLI:

```bash
huggingface-cli download latentvector/BharatVaani-s2ut s2st_multilingual_bv.pt \
    --local-dir /path/to/checkpoints

huggingface-cli download latentvector/BharatVaani-s2ut s2st_multilingual_ba.pt \
    --local-dir /path/to/checkpoints
```

### Using a checkpoint for inference / fine-tuning

Pass the downloaded path as the `--path` argument to fairseq:

```bash
fairseq-generate /path/to/lang_pair_combinations \
    --path /path/to/checkpoints/s2st_multilingual_bv.pt \
    --task speech_to_speech \
    ...
```

Or reference it in a SeamlessM4T/seamless_communication inference script:

```python
model = load_model("/path/to/checkpoints/s2st_multilingual_bv.pt")
```

### Uploading the checkpoints (maintainers)

```bash
# BharatVaani checkpoint
huggingface-cli upload latentvector/BharatVaani-s2ut \
    /DATA/nfsshare/Adarsh/Multilingual/New/Final_jsons/All_aligned_threshold_h055_l045/Checkpoints_new1/checkpoint_SPEECH_TO_SPEECH_loss_21.2747_lr_2.98E-08_infer.pt \
    s2st_multilingual_bv.pt \
    --repo-type model

# Bhashaanuvaad checkpoint
huggingface-cli upload latentvector/BharatVaani-s2ut \
    /DATA/nfsshare/Adarsh/SLAM/Multilingual_folders/bhasaanuvad/Seamless__Merged_JSON/deduplicated_json/Checkpoints_new1/checkpoint_SPEECH_TO_SPEECH_loss_21.5674_lr_7.07E-08_infer.pt \
    s2st_multilingual_ba.pt \
    --repo-type model
```

> **Note on large files**: Checkpoints are typically several GB. Hugging Face uses Git-LFS under the hood for files larger than 5 MB. The CLI handles this transparently; if you use the web UI, make sure Git-LFS is installed (`git lfs install`) before pushing.
