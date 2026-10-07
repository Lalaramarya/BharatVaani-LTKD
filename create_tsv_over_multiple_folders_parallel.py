import argparse
import logging
import os
import re
import csv
import gc
import traceback
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import torch
import torch.multiprocessing as mp
import torchaudio

from seamless_communication.datasets.huggingface import SpeechTokenizer
from seamless_communication.models.unit_extractor import UnitExtractor


# -----------------------------------------------------------------------------
# WHAT CHANGED FROM THE OLD SCRIPT, AND WHY
# -----------------------------------------------------------------------------
# 1. The old build_tsv() matched {episode}_{suffix}.txt lines to manifest.json
#    entries by fuzzily CANONICALIZING both sides' TEXT (Unicode normalize,
#    strip punctuation/combining marks, case-fold) and joining on that lossy
#    key. That was necessary because the old plain .txt files had no id at
#    all -- text was the only thing to join on, and text can collide, drift
#    under normalization, or just fail to match for reasons that have nothing
#    to do with whether the sentence is actually the same one.
#
#    The four TSVs this now reads (en_aligned, all_aligned, en_all_candidates,
#    all_all_candidates) already carry a "sentence_id" for every row -- and
#    align_and_segment.py (already updated) now writes that SAME sentence_id
#    into manifest.json as an "id" field per segment. So the join is now an
#    EXACT id == id merge -- no canonicalization, no ambiguity, no risk of
#    two different sentences silently matching each other.
#
# 2. The old generate_km1000() wrote a SEPARATE "*.km1000" file containing
#    just the audio path + generated units. This version adds "path" (the
#    matched audio_filepath) and the generated units/sample_rate as new
#    COLUMNS directly onto the same TSV, so everything for a sentence --
#    its id, text, aligned/candidate metadata that was already in the TSV,
#    its audio path, and its discrete units -- lives in one file.
# -----------------------------------------------------------------------------


# -----------------------------
# Logging and Argument Parsing
# -----------------------------

def get_logger():
    log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def get_parser():
    parser = argparse.ArgumentParser(
        description="Full Multi-GPU TSV enrichment (path + units) generator -- id-based join"
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Root directory of all languages"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        required=True,
        help="Comma-separated GPU IDs. Example: '0,1,3'"
    )
    parser.add_argument(
        "--workers_per_gpu",
        type=int,
        default=1,
        help="Workers to spawn per selected GPU"
    )
    parser.add_argument(
        "--max_chunk_seconds",
        type=float,
        default=30.0,
        help="Max audio seconds per chunk to avoid OOM"
    )
    parser.add_argument(
        "--target_sample_rate",
        type=int,
        default=16000,
        help="Resample audio to this rate"
    )
    parser.add_argument(
        "--suffixes",
        type=str,
        default="en_aligned,all_aligned,en_all_candidates,all_all_candidates",
        help="Comma-separated list of TSV suffixes to process, e.g. "
             "'{episode}_<suffix>.tsv'. Default covers all four the "
             "alignment pipeline produces."
    )
    return parser


# -----------------------------
# Utility Functions
# -----------------------------

def extract_episode_number(folder_name: str):
    m = re.search(r"MKB_(\d+)", folder_name)
    return int(m.group(1)) if m else float("inf")


# -----------------------------
# Tokenizer Wrapper (unchanged from the old script)
# -----------------------------

class UnitSpeechTokenizer(SpeechTokenizer):
    MODEL_NAME = "xlsr2_1b_v2"
    KMEANS_MODEL_URI = (
        "https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.npy"
    )
    OUTPUT_LAYER_IDX = 34

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.unit_extractor = UnitExtractor(
            model_name_or_card=self.MODEL_NAME,
            kmeans_uri=self.KMEANS_MODEL_URI,
            device=self.device,
        )

    def encode(self, wav, sr):
        return self.unit_extractor.predict(
            wav.to(self.device),
            out_layer_idx=self.OUTPUT_LAYER_IDX,
            sample_rate=sr,
        )


# -----------------------------
# Audio Helpers (unchanged from the old script)
# -----------------------------

def wav_to_unit(path):
    waveform, sr = torchaudio.load(path)
    return waveform, sr


def preprocess_waveform(wav, sr, target_sr):
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav, sr


def encode_chunked(tokenizer, wav, sr, max_chunk_secs):
    with torch.no_grad():
        n = wav.size(-1)
        if max_chunk_secs <= 0:
            return tokenizer.encode(wav, sr).reshape(-1)

        chunk_size = int(sr * max_chunk_secs)
        units_all = []
        pos = 0

        while pos < n:
            end = min(pos + chunk_size, n)
            chunk = wav[:, pos:end]

            u = tokenizer.encode(chunk, sr).reshape(-1).cpu()
            units_all.append(u)

            del chunk, u
            torch.cuda.empty_cache(); gc.collect()

            pos = end

        return torch.cat(units_all, dim=0)


# -----------------------------
# TSV enrichment: id-based join + in-place path/units columns
# -----------------------------

def enrich_tsv(language_path, episode, language, suffix, tokenizer, max_chunk, target_sr, tag):
    """
    Reads {episode}_{suffix}.tsv (2 columns, no header: sentence_id, text --
    the format save_list_column_to_lines() in the alignment pipeline writes),
    joins it EXACTLY on sentence_id against manifest.json's "id" field, adds
    "path" (audio_filepath) and generated unit columns directly onto the
    same dataframe, and re-saves it to the SAME tsv path.

    Returns the enriched dataframe, or None if the input tsv or manifest is
    missing/empty.
    """
    episode_path = os.path.join(language_path, episode)

    tsv_path = os.path.join(episode_path, f"{episode}_{suffix}.tsv")
    if not os.path.exists(tsv_path):
        tqdm.write(f"{tag} Missing tsv: {tsv_path}")
        return None

    # Same merged-manifest location force_aligned_over_folders_parallel.py
    # writes to: output_dir/language/episode/episode/manifest.json
    manifest_path = os.path.join(episode_path, episode, "manifest.json")
    if not os.path.exists(manifest_path):
        tqdm.write(f"{tag} Missing manifest: {manifest_path}")
        return None

    # --------------------------------------------------
    # Load TSV (no header -- sentence_id, text)
    # --------------------------------------------------
    try:
        df_lang = pd.read_csv(tsv_path, sep="\t", header=None, names=["sentence_id", "text"], dtype=str,
                              quoting=csv.QUOTE_NONE)
    except Exception as e:
        tqdm.write(f"{tag} ERROR reading tsv {tsv_path}: {e}")
        return None

    if df_lang.empty:
        tqdm.write(f"{tag} EMPTY tsv: {tsv_path}")
        return None

    # --------------------------------------------------
    # Load manifest (has "id" == sentence_id, per align_and_segment.py)
    # --------------------------------------------------
    df_json = pd.read_json(manifest_path, lines=True)
    if "id" not in df_json.columns:
        tqdm.write(
            f"{tag} manifest.json has no \"id\" field -- was it produced by "
            f"the updated align_and_segment.py? Falling back is not possible "
            f"for an exact id join, skipping {tsv_path}."
        )
        return None

    # --------------------------------------------------
    # EXACT id join -- no text canonicalization needed at all
    # --------------------------------------------------
    merged = df_lang.merge(
        df_json[["id", "audio_filepath", "normalized_text", "duration", "audio_start_sec"]],
        left_on="sentence_id",
        right_on="id",
        how="left"
    ).drop(columns=["id"])

    unmatched = merged["audio_filepath"].isna().sum()
    if unmatched:
        tqdm.write(f"{tag} {unmatched}/{len(merged)} rows had no manifest match for {os.path.basename(tsv_path)}")

    merged = merged.rename(columns={"audio_filepath": "path"})

    # --------------------------------------------------
    # Generate units for every row that DID get a matched path
    # --------------------------------------------------
    units_col = []
    sr_col = []
    for _, row in tqdm(merged.iterrows(), total=len(merged), desc=f"{tag} units"):
        wav_path = row["path"]
        if not isinstance(wav_path, str) or not os.path.exists(wav_path):
            units_col.append(None)
            sr_col.append(None)
            continue
        try:
            wav, sr = wav_to_unit(wav_path)
            wav, sr = preprocess_waveform(wav, sr, target_sr)
            units = encode_chunked(tokenizer, wav, sr, max_chunk)
            units_col.append(units.tolist())
            sr_col.append(sr)
        except Exception as e:
            tqdm.write(f"{tag} Error generating units for {wav_path}: {e}")
            traceback.print_exc()
            units_col.append(None)
            sr_col.append(None)
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    merged[f"{language}_Units"] = units_col
    merged[f"{language}_Sample_rate"] = sr_col

    # --------------------------------------------------
    # Save back into the SAME tsv -- path + units now live alongside
    # whatever alignment metadata (aligned/stage/similarity, if this is an
    # _all_pairs-derived suffix) the file already had.
    # --------------------------------------------------
    merged.to_csv(tsv_path, sep="\t", index=False)
    tqdm.write(f"{tag} Updated {tsv_path} (+path, +units)")
    return merged


# -----------------------------
# Worker Process
# -----------------------------

def worker_main(worker_id, cuda_visible, assigned_jobs, max_chunk, target_sr, suffixes):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible  # isolate GPU

    gpu_name_visible = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    tag = f"[Worker {worker_id} | GPU {cuda_visible} ({gpu_name_visible})]"

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    torch.set_grad_enabled(False)
    tokenizer = UnitSpeechTokenizer(device=device)

    for job in assigned_jobs:
        language = job["language"]
        language_path = job["language_path"]
        episode = job["episode"]

        ep_tag = f"{tag} [{language}:{episode}]"
        tqdm.write(f"{ep_tag} START")

        for suffix in suffixes:
            enrich_tsv(language_path, episode, language, suffix, tokenizer, max_chunk, target_sr, ep_tag)

        tqdm.write(f"{ep_tag} DONE")

        torch.cuda.empty_cache()
        gc.collect()

    tqdm.write(f"{tag} FINISHED ALL JOBS")


# -----------------------------
# Coordinator (main)
# -----------------------------

def main(args, logger):
    gpu_ids = [g.strip() for g in args.gpus.split(",")]
    logger.info(f"Using GPUs: {gpu_ids}")

    suffixes = [s.strip() for s in args.suffixes.split(",") if s.strip()]
    logger.info(f"Processing suffixes: {suffixes}")

    workers_per_gpu = args.workers_per_gpu
    total_workers = len(gpu_ids) * workers_per_gpu
    logger.info(f"Workers per GPU: {workers_per_gpu} | Total workers: {total_workers}")

    jobs = []
    for language in sorted(os.listdir(args.input_dir)):
        language_path = os.path.join(args.input_dir, language)
        if not os.path.isdir(language_path):
            continue

        episodes = sorted(
            os.listdir(language_path),
            key=extract_episode_number,
            reverse=True,
        )
        for ep in episodes:
            ep_path = os.path.join(language_path, ep)
            if os.path.isdir(ep_path):
                jobs.append({
                    "language": language,
                    "language_path": language_path,
                    "episode": ep,
                })

    logger.info(f"Total EPISODES to process: {len(jobs)}")

    job_chunks = [[] for _ in range(total_workers)]
    for i, job in enumerate(jobs):
        job_chunks[i % total_workers].append(job)

    processes = []
    worker_id = 0
    for gpu in gpu_ids:
        for _ in range(workers_per_gpu):
            assigned_jobs = job_chunks[worker_id]

            if not assigned_jobs:
                worker_id += 1
                continue

            p = mp.Process(
                target=worker_main,
                args=(
                    worker_id,
                    gpu,
                    assigned_jobs,
                    args.max_chunk_seconds,
                    args.target_sample_rate,
                    suffixes,
                ),
            )
            p.start()
            processes.append(p)
            worker_id += 1

    for p in processes:
        p.join()

    logger.info("PROCESSING COMPLETE")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = get_parser()
    args = parser.parse_args()
    logger = get_logger()
    main(args, logger)
