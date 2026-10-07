import argparse
import logging
import os
import pandas as pd
import re
import csv
from pathlib import Path
from tqdm import tqdm


# -----------------------------------------------------------------------------
# WHAT CHANGED FROM THE OLD SCRIPT, AND WHY
# -----------------------------------------------------------------------------
# 1. Filenames updated to the four tsvs the pipeline actually produces now:
#       {episode}_en_aligned.tsv
#       {episode}_all_aligned.tsv
#       {episode}_en_all_candidates.tsv   (only if --save_all_candidates was used)
#       {episode}_all_all_candidates.tsv  (only if --save_all_candidates was used)
#    "{episode}_complete_pairs.tsv" is gone -- that was never actually the
#    real filename (the diagnostic file is an .xlsx, "_all_pairs_before_
#    threshold.xlsx"), so the old script's read of that path always hit the
#    except-and-skip branch silently.
#
# 2. Dropped the separate ".km1000" merging entirely. create_tsv_over_
#    multiple_folders_parallel_v2.py now writes "path" (audio_filepath) and
#    the generated units/sample_rate DIRECTLY as columns on the same
#    en_aligned.tsv / all_aligned.tsv / etc. -- there's no more separate
#    ".km1000" file to merge; it's already part of the tsv being merged here.
#
# 3. Fixed real bugs in the old script: merged_comp_df was being
#    concatenated as `pd.concat([merged_en_df, df], ...)` (should have been
#    merged_comp_df) -- and the same mistake for merged_comp_kf. Both meant
#    the "complete pairs" merge was silently accumulating into the WRONG
#    dataframe. Doesn't apply here directly since the tsv itself is gone,
#    but the same class of copy-paste bug is guarded against below by
#    building each merge in its own loop rather than several parallel
#    per-suffix variables in one pass.
# -----------------------------------------------------------------------------


TSV_SUFFIXES = ["en_aligned", "all_aligned", "en_all_candidates", "all_all_candidates"]


def get_logger():
    log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def get_parser():
    parser = argparse.ArgumentParser(description="Merge per-episode enriched TSVs across all episode folders, per language.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing one subfolder per language, each containing one subfolder per episode",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to write the merged TSV/Excel files to",
    )
    parser.add_argument(
        "--suffixes",
        type=str,
        default=",".join(TSV_SUFFIXES),
        help="Comma-separated list of tsv suffixes to merge, e.g. '{episode}_<suffix>.tsv'.",
    )
    parser.add_argument(
        "--also_save_excel",
        action="store_true",
        help="Also write each merged tsv out as a matching .xlsx (in addition to .tsv).",
    )
    return parser


def extract_episode_number(folder_name):
    match = re.search(r'MKB_(\d+)', folder_name)
    if match:
        return int(match.group(1))
    return float('inf')


def main(args, logger):
    languages = sorted(os.listdir(args.input_dir))
    suffixes = [s.strip() for s in args.suffixes.split(",") if s.strip()]

    os.makedirs(args.output_dir, exist_ok=True)

    for language in tqdm(languages, desc="Processing across languages"):
        language_path = os.path.join(args.input_dir, language)
        if not os.path.isdir(language_path):
            continue

        episode_folders = sorted(
            os.listdir(language_path),
            key=extract_episode_number,
            reverse=True,
        )

        # One accumulator dataframe per suffix, built independently -- no
        # shared/parallel variables to accidentally cross-wire.
        merged = {suffix: pd.DataFrame() for suffix in suffixes}

        for episode_folder in tqdm(episode_folders, desc=f"Processing {language}", leave=False):
            episode_path = os.path.join(language_path, episode_folder)
            if not os.path.isdir(episode_path):
                continue

            for suffix in suffixes:
                tsv_path = os.path.join(episode_path, f"{episode_folder}_{suffix}.tsv")
                if not os.path.exists(tsv_path):
                    continue
                try:
                    df = pd.read_csv(tsv_path, sep="\t", quoting=csv.QUOTE_NONE)
                except Exception as e:
                    tqdm.write(f"{episode_folder} ({suffix}): could not read tsv -- {e}")
                    continue
                merged[suffix] = pd.concat([merged[suffix], df], ignore_index=True)

        for suffix in suffixes:
            df = merged[suffix]
            out_tsv = os.path.join(args.output_dir, f"Merged_{language}_{suffix}.tsv")
            df.to_csv(out_tsv, index=False, sep="\t", encoding="utf-8")
            tqdm.write(f"Saved {out_tsv} ({len(df)} rows)")

            if args.also_save_excel:
                out_xlsx = os.path.join(args.output_dir, f"Merged_{language}_{suffix}.xlsx")
                df.to_excel(out_xlsx, index=False)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logger = get_logger()
    logger.info(f"Starting processing with arguments: {args}")
    main(args, logger)
