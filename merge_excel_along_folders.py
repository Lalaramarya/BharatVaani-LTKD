import argparse
import logging
import os
import re
import fnmatch
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# -----------------------------------------------------------------------------
# WHAT CHANGED FROM THE OLD SCRIPT, AND WHY
# -----------------------------------------------------------------------------
# 1. The old script hardcoded two separate ~34-item language-name lists (one
#    commented out, one live) to build excel_sheets, plus a string-replace
#    ("_aligned" -> "_complete_pairs") to guess the diagnostic file's name.
#    Two real problems with that: (a) it silently misses any language not on
#    the list, or any NEW per-episode Excel file the pipeline starts writing
#    later, with no warning; (b) "_complete_pairs.xlsx" is the OLD filename --
#    the pipeline now calls this "_all_pairs_before_threshold.xlsx", and the
#    old string-replace trick can't produce that new name at all.
#
#    This version instead DISCOVERS which Excel files actually exist by
#    scanning the episode folders themselves, matching against the current
#    pipeline's real suffix patterns:
#       {lang}_aligned.xlsx                      -- final accepted output
#       {lang}_sliding_aligned.xlsx               -- sliding leg's own accepts
#       {lang}_aligned_topk_only.xlsx             -- independent full-topk leg
#       {lang}_all_pairs_before_threshold.xlsx    -- complete, unfiltered diagnostic
#       {lang}_aligned_both_candidates.xlsx       -- only if --save_all_candidates was used
#    plus the two top-level (non-per-language) combined files:
#       all_aligned.xlsx / all_aligned_without_null.xlsx
#    Whatever's actually present gets merged -- nothing to keep in sync by
#    hand when a language is added/removed or a run does/doesn't use
#    --save_all_candidates.
#
# 2. The hardcoded 16-episode skip list is now a --exclude_episodes CLI arg
#    (comma-separated, optional) instead of baked into the code -- same
#    exclusion behavior, but visible and changeable without editing the script.
# -----------------------------------------------------------------------------


PER_LANGUAGE_SUFFIX_PATTERNS = [
    "*_aligned.xlsx",
    "*_sliding_aligned.xlsx",
    "*_aligned_topk_only.xlsx",
    "*_all_pairs_before_threshold.xlsx",
    "*_aligned_both_candidates.xlsx",
]

TOP_LEVEL_FILES = ["all_aligned.xlsx", "all_aligned_without_null.xlsx",
                    "all_all_candidates.xlsx", "all_all_candidates_without_null.xlsx"]

# These would otherwise match "*_aligned.xlsx" but aren't per-language files --
# exclude them from the per-language discovery pass (they're merged separately
# as TOP_LEVEL_FILES instead).
TOP_LEVEL_EXCLUDE = set(TOP_LEVEL_FILES)


def get_parser():
    parser = argparse.ArgumentParser(description="Merge per-episode Excel/JSON outputs across all episode folders.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing one subfolder per episode",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to write the merged Excel/JSON files to",
    )
    parser.add_argument(
        "--exclude_episodes",
        type=str,
        default="",
        help="Comma-separated episode folder names to skip entirely (e.g. known-bad "
             "or already-superseded episodes). Empty by default -- everything found "
             "under --input_dir is processed.",
    )
    return parser


def extract_episode_number(folder_name):
    match = re.search(r'MKB_(\d+)', folder_name)
    if match:
        return int(match.group(1))
    return float('inf')


def discover_per_episode_excel_files(input_dir, episode_folders):
    """
    Scans every episode folder and returns the set of Excel filenames that
    actually exist ANYWHERE across all episodes, matching the known
    per-language suffix patterns. This is what replaces the old hardcoded
    language-name lists -- whatever's really there gets merged, regardless
    of which specific languages or optional flags (--save_all_candidates)
    produced it.
    """
    discovered = set()
    for episode_folder in episode_folders:
        episode_path = os.path.join(input_dir, episode_folder)
        if not os.path.isdir(episode_path):
            continue
        for fname in os.listdir(episode_path):
            if not fname.endswith(".xlsx"):
                continue
            if fname in TOP_LEVEL_EXCLUDE:
                continue
            if any(fnmatch.fnmatch(fname, pat) for pat in PER_LANGUAGE_SUFFIX_PATTERNS):
                discovered.add(fname)
    return sorted(discovered)


def main(args):
    json_files = ['all_aligned.json', 'all_aligned_without_null.json',
                  'all_all_candidates.json', 'all_all_candidates_without_null.json']

    exclude_episodes = {e.strip() for e in args.exclude_episodes.split(",") if e.strip()}

    episodes_folders = sorted(os.listdir(args.input_dir), key=extract_episode_number)
    episodes_folders = [e for e in episodes_folders if e not in exclude_episodes]

    os.makedirs(args.output_dir, exist_ok=True)

    excel_sheets = discover_per_episode_excel_files(args.input_dir, episodes_folders)
    excel_sheets.extend(TOP_LEVEL_FILES)

    logging.info(f"Discovered {len(excel_sheets)} distinct Excel filenames to merge: {excel_sheets}")

    for excel_sheet in tqdm(excel_sheets, desc="Merging excel sheets"):
        merged_df = pd.DataFrame()
        for episode_folder in tqdm(episodes_folders, desc="Across all episodes", leave=False):
            episode_path = os.path.join(args.input_dir, episode_folder)
            if not os.path.isdir(episode_path):
                tqdm.write(f"Skipping {episode_folder}")
                continue
            file_path = os.path.join(episode_path, excel_sheet)
            if os.path.exists(file_path):
                df = pd.read_excel(file_path)
                if not df.empty:
                    # FIX: this used to dedupe by TEXT (col[0]/col[1] --
                    # the target-lang and English sentence text), which
                    # silently collapses two GENUINELY DISTINCT sentence
                    # occurrences that just happen to share identical
                    # wording (a repeated phrase at two different
                    # timestamps in the same episode) -- exactly the
                    # pattern collapse_to_best_per_english() in the main
                    # alignment pipeline was fixed to KEEP separate, by
                    # grouping on english_idx (position) instead of text.
                    # Since this dedup ran independently per file type
                    # (aligned.xlsx vs. sliding_aligned.xlsx vs. ...), it
                    # could strip a different number of "duplicate text"
                    # rows from each, breaking the expected ordering
                    # relationship between them (e.g. {lang}_aligned.xlsx
                    # ending up with FEWER rows than {lang}_sliding_
                    # aligned.xlsx, which should never happen -- sliding's
                    # own rows always survive into the collapsed output).
                    # Now dedupes by sentence_id instead, when available --
                    # true exact-duplicate ROWS (same id appearing twice,
                    # e.g. from a stray re-run) still get removed, but two
                    # different sentences with matching text don't.
                    id_cols = [c for c in df.columns if c.endswith("_sentence_id")]
                    if id_cols:
                        df = df.drop_duplicates(subset=id_cols)
                    else:
                        # No id columns on this file (older run without them) --
                        # fall back to the old text-based dedup rather than
                        # skip deduping entirely.
                        col = df.columns
                        df = df.drop_duplicates(subset=[col[0]])
                        df = df.drop_duplicates(subset=[col[1]])
                        df = df.drop_duplicates(subset=[col[0], col[1]])
                merged_df = pd.concat([merged_df, df], ignore_index=True)
        merged_df.to_excel(os.path.join(args.output_dir, f"Merged_{excel_sheet}"), index=False)
        tqdm.write(f"Merged_{excel_sheet} saved ({len(merged_df)} rows)")

    for json_file in tqdm(json_files, desc="Merging json files"):
        merged_df = pd.DataFrame()
        for episode_folder in tqdm(episodes_folders, desc="Across all episodes", leave=False):
            episode_path = os.path.join(args.input_dir, episode_folder)
            if not os.path.isdir(episode_path):
                tqdm.write(f"Skipping {episode_folder}")
                continue
            file_path = os.path.join(episode_path, json_file)
            if os.path.exists(file_path):
                df = pd.read_json(file_path)
                merged_df = pd.concat([merged_df, df], ignore_index=True)
        merged_df.to_json(os.path.join(args.output_dir, f"Merged_{json_file}"), orient="records", force_ascii=False, indent=0)
        tqdm.write(f"Merged_{json_file} saved ({len(merged_df)} rows)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
    parser = get_parser()
    args = parser.parse_args()
    main(args)
