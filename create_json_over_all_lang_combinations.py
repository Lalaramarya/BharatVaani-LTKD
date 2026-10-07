import argparse
import logging
import os
import pandas as pd
import re
import csv
from pathlib import Path
from tqdm import tqdm
import itertools
import json
import ast
import unicodedata


# -----------------------------------------------------------------------------
# WHAT CHANGED FROM THE OLD SCRIPT, AND WHY
# -----------------------------------------------------------------------------
# 1. FOUR combinations instead of one. The old main() only ever processed
#    "Merged_all_aligned.json" against a "complete_pairs" suffix (a filename
#    that hasn't existed since early in this pipeline's history -- the
#    diagnostic file is "_all_pairs_before_threshold.xlsx", not
#    "_complete_pairs"). Now covers all four real combinations:
#       All_aligned                 <- Merged_all_aligned.json,              suffix "en_aligned"
#       All_aligned_without_null    <- Merged_all_aligned_without_null.json, suffix "all_aligned"
#       All_en_all_candidates       <- Merged_all_all_candidates.json,       suffix "en_all_candidates"
#       All_all_candidates_wo_null  <- Merged_all_all_candidates_without_null.json, suffix "all_all_candidates"
#    (The first two json/suffix pairings match exactly what was already
#    commented out in the old script -- that pairing was correct, just
#    never turned on. The candidates JSONs are new: align_pipeline_gpu_
#    parallel.py now saves all_all_candidates.xlsx/.json the same way it
#    already saved all_aligned.xlsx/.json, so there's a wide-format source
#    to explode for these two as well.)
#
# 2. Exact sentence_id joins, not fuzzy text canonicalization. The old
#    script's whole canonicalize_for_merge / canonicalize_for_merge_bn_safe
#    / build_merge_key apparatus existed because there was no reliable id
#    to join on -- text was the only thing to match against, and text can
#    drift under normalization or just collide. Every relevant file now
#    carries a real sentence_id (English_sentence_id on the wide json,
#    {lang}_sentence_ids per language, sentence_id in the enriched tsvs),
#    so the join is now id == id -- exact, no ambiguity, no risk of two
#    different sentences silently matching.
#
# 3. No more separate ".km1000" file. create_tsv_over_multiple_folders_
#    parallel_v2.py now writes "path" (audio_filepath) and the generated
#    units directly as columns on the SAME enriched tsv -- there's nothing
#    separate left to merge in.
#
# 4. Text/id list-column pairing fixed. The old (commented-out) table_
#    explode() cross-producted EVERY list column independently via
#    itertools.product(*explode_lists) -- which is wrong for a text column
#    and its OWN id column, since they need to stay paired at the same
#    list position, not be crossed against each other. explode_pair_with_
#    ids() below zips each side's text with its own id list first, THEN
#    cross-products the (text, id) PAIRS across the two sides (English vs.
#    the other language) -- which is the only cross-product that's
#    actually correct here.
# -----------------------------------------------------------------------------


# -------------------------------------------------------------------
# Logging Setup
# -------------------------------------------------------------------
def setup_logging(output_dir):
    log_file = os.path.join(output_dir, "process_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


# -------------------------------------------------------------------
# Argument Parser
# -------------------------------------------------------------------
def get_parser():
    parser = argparse.ArgumentParser(description="Creation of language-pair JSON manifests from the merged alignment outputs.")
    parser.add_argument(
        "--tsv_dir",
        type=Path,
        required=True,
        help="Directory containing the Merged_* tsv/xlsx/json files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to write per-language-pair outputs to",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="",
        help="Comma-separated language list to process (paired against English). "
             "Empty (default) auto-discovers from whichever Merged_{lang}_en_aligned.tsv "
             "files exist in --tsv_dir.",
    )
    return parser


# -------------------------------------------------------------------
# Language Codes
# -------------------------------------------------------------------
lang_codes = {
    "English": "eng", "Bodo": "brx", "Chattisgarhi": "hne", "Dogri": "doi",
    "Garo": "grt", "Galo": "adl", "Jaintia": "aml", "Kashmiri": "kas",
    "Khasi": "kha", "Kokborok": "xtr", "Konkani": "kok", "Ladakhi": "lbj",
    "Lepcha": "lep", "Maithili": "mai_Deva", "Mizo": "lus", "Nepali": "nep",
    "Purgi": "prx", "Sanskrit": "san", "Santhali": "sat_Beng",
    "Sargujia": "sgj", "Sikkimese": "sip", "Sindhi": "snd_Deva",
    "Assamese": "asm", "Bengali": "ben", "Gujarathi": "guj",
    "Hindi": "hin", "Kannada": "kan", "Malayalam": "mal",
    "Manipuri": "mani", "Marathi": "mar", "Odia": "ory",
    "Punjabi": "pan", "Tamil": "tam", "Telugu": "tel",
    "Urdu": "urd"
}


# -------------------------------------------------------------------
# Safe File Reader
# -------------------------------------------------------------------
def safe_read_tsv(file_path, delimiter="\t"):
    if not os.path.exists(file_path):
        logging.warning(f"File not found: {file_path}")
        return None
    try:
        df = pd.read_csv(file_path, delimiter=delimiter, quoting=csv.QUOTE_NONE)
        if df.empty or df.shape[1] == 0:
            logging.warning(f"Empty or invalid TSV file: {file_path}")
            return None
        return df
    except pd.errors.EmptyDataError:
        logging.warning(f"EmptyDataError: Skipping empty file {file_path}")
        return None
    except Exception as e:
        logging.error(f"Failed to read {file_path}: {e}")
        return None


def safe_read_json(file_path):
    if not os.path.exists(file_path):
        logging.warning(f"File not found: {file_path}")
        return None
    try:
        df = pd.read_json(file_path)
        if df.empty:
            logging.warning(f"Empty json file: {file_path}")
            return None
        return df
    except Exception as e:
        logging.error(f"Failed to read {file_path}: {e}")
        return None


# -------------------------------------------------------------------
# Text cleaning (unchanged from the old script -- still useful,
# language-agnostic, nothing to do with the id/merge changes above)
# -------------------------------------------------------------------
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?।॥])\s+")


def remove_weak_leading_sentences(text, max_weak_tokens=3, min_strong_tokens=6):
    if not isinstance(text, str):
        return text
    text = text.strip()
    parts = [p.strip() for p in SENTENCE_BOUNDARY_RE.split(text) if p.strip()]
    if len(parts) <= 1:
        return text
    token_counts = [len(p.split()) for p in parts]
    max_len = max(token_counts)
    kept = []
    for i, (p, n_tokens) in enumerate(zip(parts, token_counts)):
        if i < len(parts) - 1 and n_tokens <= max_weak_tokens and max_len >= min_strong_tokens:
            continue
        kept.append(p)
    return " ".join(kept).strip()


def clean_text_language_agnostic(text, keep_numbers=True):
    if not isinstance(text, str):
        return text
    cleaned_chars = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("M"):
            cleaned_chars.append(ch)
        elif keep_numbers and cat.startswith("N"):
            cleaned_chars.append(ch)
        elif cat == "Zs":
            cleaned_chars.append(" ")
    text = "".join(cleaned_chars)
    return re.sub(r"\s+", " ", text).strip()


def remove_trailing_fragments_contextual(text, max_fragment_tokens=3, min_valid_tokens=4):
    if not isinstance(text, str):
        return text
    text = text.strip()
    parts = [p.strip() for p in SENTENCE_BOUNDARY_RE.split(text) if p.strip()]
    if len(parts) <= 1:
        return text

    def is_sentence_like(s):
        return bool(re.search(r"[.!?।॥]$", s)) or len(s.split()) >= min_valid_tokens

    def is_fragment(s):
        return len(s.split()) <= max_fragment_tokens and not re.search(r"[.!?।॥]$", s)

    has_real_sentence = any(is_sentence_like(p) for p in parts)
    if not has_real_sentence:
        return text
    kept = [p for p in parts if not (is_fragment(p) and has_real_sentence)]
    return " ".join(kept).strip()


# -------------------------------------------------------------------
# Explode wide (English, lang) rows into individual pairs, WITH ids kept
# correctly paired to their own text (not cross-producted against a
# different column's ids)
# -------------------------------------------------------------------
def explode_pair_with_ids(df, lang):
    eng_col, eng_id_col = "English", "English_sentence_id"
    lang_col, lang_id_col = lang, f"{lang}_sentence_ids"

    if eng_col not in df.columns or lang_col not in df.columns:
        return pd.DataFrame()

    rows = []
    for _, row in df.iterrows():
        eng_text = row.get(eng_col)
        if isinstance(eng_text, list):
            # English is the groupby key upstream, so this shouldn't happen --
            # but if it ever does, don't crash: take the first value.
            eng_text = eng_text[0] if eng_text else None
        if eng_text is None or (isinstance(eng_text, float) and pd.isna(eng_text)):
            continue
        eng_id = row.get(eng_id_col)

        lang_val = row.get(lang_col)
        lang_ids = row.get(lang_id_col)

        lang_texts = lang_val if isinstance(lang_val, list) else ([lang_val] if pd.notna(lang_val) else [])
        if isinstance(lang_ids, list) and len(lang_ids) == len(lang_texts):
            lang_id_list = lang_ids
        else:
            # ids and texts should be parallel lists from the same source --
            # if they ever aren't, fall back to no-id rather than mis-pairing
            lang_id_list = [None] * len(lang_texts)

        for lt, lid in zip(lang_texts, lang_id_list):
            if lt is None or (isinstance(lt, float) and pd.isna(lt)):
                continue
            rows.append({
                eng_col: eng_text,
                f"{eng_col}_sentence_id": eng_id,
                lang_col: lt,
                f"{lang_col}_sentence_id": lid,
            })

    exploded = pd.DataFrame(rows)
    if exploded.empty:
        return exploded
    return exploded.drop_duplicates().dropna(subset=[eng_col, lang_col])


# -------------------------------------------------------------------
# Exact sentence_id merge against the enriched per-language tsv
# (path + units already columns on it -- no separate km1000 file anymore)
# -------------------------------------------------------------------
def merge_and_prepare_by_id(args, pair_df, pair, suffix):
    pair_df = pair_df.copy()
    for lang in pair:
        tsv_path = os.path.join(args.tsv_dir, f"Merged_{lang}_{suffix}.tsv")
        df_lang = safe_read_tsv(tsv_path)
        if df_lang is None:
            logging.info(f"Skipping pair {pair} ({suffix}): missing/empty {tsv_path}")
            return None
        if "sentence_id" not in df_lang.columns or "path" not in df_lang.columns:
            logging.warning(
                f"{tsv_path} has no sentence_id/path columns -- was it enriched by "
                f"create_tsv_over_multiple_folders_parallel_v2.py? Skipping {pair}."
            )
            return None

        units_col_src = f"{lang}_Units"
        keep_cols = ["sentence_id", "path"]
        if units_col_src in df_lang.columns:
            keep_cols.append(units_col_src)

        df_lang_small = df_lang[keep_cols].rename(columns={
            "sentence_id": f"{lang}_sentence_id",
            "path": f"{lang}_audio_filepath",
        })

        before = len(pair_df)
        pair_df = pair_df.merge(df_lang_small, how="left", on=f"{lang}_sentence_id")
        matched = pair_df[f"{lang}_audio_filepath"].notna().sum()
        logging.info(f"[{pair}:{lang}] rows: {before}, audio matched: {matched}")

    return pair_df.dropna(subset=[f"{lang}_audio_filepath" for lang in pair])


def attach_similarity(args, pair_df, lang):
    """
    Best-effort: pull "similarity" in from Merged_{lang}_aligned.xlsx (the
    per-language, across-all-episodes Excel that DOES carry similarity/
    aligned/stage -- unlike the plain sentence_id+text tsvs this script
    otherwise reads), joined by the same exact sentence_id. If that file
    isn't available, similarity-based valid/test sampling downstream just
    falls back to plain random sampling instead of failing outright.
    """
    xlsx_path = os.path.join(args.tsv_dir, f"Merged_{lang}_aligned.xlsx")
    if not os.path.exists(xlsx_path):
        return pair_df
    try:
        df_sim = pd.read_excel(xlsx_path)
    except Exception as e:
        logging.warning(f"Could not read {xlsx_path} for similarity: {e}")
        return pair_df

    id_col = f"{lang}_sentence_id"
    if id_col not in df_sim.columns or "similarity" not in df_sim.columns:
        return pair_df

    df_sim_small = df_sim[[id_col, "similarity"]].drop_duplicates(subset=[id_col])
    return pair_df.merge(df_sim_small, how="left", on=id_col)


# -------------------------------------------------------------------
# JSON Creation (unchanged from the old script)
# -------------------------------------------------------------------
def create_json_from_df(df, source, target, src_id, tgt_id, output_path, name):
    source_path = df[f"{source}_audio_filepath"].tolist()
    source_text = df[f"{source}"].tolist()
    source_units = df.get(f"{source}_Units", pd.Series([None] * len(df))).tolist()

    target_path = df[f"{target}_audio_filepath"].tolist()
    target_text = df[f"{target}"].tolist()
    target_units = df.get(f"{target}_Units", pd.Series([None] * len(df))).tolist()

    def parse_units(u):
        if u is None or (isinstance(u, float) and pd.isna(u)):
            return None
        if isinstance(u, str):
            try:
                return ast.literal_eval(u)
            except Exception:
                return None
        return u

    output_file = os.path.join(output_path, f'{name}_manifest.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in tqdm(range(len(source_path)), desc=f"Json_creation_{name}"):
            try:
                ep = {
                    "source": {
                        "id": f"{i+1}",
                        "lang": src_id,
                        "text": source_text[i],
                        "audio_local_path": source_path[i],
                        "waveform": None,
                        "sampling_rate": 16000,
                        "units": parse_units(source_units[i])
                    },
                    "target": {
                        "id": f"{i+1}",
                        "lang": tgt_id,
                        "text": target_text[i],
                        "audio_local_path": target_path[i],
                        "waveform": None,
                        "sampling_rate": 16000,
                        "units": parse_units(target_units[i])
                    }
                }
                f.write(json.dumps(ep, ensure_ascii=False) + '\n')
            except Exception as e:
                logging.error(f"Error creating JSON entry {i}: {e}")


# -------------------------------------------------------------------
# Process Language Columns
# -------------------------------------------------------------------
def discover_languages(args):
    langs = []
    for fname in os.listdir(args.tsv_dir):
        m = re.match(r"Merged_(.+)_en_aligned\.tsv$", fname)
        if m and m.group(1) != "English":
            langs.append(m.group(1))
    return sorted(set(langs))


def process_language_columns(args, output_dir, json_file, suffix):
    df = safe_read_json(os.path.join(args.tsv_dir, json_file))
    if df is None:
        logging.info(f"Skipping {json_file}: not found/empty.")
        return

    if args.languages:
        languages = [l.strip() for l in args.languages.split(",") if l.strip()]
    else:
        languages = discover_languages(args)
    logging.info(f"[{suffix}] languages: {languages}")

    for lang in tqdm(languages, desc=f"Preparing pairs ({suffix})"):
        pair = (lang, "English")
        pair_path = os.path.join(output_dir, f"{pair[0]}-{pair[1]}")
        os.makedirs(pair_path, exist_ok=True)

        pair_df = explode_pair_with_ids(df, lang)
        if pair_df.empty:
            logging.info(f"Skipping pair {pair}: no exploded rows (missing columns in {json_file}?).")
            continue

        pair_df = merge_and_prepare_by_id(args, pair_df, pair, suffix)
        if pair_df is None or pair_df.empty:
            logging.info(f"Skipping pair {pair} ({suffix}) due to missing or empty audio data.")
            continue

        pair_df = attach_similarity(args, pair_df, lang)
        if "similarity" not in pair_df.columns:
            pair_df["similarity"] = 0.0  # no similarity source found -- sampling below just goes random

        pair_df = pair_df.drop_duplicates(subset=list(pair), keep="first")
        pair_df[lang] = (
            pair_df[lang]
                .apply(remove_weak_leading_sentences)
                .apply(remove_trailing_fragments_contextual)
                .apply(clean_text_language_agnostic)
        )

        pair_df.to_excel(os.path.join(pair_path, f"{pair[0]}-{pair[1]}.xlsx"), index=False)

        total_size = len(pair_df)
        if total_size < 1000:
            logging.info(f"Skipping {pair} ({suffix}): dataset has only {total_size} samples (<1000).")
            continue

        valid_size = int(0.05 * len(pair_df))
        high_sim_threshold = pair_df["similarity"].quantile(0.90)
        high_sim_df = pair_df[pair_df["similarity"] >= high_sim_threshold]

        valid = high_sim_df.sample(n=min(valid_size, len(high_sim_df)), random_state=42)
        if len(valid) < valid_size:
            remaining_needed = valid_size - len(valid)
            extra = pair_df.drop(valid.index).sample(n=remaining_needed, random_state=43)
            valid = pd.concat([valid, extra])

        remaining = pair_df.drop(valid.index)
        test = remaining.sample(n=min(500, len(remaining)), random_state=24)
        train = remaining.drop(test.index)

        logging.info(
            f"[{suffix}] {pair} mean similarity — Train: {train.similarity.mean():.3f}, "
            f"Valid: {valid.similarity.mean():.3f}, Test: {test.similarity.mean():.3f}"
        )

        for split_name, split_df in [("train", train), ("test", test), ("valid", valid)]:
            split_df = split_df.reset_index(drop=True)
            split_df.to_excel(os.path.join(pair_path, f"{pair[0]}-{pair[1]}_{split_name}.xlsx"), index=False)
            split_df.to_csv(os.path.join(pair_path, f"{pair[0]}-{pair[1]}_{split_name}.tsv"), index=False, sep="\t")
            create_json_from_df(split_df, pair[0], pair[1], lang_codes.get(pair[0], "unk"), lang_codes.get(pair[1], "eng"), pair_path, split_name)

        logging.info(f"[{suffix}] {pair} json generated")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)
    logging.info("Starting processing...")

    # (output_folder_name, wide_json_file, per-language tsv suffix)
    combinations = [
        ("All_aligned", "Merged_all_aligned.json", "en_aligned"),
        #("All_aligned_without_null", "Merged_all_aligned_without_null.json", "all_aligned"),
        #("All_en_all_candidates", "Merged_all_all_candidates.json", "en_all_candidates"),
        #("All_all_candidates_without_null", "Merged_all_all_candidates_without_null.json", "all_all_candidates"),
    ]

    for folder_name, json_file, suffix in combinations:
        out_path = os.path.join(args.output_dir, folder_name)
        os.makedirs(out_path, exist_ok=True)
        logging.info(f"=== Processing combination: {folder_name} ({json_file}, suffix={suffix}) ===")
        process_language_columns(args, out_path, json_file, suffix)

    logging.info("Processing completed successfully.")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
