# align_pipeline_gpu_parallel.py
# Parallelized pipeline (Option A) with optional multi-GPU servers.
# Preserves original functionality. Use --run_mode parallel to enable.
#
# ============================================================================
# PIPELINE OVERVIEW -- what this file does, episode by episode, top to bottom
# ============================================================================
# For every episode, for every non-English language configured (see
# process_episode()):
#
#   STEP 0 -- Split into sentences (split_sentences()).
#     Each language's raw transcript text is split into sentences with
#     Stanza. Low-resource / noisy-ASR languages use lenient splitting
#     (no minimum length, no buffering); everything else uses stricter,
#     buffered splitting that assumes reasonably punctuated text. Every
#     split sentence gets a deterministic id (make_sentence_id()) of the
#     form "<episode>_<lang>_<hash>", written to a sidecar .ids.tsv file
#     alongside the plain-text file consumed by downstream forced-alignment
#     tooling.
#
#   STEP 1 -- Sliding-window alignment (align_sentences_to_paragraph_
#             twostage_unified()).
#     Every English sentence searches the FULL target-language paragraph
#     (not sentence-by-sentence) using a two-stage LaBSE -> SONAR fused
#     score. This handles cases where sentence boundaries don't line up
#     1:1 across languages. Returns:
#       - aligned_pairs:  pairs that cleared fused_min (the real output)
#       - full_units:     the induced sentence-level breakdown, aligned AND
#                          unaligned, used for audio train/test splitting
#       - all_compared:   a pre-threshold diagnostic view -- every English
#                          sentence's best candidate, whether or not it was
#                          accepted, tagged aligned=True/False
#     (A SONAR-only sibling of this function,
#     align_sentences_to_paragraph_twostage_unified_sonar_paragraph_aligned(),
#     is also defined below and still usable, but every language -- including
#     Manipuri -- now goes through the fused LaBSE+SONAR version above, so
#     results are directly comparable across languages.)
#
#   STEP 2 -- Mutual top-k alignment (align_sentences_mutual_topk()).
#     Runs over EVERY split sentence for this language (not just ones the
#     sliding stage missed) against every English sentence, using
#     reciprocal forward/backward top-k cosine similarity: a pair is only
#     accepted if each side is genuinely one of the other's best matches.
#     `max_usage` caps how many times any single sentence can be reused
#     across accepted pairs, to stop one "attractor" sentence from
#     absorbing many unrelated matches. Because this now runs over the full
#     list, it will find candidates that overlap with what Step 1 already
#     found -- see Step 3.
#
#   STEP 3 -- Merge Step 1 + Step 2 into one output per English sentence
#             (collapse_to_best_per_english()).
#     Both stages' pairs are tagged with which stage produced them and
#     whether that stage accepted them, then collapsed down to a single
#     best row per English sentence (sliding preferred over topk when both
#     accepted it, since the two stages' similarity scores are on
#     different, non-comparable scales). This is what stops the same
#     English sentence from appearing twice in the final output.
#
#   STEP 4 -- Save outputs.
#     - {lang}_aligned.xlsx / all_aligned.xlsx / all_aligned.json:
#       the real, deduplicated parallel corpus (Step 3's output).
#     - {lang}_all_pairs_before_threshold.xlsx: the FULL diagnostic view
#       (all_compared from both stages, collapsed per-sentence via
#       collapse_to_best_per_sentence() -- a SEPARATE dedup pass from
#       Step 3's, kept for inspecting rejected/borderline candidates, not
#       just the accepted ones).
#     - *_aligned_audio.txt / *_unaligned_audio.txt: plain-text splits used
#       to feed downstream MMS forced-alignment, with sentence ids in a
#       parallel sidecar file (same line order) rather than inline, since
#       the forced-alignment tool consumes these files as plain text.
# ============================================================================

import os
import re
import unicodedata
import argparse
import hashlib
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from langdetect import detect_langs
from scipy.spatial.distance import cosine
from simalign import SentenceAligner
from collections import defaultdict
import stanza
import multiprocessing as mp
import time
import uuid

# ------------------------------- User-tunable defaults -------------------------------
DEFAULT_BATCH_SIZE = 64  # embedding batch size used by GPU server
EMBED_CACHE = {}  # per-process small cache for sequential mode (kept)
# ------------------------------------------------------------------------------------

# 🔐 Global semaphore to limit concurrent GPU embedding jobs
# This is the key protection against GPU OOM when multiple workers are active.
GPU_SEMAPHORE = mp.Semaphore(1)  # allow only 1 concurrent embedding job per GPU

# Device info
MAIN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔹 Main device (launcher): {MAIN_DEVICE}")

# Keep your existing flags
USE_SONAR = True
USE_SIMALIGN = True

# ---------------------- language maps (unchanged) ----------------------
language_codes = {
    "English": "eng", "Hindi": "hin", "Urdu": "urd-script_arabic",
    "Assamese": "asm", "Bengali": "ben", "Marathi": "mar", "Tamil": "tam",
    "Telugu": "tel", "Kannada": "kan", "Malayalam": "mal", "Punjabi": "pan",
    "Manipuri": "ben","Gujarathi":"guj","Odia": "ory",
    "Bodo": "asm", "Chattisgarhi": "hin", "Dogri": "hin",
    "Garo": "eng", "Galo": "eng", "Jaintia": "eng", "Kashmiri": "urd-script_arabic",
    "Khasi": "eng", "Kokborok": "ben", "Konkani": "hin", "Ladakhi": "eng",
    "Lepcha": "eng", "Maithili": "hin", "Mizo": "eng", "Nepali": "hin",
    "Purgi": "urd-script_arabic", "Sanskrit": "hin", "Santhali": "ben",
    "Sargujia": "hin", "Sikkimese": "eng", "Sindhi": "hin",
}

sonar_language_codes = {
    "English": "eng_Latn", "Hindi": "hin_Deva", "Urdu": "urd_Arab",
    "Assamese": "asm_Beng", "Bengali": "ben_Beng", "Marathi": "mar_Deva",
    "Tamil": "tam_Taml", "Telugu": "tel_Telu", "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym", "Punjabi": "pan_Guru","Manipuri":"ben_Beng",
    "Gujarathi": "guj_Gujr","Odia": "ory_Orya",
    "Bodo": "asm_Beng", "Chattisgarhi": "hin_Deva", "Dogri": "hin_Deva",
    "Garo": "eng_Latn", "Galo": "eng_Latn", "Jaintia": "eng_Latn", "Kashmiri": "urd_Arab",
    "Khasi": "eng_Latn", "Kokborok": "ben_Beng", "Konkani": "hin_Deva", "Ladakhi": "eng_Latn",
    "Lepcha": "eng_Latn", "Maithili": "hin_Deva", "Mizo": "eng_Latn", "Nepali": "hin_Deva",
    "Purgi": "urd_Arab", "Sanskrit": "hin_Deva", "Santhali": "ben_Beng",
    "Sargujia": "hin_Deva", "Sikkimese": "eng_Latn", "Sindhi": "hin_Deva",
}

detect_lang = {
    "eng": "en", "hin": "hi", "urd-script_arabic": "ur",
    "asm": "bn", "ben": "bn", "mar": "mr", "tam": "ta",
    "tel": "te", "kan": "multilingual", "mal": "ml", "pan": "multilingual",
    "mani":"bn","guj":"multilingual","ory":"or","bod":"multilingual"
}
LOW_RESOURCE_LANGS = [
    "Bodo", "Chattisgarhi", "Dogri","Garo", "Galo", "Jaintia", "Kashmiri","Khasi", "Kokborok", "Konkani", "Ladakhi","Lepcha", "Maithili", "Manipuri","Mizo", "Nepali",
    "Purgi", "Sanskrit", "Santhali","Sargujia", "Sikkimese", "Sindhi"]

# ---------------------- deterministic sentence IDs ----------------------
# Called from every save point below (split_sentences, aligned/unaligned
# audio text, *_aligned.xlsx, *_all_pairs_before_threshold.xlsx,
# all_aligned.xlsx/json). Depends on nothing but its three inputs, so any
# two occurrences of the exact same (lang, episode, text) -- computed at
# completely different points in the pipeline -- automatically get the
# SAME id, with no lookup table or shared state required. Only genuinely
# different text (e.g. a ragged sliding-window/topk fragment that doesn't
# match a clean split_sentences() boundary) gets a different id, which is
# itself a useful signal downstream.
#
# normalize_for_id() must NEVER change after episodes have been processed
# with it -- changing it changes every id for previously-processed data,
# silently breaking any join against data saved before the change.

def normalize_for_id(text, lang=None):
    """
    Normalization used ONLY for id hashing -- the original text is still
    written out as-is everywhere (this never touches the actual saved text).

    FIX: this used to be whitespace-only. But normalize_final_pairs() (the
    step that builds the real accepted-pair output -- en_aligned.tsv,
    all_aligned.tsv, etc.) already runs the target-language text through
    normalize_text_new() (punctuation, digit, and -- for several languages
    -- Indic script normalization, e.g. chandrabindu "ँ" -> anusvara "ं" for
    Hindi-family languages) before it's saved. split_sentences.txt /
    full_paragraph.txt (and their .ids.tsv sidecars, which is what
    manifest.json's "id" field ultimately traces back to) are written from
    the RAW, un-normalized source text. Same underlying sentence, two
    different strings, therefore two different hashes -- which is exactly
    why a real run showed ~47% of en_aligned.tsv's ids failing to find a
    manifest match, even though the actual audio for those sentences did
    get split (see MKB_120_March_2025, Chattisgarhi: "इहाँ" (raw, U+0901
    chandrabindu) vs "इहां" (post-normalize_text_new, U+0902 anusvara) --
    same word, different id.

    Now: apply the SAME normalize_text_new() treatment here too (when a
    language is resolvable via language_codes), so raw text and
    already-normalized text both collapse to the identical canonical form
    before hashing -- confirmed normalize_text_new() is idempotent, so this
    doesn't change behavior for text that was already normalized. Falls
    back to the old whitespace-only behavior if lang isn't given or isn't
    in language_codes (e.g. "English", which normalize_text_new doesn't
    apply any script-specific normalization to anyway).
    """
    text = str(text)
    lang_code = language_codes.get(lang) if lang else None
    if lang_code:
        try:
            text = normalize_text_new(text, lang_code)
        except Exception:
            pass  # never let id computation itself fail on odd input
    return " ".join(text.split())


def _safe_id_component(value):
    """Make an episode/lang name safe to sit inside an id string: collapse
    any whitespace to underscores. We don't need to guard against "_"
    itself (ids are never split back apart programmatically -- they're
    read as one opaque string), so this is intentionally minimal."""
    return re.sub(r"\s+", "_", str(value).strip())


def make_sentence_id(lang, episode, text, length=16):
    """Deterministic id: same (lang, episode, text) -> same id everywhere,
    REGARDLESS of whether that text has already passed through
    normalize_text_new() or not (see normalize_for_id()'s docstring).

    Format: "<episode>_<lang>_<hash>", e.g. "ep03_Manipuri_a1b2c3d4e5f6a7b8".
    The hash suffix (length=16 hex chars = 64 bits) is still what
    guarantees uniqueness/determinism -- the episode/lang prefix is purely
    for human readability when scanning a spreadsheet or grepping a file,
    it plays no role in collision-avoidance.

    IMPORTANT: this must stay a pure function of (lang, episode, text) --
    no other input (e.g. which stage produced a row) may change the id.
    An earlier version accepted an optional `align_type` to tag ids with
    "sliding"/"topk", but that meant the SAME sentence got a DIFFERENT id
    in the Excel sheets than in split_sentences.txt's/aligned_audio.txt's/
    unaligned_audio.txt's `.ids.tsv` sidecars (which never had access to a
    stage and always used the plain format) -- breaking any join between
    them. Reverted: every id, everywhere in the pipeline, is only ever
    <episode>_<lang>_<hash>.

    NOTE: this changes ids again vs. the previous (whitespace-only)
    version, for any text that normalize_text_new() actually changes
    (punctuation/digits/certain Indic matras) -- expected and necessary,
    since the old ids were the actual bug. Anything joining against ids
    from before this fix needs those files regenerated.
    """
    normalized = normalize_for_id(text, lang)
    key = f"{lang}|{episode}|{normalized}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:length]
    return f"{_safe_id_component(episode)}_{_safe_id_component(lang)}_{h}"


def sanitize_for_tsv_field(text):
    """
    Strips/replaces characters that would corrupt a hand-built tab-
    separated row if they showed up inside a text field: a literal
    embedded tab shifts every column after it, and a literal embedded
    CR/LF splits one logical row into multiple physical lines. Neither
    is meaningful content in a single sentence/span of text -- it's
    always some kind of stray control character or copy-paste artifact.

    This is a write-time fix, not a read-time one: no read-side csv
    quoting setting can recover a row that was already corrupted before
    it was written. Doesn't touch the plain single-column MMS-input
    files (split_sentences.txt/aligned_audio.txt/unaligned_audio.txt/
    full_paragraph.txt) -- those aren't tab-separated, so an embedded
    tab is harmless there; only a literal newline would matter, and text
    that far gone is already a display/quality problem, not a structural
    one, for those files specifically.
    """
    if text is None:
        return ""
    return re.sub(r"[\t\r\n]+", " ", str(text)).strip()


def save_sentence_id_sidecar(sentences, lang, episode, text_file_path):
    """
    Writes a companion "<text_file_path>.ids.tsv" alongside a PLAIN TEXT
    file that some other tool (MMS forced-alignment) consumes directly --
    so that file's content itself stays untouched (one sentence per line,
    no id prefix) while still making sentence_id available to anything
    downstream that wants it. Format: line_index<TAB>sentence_id<TAB>text,
    one row per line of `sentences`, in the SAME order.

    Consumers (e.g. force_aligned_over_folders_parallel.py, once MMS
    forced-alignment has produced a manifest.json for this same text file)
    can join this sidecar back onto the manifest by LINE ORDER, since MMS
    forced-alignment consumes the text file 1:1, one manifest entry per
    input line, in order.
    """
    sidecar_path = text_file_path + ".ids.tsv"
    with open(sidecar_path, "w", encoding="utf-8") as f:
        for i, s in enumerate(sentences):
            sid = make_sentence_id(lang, episode, s)
            f.write(f"{i}\t{sid}\t{sanitize_for_tsv_field(s)}\n")
    return sidecar_path

# ---------------------- Utilities and normalization (unchanged) ----------------------
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def normalize_unicode(text):
    return unicodedata.normalize("NFC", text)

def strip_control_chars(text):
    return re.sub(r"[\u200B-\u200D\uFEFF]", "", text)

def normalize_whitespace(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def normalize_punctuation(text):
    replacements = {
        "“": '"', "”": '"', "‘": "'", "’": "'", "—": "-", "–": "-",
        "…": "...", "ـ": "", "•": ".", "·": "."
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

def normalize_digits(text):
    return "".join(str(unicodedata.digit(ch)) if ch.isdigit() else ch for ch in text)

def normalize_arabic_script(text):
    replacements = {
        "ى": "ي", "يٰ": "ي", "ة": "ه", "ۀ": "ه",
        "أ": "ا", "إ": "ا", "آ": "ا",
        "ً": "", "ٌ": "", "ٍ": "", "َ": "", "ُ": "", "ِ": "", "ّ": "", "ْ": ""
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    text = text.replace("ـ", "")
    return text

def normalize_indic_script(text):
    nukta_map = {
        "ड़": "ड", "ढ़": "ढ",
        "क़": "क", "ख़": "ख", "ग़": "ग",
        "ज़": "ज", "फ़": "फ", "य़": "य"
    }
    for k, v in nukta_map.items():
        text = text.replace(k, v)
    text = text.replace("ँ", "ं")
    return text

def normalize_assamese(text):
    text = re.sub(r"(?<![অ-হয়ৰৱ])([ািীুূৃেৈোৌ])", r"", text)
    text = re.sub(r"([োো])\s+([োো])", r"\1", text)
    text = re.sub(r"([অ-হয়ৰৱ])\s+([া-ৌেো])", r"\1\2", text)
    text = re.sub(r"^[া-ৌেো]+" , "", text)
    text = re.sub(r"([়])\1+", r"\1", text)
    text = text.replace("্ী", "ী")
    text = text.replace("  ", " ")
    return text

def normalize_text_new(text, lang_code):
    text = normalize_unicode(text)
    text = strip_control_chars(text)
    text = normalize_punctuation(text)
    text = normalize_digits(text)
    text = normalize_whitespace(text)

    if "urd" in lang_code or "arabic" in lang_code:
        text = normalize_arabic_script(text)
    elif lang_code == "asm" or lang_code == "ben":
        text = normalize_assamese(text)
    elif lang_code in {"hin","mar","tam","tel","kan","mal","pan","guj","ory"}:
        text = normalize_indic_script(text)

    text = normalize_whitespace(text)
    return text

# Splitting (use stanza as before)
nlp_pipelines = {}
def get_nlp_pipeline(lang_code):
    if lang_code not in nlp_pipelines:
        nlp_pipelines[lang_code] = stanza.Pipeline(
            lang=detect_lang[lang_code],
            processors='tokenize',
            tokenize_no_ssplit=False,
            use_gpu=torch.cuda.is_available(),
            download_method=stanza.DownloadMethod.REUSE_RESOURCES
        )
    return nlp_pipelines[lang_code]

def split_sentences(text, lang_code=None, min_len=0, min_chars=0, use_buffer=False, normalize=False):
    """
    min_len: existing behavior, unchanged -- a candidate must satisfy BOTH
      len(candidate) >= min_len AND len(candidate.split()) >= min_len (or,
      in use_buffer mode, get merged into a neighbor if it doesn't). Note
      this ties character-length and word-count to the SAME threshold
      value, which is fine when min_len is small/0 but gets awkward for
      anything requiring multiple words.
    min_chars: NEW, independent character-length floor, default 0 (no
      effect, fully backward compatible). Use this when you want to drop
      tiny junk (a lone letter, a 2-3 character ASR fragment) WITHOUT
      also requiring multiple words -- e.g. min_chars=3, min_len=0 drops
      anything under 3 characters but still keeps valid short one-word
      utterances, unlike bumping min_len itself (which would also start
      requiring len(candidate.split()) >= min_len, dropping legitimate
      single-word sentences).
    """
    sentences = []
    if lang_code:
        try:
            nlp = get_nlp_pipeline(lang_code)
            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sentences if sent.text.strip()]
        except Exception as e:
            tqdm.write(f"⚠️ Stanza splitting failed: {e}")
            sentences = [text.strip()]
    else:
        sentences = [text.strip()]

    final_sentences = []
    #sentence_enders = r'[।?؟॥።।܀۔⁇⁈⁉‼⸮⸼…—\~\n\r\t\.|]'
    sentence_enders = r"(?:[।?؟॥።܀۔⁇⁈⁉‼⸮⸼…—~\.\|]|[\n\r\t]+)"
    pattern = f'({sentence_enders}+)\\s*'

    for sent in sentences:
        parts = re.split(pattern, sent)
        combined = [(parts[i] + parts[i + 1]).strip() for i in range(0, len(parts) - 1, 2)]
        if len(parts) % 2 == 1:
            combined.append(parts[-1].strip())

        buffer = ""
        for s in combined:
            candidate = s.strip()
            if normalize:
                candidate = normalize_text_new(candidate, lang_code)

            if use_buffer:
                if (
                    (len(candidate) <= min_len and len(candidate.split()) < min_len)
                    or len(candidate) < min_chars
                    or re.fullmatch(r'[\W_]+', candidate)
                ):
                    buffer += " " + candidate
                else:
                    if buffer:
                        if final_sentences:
                            final_sentences[-1] += buffer
                        else:
                            final_sentences.append(buffer.strip())
                        buffer = ""
                    final_sentences.append(candidate)
            else:
                if (
                    (len(candidate) >= min_len and len(candidate.split()) >= min_len)
                    and len(candidate) >= min_chars
                    and not re.fullmatch(r'[\W_]+', candidate)
                ):
                    final_sentences.append(candidate)

        if use_buffer and buffer:
            if final_sentences:
                final_sentences[-1] += buffer
            else:
                final_sentences.append(buffer.strip())

    return [s for s in final_sentences if s.strip()]

def filter_language_mismatch(sentences, lang):
    filtered = []
    for s in sentences:
        try:
            detected = detect_langs(s)[0].lang
            if lang == "English" and detected != "en":
                continue
            if lang != "English" and detected == "en":
                continue
        except:
            pass
        filtered.append(s)
    return filtered

# mean_pooling (unchanged)
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

# --------------------------------------------------------------------------------
# Sequential (single-process) model loading for local runs (same as original)
# --------------------------------------------------------------------------------
from transformers import AutoTokenizer, AutoModel

# ---------------------------------------------------------------------------
# Sequential model set -- declared as None here, actually loaded by calling
# init_sequential_models() (see below). NOT loaded automatically at import.
#
# WHY THIS CHANGED FROM UNCONDITIONAL TOP-LEVEL LOADING:
# This used to run as bare module-level code, so it executed on EVERY import
# of this file -- including inside every spawned child process (mp uses the
# "spawn" start method, which re-imports the module in the child before
# running the target function). Two problems followed from that:
#
#   1. In --run_mode parallel (old GPU-server design), each gpu_server_loop
#      child process would import this module (loading a full model set
#      onto MAIN_DEVICE, i.e. always cuda:0, regardless of which GPU that
#      server was actually assigned) and THEN call load_models_on_device()
#      to load a SECOND full model set onto its real assigned device --
#      silently doubling GPU memory usage in every server process.
#   2. There was no way to load a full model set onto a NON-default device
#      without also paying for an unwanted cuda:0 copy first.
#
# Making this an explicit function call fixes both: each process now loads
# exactly one model set, on exactly the device it asks for.
# ---------------------------------------------------------------------------
text_embedder_seq = None
labse_model_seq = None
gte_tokenizer_seq = None
gte_model_seq = None
bge_tokenizer_seq = None
bge_model_seq = None
aligner_seq = None

# Device actually used by the model set currently loaded in THIS process.
# get_embeddings_sequential() / cosine_sim_matrix() read this instead of
# hardcoding MAIN_DEVICE, so a worker whose models were loaded onto e.g.
# cuda:2 doesn't silently do its tensor math against cuda:0.
CURRENT_DEVICE = MAIN_DEVICE


def init_sequential_models(device=None):
    """
    Loads ONE full copy of every model (SONAR, LaBSE, GTE, BGE, SimAlign)
    onto `device` (defaults to MAIN_DEVICE) into this module's globals --
    text_embedder_seq / labse_model_seq / gte_model_seq / bge_model_seq /
    aligner_seq -- which get_embeddings_sequential() and
    align_sentences_to_paragraph_twostage_unified() read directly.

    Call this exactly ONCE per process, before doing any alignment work:
      - run_sequential_mode() calls it once in the main process.
      - the parallel worker-pool (_pool_worker_init) calls it once per
        worker process, pointed at that worker's assigned GPU -- so every
        worker gets its OWN full model copy (~8-9GB) rather than every
        process funnelling embedding requests through one shared model via
        a queue + semaphore.
    """
    global text_embedder_seq, labse_model_seq, gte_tokenizer_seq, gte_model_seq
    global bge_tokenizer_seq, bge_model_seq, aligner_seq, CURRENT_DEVICE

    dev = device if device is not None else MAIN_DEVICE
    CURRENT_DEVICE = dev

    if USE_SONAR:
        print(f"🔹 [pid={os.getpid()}] Loading Sonar on {dev} ...")
        text_embedder_seq = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=dev
        )

    labse_model_seq = SentenceTransformer("sentence-transformers/LaBSE", device=str(dev))

    gte_tokenizer_seq = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True)
    gte_model_seq = AutoModel.from_pretrained("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True).to(dev)

    bge_tokenizer_seq = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    bge_model_seq = AutoModel.from_pretrained("BAAI/bge-m3").to(dev)

    if USE_SIMALIGN:
        print(f"🔹 [pid={os.getpid()}] Loading SimAlign ...")
        aligner_seq = SentenceAligner(model="bert", token_type="bpe", matching_methods="mai")

    print(f"[INFO] [pid={os.getpid()}] Sequential models loaded on {dev}.")

# -------------------------------------------------------------------------
# LEGACY / not used by default --run_mode parallel anymore (see
# run_parallel_mode's worker-pool docstring above for why: this shared-
# model + semaphore design serialized every embedding call GPU-wide,
# regardless of worker count). Left in place in case a future setup
# genuinely can't fit more than one model copy per GPU and needs a shared-
# server fallback instead of independent per-worker copies.
#
# GPU server code: runs in its own process and loads models on assigned GPU
# -------------------------------------------------------------------------
def load_models_on_device(device_id):
    """Load the same set of models on the specified CUDA device and return as dict."""
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    models = {}
    if USE_SONAR:
        models["text_embedder"] = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=device
        )
    models["labse"] = SentenceTransformer("sentence-transformers/LaBSE", device=str(device))
    from transformers import AutoTokenizer, AutoModel
    #models["indic_tokenizer"] = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
    #models["indic_model"] = AutoModel.from_pretrained("ai4bharat/indic-bert").to(device)
    models["gte_tokenizer"] = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True)
    models["gte_model"] = AutoModel.from_pretrained("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True).to(device)
    models["bge_tokenizer"] = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    models["bge_model"] = AutoModel.from_pretrained("BAAI/bge-m3").to(device)
    if USE_SIMALIGN:
        models["aligner"] = SentenceAligner(model="bert", token_type="bpe", matching_methods="mai")  # CPU
    models["device"] = device
    print(f"[GPU SERVER] Models loaded on {device}")
    return models

def get_safe_max_len(model, tokenizer):
    if hasattr(model.config, "max_position_embeddings"):
        max_pos = int(model.config.max_position_embeddings)
    else:
        max_pos = int(getattr(tokenizer, "model_max_length", 512))
    return max(2, max_pos - 2)

def batched_model_encode_with_models(model, tokenizer, texts, device, batch_size=32, use_pooler=True, pooling_fn=None):
    if pooling_fn is None:
        pooling_fn = mean_pooling
    safe_max_len = get_safe_max_len(model, tokenizer)
    all_vecs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True,
                                max_length=safe_max_len, return_tensors='pt')
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            if use_pooler and hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                pooled = outputs.pooler_output
            else:
                pooled = pooling_fn(outputs, encoded['attention_mask'])
            all_vecs.append(pooled.cpu())
    if all_vecs:
        return torch.cat(all_vecs, dim=0).numpy()
    else:
        return np.zeros((len(texts), model.config.hidden_size))

def gpu_get_embeddings(sentences, lang, models, batch_size=DEFAULT_BATCH_SIZE):
    """
    Embedding logic executed on GPU server using loaded 'models' dict.
    """
    n = len(sentences)
    if n == 0:
        return np.zeros((0, 0))

    device = models.get("device", MAIN_DEVICE)

    # SONAR
    if USE_SONAR:
        lang_tag = sonar_language_codes.get(lang.lower(), 'eng_Latn')
        emb_sonar_raw = models["text_embedder"].predict(sentences, source_lang=lang_tag)
        sonars = [e.detach().cpu().numpy() if torch.is_tensor(e) else np.array(e) for e in emb_sonar_raw]
        emb_sonar = np.vstack(sonars)
    else:
        emb_sonar = np.zeros((n, 1024))

    # LaBSE
    labse_raw = models["labse"].encode(
        sentences, convert_to_numpy=True,
        normalize_embeddings=False, batch_size=batch_size
    )
    emb_labse = np.array(labse_raw)

    # IndicBERT
    '''emb_indic = batched_model_encode_with_models(
        models["indic_model"], models["indic_tokenizer"],
        sentences, device, batch_size=batch_size,
        use_pooler=True, pooling_fn=mean_pooling
    )'''

    # GTE
    emb_gte = batched_model_encode_with_models(
        models["gte_model"], models["gte_tokenizer"],
        sentences, device, batch_size=batch_size,
        use_pooler=False, pooling_fn=mean_pooling
    )

    # BGE
    emb_bge = batched_model_encode_with_models(
        models["bge_model"], models["bge_tokenizer"],
        sentences, device, batch_size=batch_size,
        use_pooler=False, pooling_fn=mean_pooling
    )

    final = np.concatenate([emb_sonar, emb_labse, emb_gte, emb_bge], axis=1)
    norms = np.linalg.norm(final, axis=1, keepdims=True) + 1e-9
    final = final / norms

    # Explicit cleanup
    del emb_sonar_raw, sonars, labse_raw, emb_labse, emb_gte, emb_bge
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except:
        pass

    return final

def gpu_server_loop(gpu_id, request_queue, response_store, stop_flag, batch_size=DEFAULT_BATCH_SIZE):
    """
    Process that runs on a specific GPU, loads models there and services embedding requests.
    """
    models = load_models_on_device(gpu_id)
    while not stop_flag.is_set():
        try:
            item = request_queue.get(timeout=0.5)
        except Exception:
            continue
        if item == "__STOP__":
            break
        try:
            req_id = item["req_id"]
            sentences = item["sentences"]
            lang = item["lang"]

            emb = gpu_get_embeddings(sentences, lang, models, batch_size=batch_size)

            response_store[req_id] = emb

            # Extra safety: free memory per request
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except:
                pass

        except Exception as e:
            response_store[req_id] = None
            print(f"[GPU SERVER {gpu_id}] Error processing request {req_id}: {e}")

# -------------------------
# Proxy for embedding calls when in parallel mode
# -------------------------
def embed_via_gpu(sentences, lang, request_queues, response_store, timeout=120.0):
    """
    Round-robin across request_queues (one per GPU server).
    Uses a global semaphore to ensure we do NOT overload GPU with
    concurrent embedding jobs from multiple workers → prevents OOM.
    """
    with GPU_SEMAPHORE:
        req_id = str(uuid.uuid4())
        item = {"req_id": req_id, "sentences": sentences, "lang": lang}

        chosen_q = request_queues[hash(req_id) % len(request_queues)]
        chosen_q.put(item)

        t0 = time.time()
        while True:
            if req_id in response_store:
                res = response_store.pop(req_id)
                # Post-embedding GPU cleanup (extra safety)
                try:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                except:
                    pass
                return res

            if time.time() - t0 > timeout:
                raise TimeoutError(f"Embedding request {req_id} timed out")

            time.sleep(0.01)

# --------------------------------------------------------------------------------
# Sequential get_embeddings (original behavior)
# --------------------------------------------------------------------------------
def get_embeddings_sequential(sentences, lang, batch_size=64):
    n = len(sentences)
    if n == 0:
        return np.zeros((0, 0))

    if USE_SONAR:
        lang_tag = sonar_language_codes.get(lang.lower(), 'eng_Latn')
        emb_sonar_raw = text_embedder_seq.predict(sentences, source_lang=lang_tag)
        sonars = [e.detach().cpu().numpy() if torch.is_tensor(e) else np.array(e) for e in emb_sonar_raw]
        emb_sonar = np.vstack(sonars)
    else:
        emb_sonar = np.zeros((n, 1024))

    labse_raw = labse_model_seq.encode(
        sentences, convert_to_numpy=True,
        normalize_embeddings=False, batch_size=batch_size
    )
    emb_labse = np.array(labse_raw)

    '''emb_indic = batched_model_encode_with_models(
        indic_model_seq, indic_tokenizer_seq, sentences,
        MAIN_DEVICE, batch_size=batch_size,
        use_pooler=True, pooling_fn=mean_pooling
    )'''
    emb_gte = batched_model_encode_with_models(
        gte_model_seq, gte_tokenizer_seq, sentences,
        CURRENT_DEVICE, batch_size=batch_size,
        use_pooler=False, pooling_fn=mean_pooling
    )
    emb_bge = batched_model_encode_with_models(
        bge_model_seq, bge_tokenizer_seq, sentences,
        CURRENT_DEVICE, batch_size=batch_size,
        use_pooler=False, pooling_fn=mean_pooling
    )

    final = np.concatenate([emb_sonar, emb_labse, emb_gte, emb_bge], axis=1)
    norms = np.linalg.norm(final, axis=1, keepdims=True) + 1e-9
    final = final / norms

    del emb_sonar_raw, sonars, labse_raw, emb_labse, emb_gte, emb_bge
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except:
        pass

    return final

def align_sentences_to_paragraph_twostage_unified(
    src_sentences,
    tgt_paragraph,
    src_lang_tag,
    tgt_lang_tag,
    labse_model,
    sonar_model,
    lang,
    span_sizes=((1,4),(4,6),(6,10),(10,18),(18,25)),
    topk_labse=None,
    labse_min=None,
    fused_min=None,
    min_span_chars=3,
    tolerance_words=15,
    confidence_override_min=0.75,
    fuse_weights=(0.45, 0.55),
    show_progress=True
):
    """
    Unified alignment:
    - English sentences search FULL target paragraph (robust)
    - Sentence units are INDUCED from aligned spans
    - Returns:
        1) aligned_pairs        -- final accepted (post-threshold) pairs
        2) full_units           -- aligned + unaligned, audio-ready
        3) all_compared         -- ONE row per English sentence: its single
           best-scoring target span, whether or not it cleared fused_min.
           This is a pure capture of an already-computed score (no extra
           compute), used to build a pre-threshold "all candidate pairs" view.
        4) fused_min            -- the RESOLVED threshold actually used (after
           the None -> low-resource-default fallback above), so callers can
           record it alongside each row for later margin/override checks.
    """

    # --------------------------------------------------
    # 0. Normalize target paragraph
    # --------------------------------------------------
    tgt_paragraph = re.sub(r"\s*\r?\n\s*", " ", tgt_paragraph)
    tgt_paragraph = re.sub(r"\s+", " ", tgt_paragraph).strip()

    # FIX: these three used to be unconditionally overwritten here based on
    # `lang` alone, regardless of what the caller passed in for
    # labse_min/fused_min/topk_labse -- making those function parameters
    # dead. Now the lang-based low-resource defaults only kick in when the
    # caller left the argument at its sentinel (None), so an explicit
    # caller-supplied value is actually respected.
    if labse_min is None:
        labse_min = 0.35 if lang in LOW_RESOURCE_LANGS else 0.48
    if fused_min is None:
        fused_min = 0.45 if lang in LOW_RESOURCE_LANGS else 0.55
    if topk_labse is None:
        topk_labse = 75 if lang in LOW_RESOURCE_LANGS else 25

    # The frontier/tolerance/confidence-override acceptance rule only
    # applies to low-resource languages (the ones actually transcribed via
    # MMS ASR, where translation-order drift + occasional noisy scoring
    # motivated it). High-resource languages have clean, non-ASR
    # transcripts and keep the OLD acceptance rule unchanged: a candidate
    # is accepted purely on fused_min, with no positional gating at all --
    # exactly how this function worked before the tolerance/override
    # rework. Both cases still go through the same overlap-resolution
    # safety-net afterward, which for high-resource IS the old design's
    # actual overlap-resolution mechanism (not just a rare backstop).
    apply_positional_constraints = lang in LOW_RESOURCE_LANGS

    if not tgt_paragraph or not src_sentences:
        return [], [], [], fused_min

    tgt_words = tgt_paragraph.split()
    n_words = len(tgt_words)

    # --------------------------------------------------
    # 1. Build ALL sliding spans over FULL paragraph
    # --------------------------------------------------
    # min_span_chars: drops any candidate span whose joined text is under
    # this many characters -- e.g. a 1-word span that happens to be a lone
    # letter or 2-3 char fragment. This matters most for noisy/corrupted
    # source text (garbled OCR, legacy-font mojibake), where such tiny
    # fragments can otherwise still get embedded and occasionally score
    # high purely by chance, polluting the candidate pool with meaningless
    # matches. Doesn't change behavior for normal well-formed spans (mn>=1
    # word spans of real language are almost always >=3 chars anyway).
    #
    # script_ranges / has_intra_word_script_mixing: drops any candidate
    # span containing a word that mixes the expected script with a
    # DIFFERENT script inside that same word (e.g. "বaজadsম0গa") --
    # the actual signature of MMS ASR near-script noise on languages it
    # has no real model for. Deliberately does NOT reject normal
    # code-switching (a whole word borrowed from English like "exam"
    # sitting on its own) -- only intra-word mixing.
    #
    # Only LOW_RESOURCE_LANGS go through MMS ASR at all -- high-resource
    # languages have clean, non-ASR transcripts, so this filter has
    # nothing to catch there. get_expected_script_ranges(lang) is only
    # called for low-resource languages; None makes
    # has_intra_word_script_mixing() a permanent no-op below, so
    # high-resource spans are completely untouched by this -- old
    # behavior, unchanged.
    script_ranges = get_expected_script_ranges(lang) if lang in LOW_RESOURCE_LANGS else None
    spans = []
    span_ranges = []

    for mn, mx in span_sizes:
        for i in range(n_words - mn + 1):
            max_w = min(mx, n_words - i)
            for w in range(mn, max_w + 1):
                candidate_text = " ".join(tgt_words[i:i+w])
                if len(candidate_text) < min_span_chars:
                    continue
                if has_intra_word_script_mixing(candidate_text, script_ranges):
                    continue
                spans.append(candidate_text)
                span_ranges.append((i, i + w))

    if not spans:
        return [], [], [], fused_min

    # --------------------------------------------------
    # 2. Stage-1 embeddings (LaBSE)
    # --------------------------------------------------
    if show_progress:
        print(f"Embedding {len(spans)} target spans")

    tgt_labse = labse_model.encode(
        spans,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    src_labse = labse_model.encode(
        src_sentences,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # --------------------------------------------------
    # 3. Align each English sentence to paragraph
    #    (sequential frontier walk, shared design with the SONAR-only
    #    function -- see min_start/tolerance_words/confidence_override_min
    #    below)
    # --------------------------------------------------
    aligned_spans = []  # (start, end, eng, score) -- accepted, in narrative order
    all_compared = []   # every English sentence's single best candidate, pre-threshold
    min_start = 0        # narrative frontier -- only ever advances

    iterator = tqdm(
        range(len(src_sentences)),
        desc=f"Aligning English → {lang} paragraph",
        disable=not show_progress
    )

    for i in iterator:
        labse_scores = src_labse[i] @ tgt_labse.T
        # NOTE: topk_labse is a computational shortlist only (which spans
        # even get a SONAR fused score computed), not an acceptance
        # threshold. It is NOT filtered by labse_min here -- that used to
        # mean an English sentence with no span clearing labse_min got NO
        # row at all in all_compared, i.e. the comparison table's contents
        # depended on a threshold. Every source sentence now always gets
        # a "best candidate" comparison row, regardless of labse_min/
        # fused_min; thresholds only decide the "aligned" flag / final
        # accepted output, never whether a comparison is captured.
        candidate_idx = list(np.argpartition(labse_scores, -topk_labse)[-topk_labse:])

        candidate_spans = [spans[idx] for idx in candidate_idx]

        sonar_embs = np.vstack([
            e.detach().cpu().numpy() if hasattr(e, "detach") else np.asarray(e)
            for e in sonar_model.predict(candidate_spans, source_lang=tgt_lang_tag)
        ])
        sonar_embs /= np.linalg.norm(sonar_embs, axis=1, keepdims=True) + 1e-9

        src_emb = sonar_model.predict(
            [src_sentences[i]], source_lang=src_lang_tag
        )[0]
        src_emb = (
            src_emb.detach().cpu().numpy()
            if hasattr(src_emb, "detach")
            else np.asarray(src_emb)
        )
        src_emb /= np.linalg.norm(src_emb) + 1e-9

        raw_best = None  # highest-scoring candidate, ignoring position/threshold entirely
        best = None       # highest-scoring candidate that ALSO clears the position+threshold gate
        for j, idx in enumerate(candidate_idx):
            fused = (
                fuse_weights[0] * float(labse_scores[idx]) +
                fuse_weights[1] * float(src_emb @ sonar_embs[j])
            )
            s, e = span_ranges[idx]

            if raw_best is None or fused > raw_best[3]:
                raw_best = (s, e, src_sentences[i], fused, i)

            # Position + threshold gate -- LOW-RESOURCE ONLY. Candidates
            # within tolerance_words of the current frontier need only
            # clear the normal fused_min bar. Candidates further "behind"
            # the frontier are still allowed through, but ONLY if they're
            # confident enough (>= confidence_override_min) that being
            # wrong is unlikely -- a large jump backward in the narrative
            # is a real correctness risk for downstream audio splitting.
            #
            # HIGH-RESOURCE: old behavior, unchanged -- pure fused_min
            # threshold, no positional gating whatsoever (clean transcripts
            # don't need this; overlap-resolution below is what did, and
            # still does, all the work for these languages).
            if apply_positional_constraints:
                if s >= min_start - tolerance_words:
                    passes = fused >= fused_min
                else:
                    passes = fused >= confidence_override_min
            else:
                passes = fused >= fused_min

            if not passes:
                continue
            if best is None or fused > best[3]:
                best = (s, e, src_sentences[i], fused, i)

        # Capture exactly one row for this English sentence: the REAL
        # accepted span (best) if one exists -- with its own true score,
        # never raw_best's -- otherwise the raw highest-scoring candidate,
        # marked aligned=False. Existence of this row never depends on any
        # threshold; only which span is shown and the aligned flag do.
        # "aligned" here is a placeholder -- finalized after the overlap
        # safety-net below, since tolerance can (rarely) let two accepted
        # spans overlap.
        if best is not None:
            s_b, e_b, eng_b, score_b, idx_b = best
            all_compared.append({
                lang: " ".join(tgt_words[s_b:e_b]),
                "English": eng_b,
                "english_idx": idx_b,
                "similarity": round(float(score_b), 4),
                "raw_similarity": round(float(score_b), 4),
                "aligned": False,  # placeholder; finalized after overlap safety-net
                "stage": "sliding_fused",
                "_span_key": (s_b, e_b, eng_b, idx_b),
            })
            aligned_spans.append(best)
            min_start = max(min_start, best[1])  # frontier only ever advances
        elif raw_best is not None:
            s_b, e_b, eng_b, score_b, idx_b = raw_best
            all_compared.append({
                lang: " ".join(tgt_words[s_b:e_b]),
                "English": eng_b,
                "english_idx": idx_b,
                "similarity": round(float(score_b), 4),
                "raw_similarity": round(float(score_b), 4),
                "aligned": False,
                "stage": "sliding_fused",
                "_span_key": (s_b, e_b, eng_b, idx_b),
            })

    # --------------------------------------------------
    # 3b. Overlap safety-net (rare). The sequential frontier walk above
    # already prevents most overlaps by construction -- but
    # tolerance_words allows a candidate to start "behind" the frontier,
    # so in rare cases two accepted spans can still end up overlapping in
    # word-range. Resolve any such conflict by keeping whichever scored
    # higher, same greedy-by-score approach the old global-only design
    # used everywhere -- now only needed as a backstop, not the primary
    # mechanism.
    # --------------------------------------------------
    candidates = sorted(aligned_spans, key=lambda x: x[3], reverse=True)
    accepted = []
    occupied = []  # list of (start, end) already claimed, kept sorted-free (small N is fine)

    def overlaps(s, e):
        for os_, oe_ in occupied:
            if s < oe_ and os_ < e:
                return True
        return False

    for s, e, eng, score, idx in candidates:
        if overlaps(s, e):
            continue
        accepted.append((s, e, eng, score, idx))
        occupied.append((s, e))

    aligned_spans = accepted

    # ---- Now that the safety-net is final, set the REAL aligned flag on
    # all_compared: True iff this exact (span, English sentence, index)
    # survived into `accepted`. ----
    accepted_span_keys = {(s, e, eng, idx) for s, e, eng, score, idx in accepted}
    for row in all_compared:
        row["aligned"] = row.pop("_span_key") in accepted_span_keys
        row["similarity"] = display_similarity(row["raw_similarity"], row["aligned"])

    # --------------------------------------------------
    # 4. Induce sentence units from aligned spans
    # --------------------------------------------------
    aligned_spans = sorted(aligned_spans, key=lambda x: x[0])
    full_units = []
    aligned_pairs = []

    last = 0
    for s, e, eng, score, idx in aligned_spans:
        if s > last:
            full_units.append({
                "English": None,
                lang: " ".join(tgt_words[last:s]),
                "similarity": None,
                "aligned": False
            })

        segment = " ".join(tgt_words[s:e])
        full_units.append({
            "English": eng,
            lang: segment,
            "similarity": round(score, 4),
            "aligned": True
        })
        aligned_pairs.append({
            "English": eng,
            "english_idx": idx,
            lang: segment,
            "similarity": round(score, 4)
        })

        last = e

    if last < n_words:
        full_units.append({
            "English": None,
            lang: " ".join(tgt_words[last:]),
            "similarity": None,
            "aligned": False
        })

    # ---- Coverage check ("see through"): full_units is built by walking
    # tgt_words[0:n_words] exactly once (accepted spans + the gaps between
    # them), so the word count across every segment here must equal
    # n_words exactly -- if it doesn't, some part of the paragraph is being
    # silently skipped or double-counted. This is a structural invariant,
    # not a threshold-dependent one, so it should never fire; if it does,
    # something upstream (e.g. overlapping "accepted" spans that slipped
    # past overlap resolution) needs investigating. ----
    reconstructed_words = sum(len(u[lang].split()) for u in full_units if u.get(lang))
    if reconstructed_words != n_words:
        tqdm.write(
            f"⚠️ {lang}: full_units covers {reconstructed_words}/{n_words} paragraph words "
            f"-- some content may be missing or double-counted."
        )

    return aligned_pairs, full_units, all_compared, fused_min
    
       
# -------------------------
# similarity & alignment
# -------------------------
SCRIPT_UNICODE_RANGES = {
    "Deva": [(0x0900, 0x097F)],   # Devanagari (Hindi, Marathi, etc.)
    "Beng": [(0x0980, 0x09FF)],   # Bengali (also used for Manipuri here)
    "Taml": [(0x0B80, 0x0BFF)],   # Tamil
    "Telu": [(0x0C00, 0x0C7F)],   # Telugu
    "Knda": [(0x0C80, 0x0CFF)],   # Kannada
    "Mlym": [(0x0D00, 0x0D7F)],   # Malayalam
    "Guru": [(0x0A00, 0x0A7F)],   # Gurmukhi (Punjabi)
    "Gujr": [(0x0A80, 0x0AFF)],   # Gujarati
    "Orya": [(0x0B00, 0x0B7F)],   # Oriya
    "Arab": [(0x0600, 0x06FF), (0x0750, 0x077F)],  # Arabic (Urdu, Kashmiri, etc.)
    "Latn": [(0x0041, 0x005A), (0x0061, 0x007A)],  # basic Latin letters
}


def get_expected_script_ranges(lang):
    """Resolves lang -> its expected Unicode script ranges, from the
    _Xxxx suffix already present in sonar_language_codes (e.g.
    "ben_Beng" -> Beng). Returns None for languages whose expected script
    IS Latin (English, and the several LOW_RESOURCE_LANGS entries mapped
    to "eng_Latn" already) or anything unrecognized -- no filtering
    applied in either case, since there's nothing meaningful to compare
    against for those."""
    code = sonar_language_codes.get(lang)
    if not code or "_" not in code:
        return None
    script = code.split("_")[-1]
    if script == "Latn":
        return None
    return SCRIPT_UNICODE_RANGES.get(script)


def has_intra_word_script_mixing(text, script_ranges):
    """
    True if ANY single word in `text` mixes the expected script with a
    DIFFERENT script (typically Latin letters/digits) WITHIN that same
    word -- e.g. "বaজadsম0গaষaঙ6c" (Bengali + Latin + digits interleaved
    in one token). This is the actual signature of MMS ASR near-script
    noise on languages it has no real model for (per-word phonetic
    guesses using whatever characters, not real vocabulary).

    Deliberately NOT flagging normal code-switching -- a whole word
    legitimately borrowed from English (e.g. "exam", "App", "positive")
    sitting on its own between real target-script words is completely
    normal in these transcripts and must NOT be rejected. The distinction
    is per-word: does a SINGLE token mix scripts internally, vs. is a
    WHOLE token from a different script. Only the former is flagged.
    """
    if not script_ranges:
        return False
    for word in text.split():
        has_expected = False
        has_other_alpha = False
        for ch in word:
            if not ch.isalpha():
                continue
            cp = ord(ch)
            if any(lo <= cp <= hi for lo, hi in script_ranges):
                has_expected = True
            else:
                has_other_alpha = True
        if has_expected and has_other_alpha:
            return True
    return False


def display_similarity(raw_similarity, aligned, band=0.5):
    """
    Cosmetic-only rescaling for the "similarity" column shown in
    all_compared / all_pairs_before_threshold / sliding_only / topk_only:
    a row marked aligned=False gets its shown similarity compressed into
    [0, band] (default [0, 0.5]) regardless of its raw cosine value, so a
    REJECTED candidate that happens to have a high raw score (e.g. it lost
    the mutual/overlap/max_usage tiebreak, not because it was a bad
    match -- see the Manipuri 0.8872 case earlier in this pipeline's
    history) doesn't visually read as "this should have been aligned".

    - Ordering among unaligned rows is preserved (still monotonic in
      raw_similarity), so relative comparisons among rejects still work.
    - aligned=True rows are returned UNCHANGED -- this never touches an
      accepted pair's real score.
    - The uncompressed value is ALWAYS also kept, under "raw_similarity",
      wherever this is used -- nothing is lost, this only changes what
      "similarity" shows for rejected rows.
    """
    raw = float(raw_similarity)
    if aligned:
        return round(raw, 4)
    return round(raw * band, 4)


def cosine_sim_matrix(A, B):
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]))
    a = torch.from_numpy(A).to(CURRENT_DEVICE)
    b = torch.from_numpy(B).to(CURRENT_DEVICE)
    sims = torch.matmul(a, b.t())
    return sims.cpu().numpy()

LOW_RESOURCE_TOPK_THRESHOLD = 0.45  # real, lower bar for low-resource langs.
# NOTE: earlier versions of this pipeline BYPASSED the threshold check
# entirely for low-resource languages (`row[j] >= threshold or lang in
# LOW_RESOURCE_LANGS`), which meant forward/backward top-k sets were
# populated no matter how weak the match was. That made the "aligned" flag
# depend almost entirely on reciprocal-top-1 tie-breaking rather than on
# any real similarity floor, and produced lots of high-similarity-but-
# unaligned rows whenever multiple candidates on one side collided onto
# the same sentence on the other side. Using a real (lower) threshold here
# instead keeps the low-resource case more permissive than the default,
# without making similarity meaningless.

def align_sentences_mutual_topk(
    sentences1,
    sentences2,
    lang1,
    lang2="English",
    threshold=0.75,
    top_k=3,
    embed_fn=None,
    low_resource_threshold=LOW_RESOURCE_TOPK_THRESHOLD,
    return_all_compared=False,
    max_usage=1,
    sentence2_original_indices=None,
):
    """
    Mutual (forward–backward) Top-K sentence alignment.

    A pair (i, j) is accepted iff:
      - j ∈ TopK(i → sentences2)
      - i ∈ TopK(j → sentences1)
      - neither sentences1[i] nor sentences2[j] has already been used
        `max_usage` times by a HIGHER-scoring accepted pair (see below)

    For languages in LOW_RESOURCE_LANGS, `low_resource_threshold` is used in
    place of `threshold` -- a genuinely lower bar, not a bypass.

    `max_usage` (previously a dead CLI flag in every version of this
    pipeline -- declared but never read) caps how many times any single
    sentence on either side can appear across the FINAL accepted pairs.
    This directly limits the "attractor sentence" failure mode where many
    unrelated candidates all point at the same sentence: even if several
    pairs pass the mutual/reciprocal test, only the `max_usage` highest-
    scoring ones per sentence are kept.

    sentence2_original_indices: maps each LOCAL index j (position within
    THIS call's `sentences2`) to its TRUE position in the full english_sents
    list for the episode. Needed because the leftover-topk leg in
    process_episode calls this with `sentences2` already filtered down to
    only the English sentences sliding missed -- so j=0 there is NOT
    English sentence #0 of the episode. Defaults to identity (j itself)
    for the ordinary full-topk call, where sentences2 IS english_sents.
    Every output row's "english_idx" uses this mapped, true index -- so
    collapse_to_best_per_english() can group by real sentence position
    instead of by text (which silently collapses two DIFFERENT sentence
    occurrences that happen to share identical wording).

    If return_all_compared=True, also returns all_compared: every pair that
    was ACTUALLY accepted into aligned_pairs (aligned=True, guaranteed
    present here), PLUS each remaining sentence's single best candidate on
    either side (row-best/column-best) for diagnosing rejections -- a
    pre-threshold view of every candidate pair seriously considered.
    """

    if embed_fn is None:
        embed_fn = get_embeddings_sequential

    if sentence2_original_indices is None:
        sentence2_original_indices = list(range(len(sentences2)))

    effective_threshold = (
        low_resource_threshold if lang1 in LOW_RESOURCE_LANGS else threshold
    )

    # ---- Embed once ----
    emb1 = embed_fn(sentences1, lang1)
    emb2 = embed_fn(sentences2, lang2)

    sims = cosine_sim_matrix(emb1, emb2)
    n1, n2 = sims.shape

    # ---- Forward Top-K: src → tgt ----
    forward = defaultdict(set)
    for i in range(n1):
        row = sims[i]
        for j in np.argsort(-row)[:top_k]:
            if row[j] >= effective_threshold:
                forward[i].add(j)

    # ---- Backward Top-K: tgt → src ----
    backward = defaultdict(set)
    for j in range(n2):
        col = sims[:, j]
        for i in np.argsort(-col)[:top_k]:
            if col[i] >= effective_threshold:
                backward[j].add(i)

    # ---- Mutual agreement, then cap reuse via max_usage ----
    mutual_candidates = []
    for i in range(n1):
        for j in forward[i]:
            if i in backward[j]:
                mutual_candidates.append((i, j, float(sims[i, j])))

    mutual_candidates.sort(key=lambda t: t[2], reverse=True)

    usage1 = defaultdict(int)
    usage2 = defaultdict(int)
    aligned_pairs = []
    accepted_keys = set()  # (i, j) pairs that actually survived max_usage capping
    for i, j, score in tqdm(mutual_candidates, desc=f"Mutual Aligning {lang1} ↔ {lang2}"):
        if usage1[i] >= max_usage or usage2[j] >= max_usage:
            continue
        usage1[i] += 1
        usage2[j] += 1
        aligned_pairs.append({
            lang1: sentences1[i], lang2: sentences2[j],
            "english_idx": sentence2_original_indices[j],
            "similarity": score
        })
        accepted_keys.add((i, j))

    if not return_all_compared:
        return aligned_pairs

    # ---- Build all_compared: exactly ONE row per sentences2 (English)
    # sentence -- a sheer, threshold-free comparison, not a decision.
    #
    # FIX: this used to also loop over sentences1 (row-best) AND
    # sentences2 (col-best) and union them, so all_compared could have up
    # to n1 + n2 rows -- far more (or, after dedup collisions, sometimes
    # fewer) than the number of English sentences. That made "one row per
    # English sentence, independent of threshold" impossible to rely on.
    # Now: for each j, if it has a REAL accepted partner (survived the
    # mutual + max_usage test above), show that pair with aligned=True and
    # its true similarity -- guaranteeing every real accepted pair is
    # visible. Otherwise, show j's raw column-argmax partner (the best
    # match by cosine alone, no threshold gating on whether this row
    # exists) with aligned=False. Either way: exactly n2 rows, similarity
    # is always the real cosine (never an averaged/adjusted value), and
    # "aligned" is decided ONLY by the actual mutual+max_usage outcome
    # computed above -- not recomputed here from forward/backward sets.
    accepted_by_col = {}
    for i, j, score in mutual_candidates:
        if (i, j) not in accepted_keys:
            continue
        if j not in accepted_by_col or score > accepted_by_col[j][1]:
            accepted_by_col[j] = (i, score)

    col_best_idx = np.argmax(sims, axis=0) if n1 > 0 else []

    all_compared = []
    for j in range(n2):
        if j in accepted_by_col:
            i, score = accepted_by_col[j]
            all_compared.append({
                lang1: sentences1[i],
                lang2: sentences2[j],
                "english_idx": sentence2_original_indices[j],
                "similarity": display_similarity(score, aligned=True),
                "raw_similarity": round(float(score), 4),
                "aligned": True,
                "stage": "mutual_topk_cosine"
            })
        else:
            i = int(col_best_idx[j])
            raw = float(sims[i, j])
            all_compared.append({
                lang1: sentences1[i],
                lang2: sentences2[j],
                "english_idx": sentence2_original_indices[j],
                "similarity": display_similarity(raw, aligned=False),
                "raw_similarity": round(raw, 4),
                "aligned": False,
                "stage": "mutual_topk_cosine"
            })

    return aligned_pairs, all_compared, effective_threshold
   
def align_sentences_to_paragraph_twostage_unified_sonar_paragraph_aligned(
    src_sentences,
    tgt_paragraph,
    src_lang_tag,
    tgt_lang_tag,
    sonar_model,
    lang,
    span_sizes=((1,4),(4,6),(6,10),(10,18),(18,25)),
    topk_sonar=60,
    sonar_min=0.45,
    min_span_chars=3,
    tolerance_words=15,
    confidence_override_min=0.5,
    show_progress=True
):
    """
    Unified SONAR-only alignment (no LaBSE fusion) -- brought back into use
    for Manipuri, since the fused LaBSE+SONAR path wasn't producing better
    samples for it.
    - English sentences search FULL target paragraph
    - Sentence units are INDUCED from aligned spans
    - Returns:
        1) aligned_pairs        -- final accepted (post-threshold) pairs
        2) full_units           -- aligned + unaligned, audio-ready
        3) all_compared         -- ONE row per English sentence: if a span
           was actually accepted (survived sonar_min + monotonicity), shows
           THAT span's own true score with aligned=True -- never the raw
           highest-scoring candidate's score when the two differ (that was
           the original bug in this function: logging raw_best under an
           aligned=True label even when a different, lower-scoring span
           was the one actually accepted). If nothing was accepted, shows
           the raw highest-scoring candidate with aligned=False. Either
           way: exactly len(src_sentences) rows, existence never gated by
           sonar_min.
        4) sonar_min             -- the threshold actually used, so callers
           can attach it to rows for margin/override checks.
    """

    # --------------------------------------------------
    # 0. Normalize target paragraph
    # --------------------------------------------------
    tgt_paragraph = re.sub(r"\s*\r?\n\s*", " ", tgt_paragraph)
    tgt_paragraph = re.sub(r"\s+", " ", tgt_paragraph).strip()

    if not tgt_paragraph or not src_sentences:
        return [], [], [], sonar_min

    tgt_words = tgt_paragraph.split()
    n_words = len(tgt_words)

    # --------------------------------------------------
    # 1. Build ALL sliding spans over FULL paragraph
    # --------------------------------------------------
    # min_span_chars: same mitigation as the fused function -- drops any
    # candidate span under this many characters, so degenerate 1-2 char
    # fragments (a lone letter, garbled/corrupted-text remnants) never
    # become alignment candidates in the first place.
    #
    # script_ranges / has_intra_word_script_mixing: same as the fused
    # function -- drops spans containing a word that mixes the expected
    # script with a different one WITHIN that word (MMS near-script ASR
    # noise), without rejecting normal whole-word code-switching. Scoped
    # to LOW_RESOURCE_LANGS same as the fused function, for consistency
    # (this function is currently only ever called for Manipuri, which is
    # low-resource, but the explicit check keeps both functions aligned).
    script_ranges = get_expected_script_ranges(lang) if lang in LOW_RESOURCE_LANGS else None
    spans = []
    span_ranges = []

    for mn, mx in span_sizes:
        for i in range(n_words - mn + 1):
            max_w = min(mx, n_words - i)
            for w in range(mn, max_w + 1):
                candidate_text = " ".join(tgt_words[i:i+w])
                if len(candidate_text) < min_span_chars:
                    continue
                if has_intra_word_script_mixing(candidate_text, script_ranges):
                    continue
                spans.append(candidate_text)
                span_ranges.append((i, i + w))

    if not spans:
        return [], [], [], sonar_min

    # --------------------------------------------------
    # 2. Embed ALL spans + source sentences (SONAR)
    # --------------------------------------------------
    if show_progress:
        print(f"Embedding {len(spans)} target spans with SONAR")

    tgt_embs = np.vstack([
        e.detach().cpu().numpy() if hasattr(e, "detach") else np.asarray(e)
        for e in sonar_model.predict(spans, source_lang=tgt_lang_tag)
    ])
    tgt_embs /= np.linalg.norm(tgt_embs, axis=1, keepdims=True) + 1e-9

    src_embs = np.vstack([
        e.detach().cpu().numpy() if hasattr(e, "detach") else np.asarray(e)
        for e in sonar_model.predict(src_sentences, source_lang=src_lang_tag)
    ])
    src_embs /= np.linalg.norm(src_embs, axis=1, keepdims=True) + 1e-9

    # --------------------------------------------------
    # 3. Align each English sentence to paragraph
    #    (MONOTONIC TARGET SPAN SELECTION)
    # --------------------------------------------------
    aligned_spans = []  # (start, end, eng, score) -- only those that passed sonar_min + monotonicity
    all_compared = []   # exactly one row per English sentence, pre-threshold
    min_start = 0       # enforces paragraph order

    iterator = tqdm(
        range(len(src_sentences)),
        desc=f"Aligning English → {lang} paragraph (SONAR)",
        disable=not show_progress
    )

    for i in iterator:
        scores = src_embs[i] @ tgt_embs.T
        top_idx = np.argpartition(scores, -topk_sonar)[-topk_sonar:]

        best = None
        raw_best = None  # highest-similarity candidate, ignoring sonar_min/position constraint
        for idx in top_idx:
            score = scores[idx]
            s, e = span_ranges[idx]

            if raw_best is None or score > raw_best[3]:
                raw_best = (s, e, src_sentences[i], float(score), i)

            # Position + threshold gate, shared design with the fused
            # function: within tolerance_words of the frontier, the normal
            # sonar_min bar applies. Further "behind" than that, only a
            # confidently high score (>= confidence_override_min) is
            # allowed through -- protects downstream audio-split ordering
            # from a noisy embedding accidentally jumping far backward,
            # while still recovering genuine small reorderings that a
            # strict monotonic rule would have rejected outright.
            if s >= min_start - tolerance_words:
                passes = score >= sonar_min
            else:
                passes = score >= confidence_override_min

            if not passes:
                continue

            if best is None or score > best[3]:
                best = (s, e, src_sentences[i], float(score), i)

        # Capture exactly one row for this English sentence: the REAL
        # accepted span (best) if one exists -- with its own true score,
        # never raw_best's -- otherwise the raw highest-scoring candidate,
        # marked aligned=False. Existence of this row never depends on
        # sonar_min; only the "aligned" flag and which span is shown does.
        # "aligned" here is a placeholder -- finalized after the overlap
        # safety-net below (tolerance can rarely let two accepted spans
        # overlap).
        if best is not None:
            s_b, e_b, eng_b, score_b, idx_b = best
            all_compared.append({
                lang: " ".join(tgt_words[s_b:e_b]),
                "English": eng_b,
                "english_idx": idx_b,
                "similarity": round(float(score_b), 4),
                "raw_similarity": round(float(score_b), 4),
                "aligned": False,  # placeholder; finalized after overlap safety-net
                "stage": "sliding_sonar",
                "_span_key": (s_b, e_b, eng_b, idx_b),
            })
            aligned_spans.append(best)
            min_start = max(min_start, best[1])  # frontier only ever advances
        elif raw_best is not None:
            s_b, e_b, eng_b, score_b, idx_b = raw_best
            all_compared.append({
                lang: " ".join(tgt_words[s_b:e_b]),
                "English": eng_b,
                "english_idx": idx_b,
                "similarity": display_similarity(score_b, aligned=False),
                "raw_similarity": round(float(score_b), 4),
                "aligned": False,
                "stage": "sliding_sonar"
            })

    # --------------------------------------------------
    # 3b. Overlap safety-net (rare). tolerance_words can occasionally let
    # a candidate be accepted "behind" the frontier, so in rare cases two
    # accepted spans end up overlapping in word-range. Resolve any such
    # conflict by keeping whichever scored higher -- same greedy-by-score
    # approach the fused function uses, only needed as a backstop here
    # since strict monotonicity alone used to make this impossible.
    # --------------------------------------------------
    candidates = sorted(aligned_spans, key=lambda x: x[3], reverse=True)
    accepted = []
    occupied = []

    def overlaps(s, e):
        for os_, oe_ in occupied:
            if s < oe_ and os_ < e:
                return True
        return False

    for s, e, eng, score, idx in candidates:
        if overlaps(s, e):
            continue
        accepted.append((s, e, eng, score, idx))
        occupied.append((s, e))

    aligned_spans = accepted

    accepted_span_keys = {(s, e, eng, idx) for s, e, eng, score, idx in accepted}
    for row in all_compared:
        if "_span_key" not in row:
            continue  # already-finalized aligned=False rows (raw_best only) skip this
        row["aligned"] = row.pop("_span_key") in accepted_span_keys
        row["similarity"] = display_similarity(row["raw_similarity"], row["aligned"])

    # --------------------------------------------------
    # 4. Induce sentence units from aligned spans
    # --------------------------------------------------
    aligned_spans.sort(key=lambda x: x[0])

    full_units = []
    aligned_pairs = []

    last = 0
    for s, e, eng, score, idx in aligned_spans:
        if s > last:
            full_units.append({
                "English": None,
                lang: " ".join(tgt_words[last:s]),
                "similarity": None,
                "aligned": False
            })

        segment = " ".join(tgt_words[s:e])
        full_units.append({
            "English": eng,
            lang: segment,
            "similarity": round(score, 4),
            "aligned": True
        })

        aligned_pairs.append({
            "English": eng,
            "english_idx": idx,
            lang: segment,
            "similarity": round(score, 4)
        })

        last = e

    if last < n_words:
        full_units.append({
            "English": None,
            lang: " ".join(tgt_words[last:]),
            "similarity": None,
            "aligned": False
        })

    # ---- Coverage check ("see through"): same invariant as the fused
    # sliding function -- full_units walks tgt_words[0:n_words] exactly
    # once, so its total word count must equal n_words. ----
    reconstructed_words = sum(len(u[lang].split()) for u in full_units if u.get(lang))
    if reconstructed_words != n_words:
        tqdm.write(
            f"⚠️ {lang}: full_units covers {reconstructed_words}/{n_words} paragraph words "
            f"-- some content may be missing or double-counted."
        )

    return aligned_pairs, full_units, all_compared, sonar_min


def save_list_column_to_lines(df, column, out_path, episode):
    """
    TSV output now: sentence_id<TAB>text (one line per sentence), instead
    of bare text lines. `column` doubles as the language key for
    make_sentence_id() -- it's "English" for the English column and the
    target language name for every other column in merged_df, which is
    exactly the id scheme's expected `lang` argument.

    Any downstream reader of these files needs a one-line change: split
    each line on "\t" and keep both fields, instead of treating the whole
    line as the sentence text.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in df[column].dropna().drop_duplicates():
            if isinstance(row, list):
                for item in row:
                    sid = make_sentence_id(column, episode, item)
                    f.write(f"{sid}\t{sanitize_for_tsv_field(item)}\n")
            else:
                sid = make_sentence_id(column, episode, row)
                f.write(f"{sid}\t{sanitize_for_tsv_field(row)}\n")

def normalize_final_pairs(merged_dicts, lang):
    """
    Final safety filter + text normalization before saving the real
    parallel-corpus output -- but, unlike the old normalisation_as_list /
    normalize_alignment_dicts_to_tuples pair, this KEEPS "aligned"/"stage"
    on each row instead of collapsing everything down to a bare
    (src, English, similarity) tuple. Those two columns are needed both
    for the {lang}_aligned.xlsx column spec (which includes aligned/stage).
    """
    src_lang_code = language_codes.get(lang, None)
    final_pairs = []

    for r in merged_dicts:
        if not isinstance(r, dict):
            continue
        src = r.get(lang)
        eng = r.get("English")
        sim = r.get("similarity")
        if not src or not eng or sim is None:
            continue

        try:
            normalized_src = normalize_text_new(src, src_lang_code) if src_lang_code else src
        except Exception:
            normalized_src = src

        final_pairs.append({
            lang: normalized_src,
            "English": eng,
            "similarity": float(sim),
            "aligned": bool(r.get("aligned", True)),
            "stage": r.get("stage"),
        })

    return final_pairs

STANDARD_ALIGNMENT_COLUMNS_TEMPLATE = [
    "{lang}", "English", "similarity", "raw_similarity", "aligned", "stage",
    "{lang}_sentence_id", "English_sentence_id"
]


def safe_write_alignment_excel(df, lang, out_path, episode=None, default_stage=None):
    """
    Writes ONE normalized Excel schema for every alignment output file
    (sliding_only, topk_only, aligned, all_pairs_before_threshold), so a
    downstream reader never has to special-case which file it opened:

        {lang}  English  similarity  aligned  stage
        {lang}_sentence_id  English_sentence_id

    - "aligned"/"stage" are filled in with `default_stage`/True when the
      caller's df doesn't already carry them (sliding_only/topk_only rows
      ARE that stage's own already-accepted output, so aligned=True is a
      fact there, not a guess).
    - sentence_id columns are computed the same, plain way as everywhere
      else in the pipeline (make_sentence_id(lang, episode, text), no
      stage tagging) -- so the same sentence always gets the same id
      whether it's read from this Excel file, split_sentences.txt's
      sidecar, or the aligned/unaligned audio sidecars.
    """
    required = {"English", lang, "similarity"}
    if not required.issubset(df.columns) or df.empty:
        tqdm.write(
            f"⚠️ Skipped Excel write for {lang} "
            f"(missing columns: {required - set(df.columns)})"
        )
        return

    df = df.copy()
    if "stage" not in df.columns:
        df["stage"] = default_stage
    if "aligned" not in df.columns:
        df["aligned"] = True
    if "raw_similarity" not in df.columns:
        df["raw_similarity"] = df["similarity"]

    if episode is not None:
        df[f"{lang}_sentence_id"] = df.apply(
            lambda r: make_sentence_id(lang, episode, r[lang]),
            axis=1
        )
        df["English_sentence_id"] = df.apply(
            lambda r: make_sentence_id("English", episode, r["English"]),
            axis=1
        )

    cols = [c.format(lang=lang) for c in STANDARD_ALIGNMENT_COLUMNS_TEMPLATE if c.format(lang=lang) in df.columns]
    df[cols].to_excel(out_path, index=False)


def collapse_to_best_per_english(pairs, lang, allow_topk_override_weak_sliding=False,
                                  low_confidence_margin=0.10, return_all_candidates=False):
    """
    Keep exactly one best row per English sentence -- with an optional
    low-confidence override, and an optional side-channel returning EVERY
    candidate row (not just the winner) for auditing.

    Default tie-break (when both stages produced an aligned=True row for
    the same English sentence): sliding wins, since it was inserted first
    in the caller's `sliding_dicts + topk_dicts` list and the two
    similarity scales (fused LaBSE+SONAR vs. plain cosine) aren't directly
    comparable.

    --allow_topk_override_weak_sliding changes that ONE case: if sliding's
    own margin above its OWN threshold -- (similarity - threshold) /
    threshold -- is below low_confidence_margin, and topk ALSO produced an
    aligned=True row for the same sentence, topk's row wins instead. This
    never fires when only one stage has a row for a sentence (existence
    alone still decides those cases, same as before), and it never
    compares raw similarity across stages -- only each stage's own score
    against its own threshold.

    Returns a (best_rows, all_candidate_rows) tuple. all_candidate_rows is
    None unless return_all_candidates=True, in which case it contains
    EVERY row passed in (not just winners), each tagged with
    "is_selected_best" -- for --save_all_candidates' both_candidates.xlsx.
    """
    def margin_ratio(row):
        thr = row.get("threshold")
        sim = row.get("similarity")
        if not thr or sim is None:
            return None
        return (sim - thr) / thr

    def is_weak(row):
        m = margin_ratio(row)
        return m is not None and m < low_confidence_margin

    def better(new, old):
        if old is None:
            return True
        new_aligned = bool(new.get("aligned", False))
        old_aligned = bool(old.get("aligned", False))
        # Prefer aligned rows
        if new_aligned != old_aligned:
            return new_aligned
        if allow_topk_override_weak_sliding and new_aligned and old_aligned:
            # Weak sliding already held, topk now agrees -> topk overrides.
            if str(old.get("stage", "")).startswith("sliding") and new.get("stage") == "mutual_topk_cosine" and is_weak(old):
                return True
            # Topk already held, weak sliding just arrived -> topk stays.
            if str(new.get("stage", "")).startswith("sliding") and old.get("stage") == "mutual_topk_cosine" and is_weak(new):
                return False
        # Compare similarity only within the same stage
        if new.get("stage") == old.get("stage"):
            return (new.get("similarity") or 0) > (old.get("similarity") or 0)
        # Different stage, no override triggered -> keep existing (arbitrary
        # but stable; avoids comparing incomparable similarity scales)
        return False

    best_by_eng = {}
    all_by_eng = defaultdict(list) if return_all_candidates else None

    for p in pairs:
        # FIX: grouping used to be by English TEXT (p.get("English")) --
        # a plain dict key, so two DIFFERENT positions in english_sents
        # that happen to share identical wording (a repeated phrase) would
        # collapse into ONE row here, even though they're two distinct
        # sentence occurrences that need their own alignment row (and,
        # for audio splitting, their own timestamp). Now keyed by
        # english_idx (the sentence's actual position), which every
        # producer (both sliding functions, both topk legs) now tags on
        # every row. Falls back to text only if a row is somehow missing
        # english_idx (shouldn't happen with current producers, but keeps
        # this from silently dropping a row if it ever does).
        eng_key = p.get("english_idx")
        if eng_key is None:
            eng_key = p.get("English")
        if eng_key is None:
            continue
        if return_all_candidates:
            all_by_eng[eng_key].append(p)
        if better(p, best_by_eng.get(eng_key)):
            best_by_eng[eng_key] = p

    best_rows = list(best_by_eng.values())

    if not return_all_candidates:
        return best_rows, None

    all_candidate_rows = []
    for eng, rows in all_by_eng.items():
        winner = best_by_eng.get(eng)
        for r in rows:
            all_candidate_rows.append({**r, "is_selected_best": r is winner})

    return best_rows, all_candidate_rows


# -------------------------
# Episode-level processing
# -------------------------
def process_episode(episode, args, embed_fn=None):
    text_dir = args.text_dir
    output_dir = args.output_dir
    threshold = args.threshold

    all_languages = sorted([l for l in os.listdir(text_dir) if os.path.isdir(os.path.join(text_dir, l))])

    # ------------------------------------------------------------
    # FIX: earlier variants of this pipeline hardcoded which languages
    # get processed (e.g. silently `continue`-ing past anything not in
    # LOW_RESOURCE_LANGS). That's a correctness footgun -- mainstream
    # languages would vanish from the output with no warning. Language
    # scope is now an explicit, opt-in CLI choice:
    #   --languages en,hi,...      -> only process these (by folder name)
    #   --low_resource_only        -> only process LOW_RESOURCE_LANGS + English
    #   (default)                  -> process every language folder found
    # ------------------------------------------------------------
    if args.languages:
        wanted = {x.strip() for x in args.languages.split(",") if x.strip()}
        languages = [l for l in all_languages if l in wanted or l == "English"]
    elif args.low_resource_only:
        languages = [l for l in all_languages if l in LOW_RESOURCE_LANGS or l == "English"]
    else:
        languages = all_languages

    texts = {}
    split_sentences_dict = {}

    for lang in languages:
        file_path = os.path.join(args.text_dir, lang, episode, f"{episode}.txt")
        if os.path.exists(file_path):
            texts[lang] = read_file(file_path)
            if lang=="Assamese":
                texts[lang] = texts[lang].replace("_", " ")
        else:
            tqdm.write(f"⚠️ Missing file for {lang}/{episode}: {file_path}")

    for lang, raw_text in texts.items():
        try:
            lang_code = language_codes.get(lang, None)
            # Low-resource / noisy-transcript languages (often ASR/whisper
            # output without reliable punctuation) get lenient splitting;
            # everything else keeps the stricter, buffered splitting that
            # assumes reasonably punctuated text.
            if lang != "English" and lang in LOW_RESOURCE_LANGS:
                # FIX: this used to be min_len=0 with no floor at all -- so a
                # lone letter or a 2-3 character ASR fragment survived as its
                # own "sentence" (only pure-punctuation strings like "." were
                # caught). min_chars=3 drops those without requiring
                # multiple words (which bumping min_len itself would do,
                # risking real single-word utterances in noisy transcripts).
                sents = split_sentences(raw_text, lang_code=lang_code, min_len=0, min_chars=3, use_buffer=False, normalize=False)
            else:
                sents = split_sentences(raw_text, lang_code=lang_code, min_len=5, use_buffer=True, normalize=False)

            # IMPORTANT: this file is fed DIRECTLY into MMS forced-alignment
            # by force_aligned_over_folders_parallel.py (--text_filepath),
            # which expects plain text, one sentence per line -- it must NOT
            # be TSV. IDs are written to a separate sidecar file instead,
            # in the exact same line order, so force_aligned_over_folders_
            # parallel.py can attach sentence_id to each forced-alignment
            # manifest entry positionally (see save_sentence_id_sidecar()).
            split_path = os.path.join(text_dir, lang, episode, f"{episode}_split_sentences.txt")
            os.makedirs(os.path.dirname(split_path), exist_ok=True)
            with open(split_path, "w", encoding="utf-8") as f:
                for s in sents:
                    f.write(f"{str(s).strip()}\n")
            save_sentence_id_sidecar(sents, lang, episode, split_path)

            #sents = filter_language_mismatch(sents, lang) if sents else []
            split_sentences_dict[lang] = sents
            tqdm.write(f"  ▸ {lang}: {len(sents)} sentences")
        except Exception as e:
            tqdm.write(f"⚠️ Failed splitting for {lang}: {e}")
            split_sentences_dict[lang] = [raw_text] if raw_text.strip() else []

    output = os.path.join(args.output_dir, episode)
    os.makedirs(output, exist_ok=True)

    parallel_corpus = []
    if "English" not in split_sentences_dict or not split_sentences_dict["English"]:
        tqdm.write(f"⚠️ No English sentences found for episode {episode} — skipping alignment.")
        return None

    english_sents = split_sentences_dict["English"]

    # Accumulates (lang, [ {lang: ..., "English": ..., "similarity": ...,
    # "aligned": ..., "stage": ...}, ... ]) across ALL languages -- every
    # candidate pair seriously considered by ANY stage, before thresholding.
    complete_pairs_corpus = []

    # Accumulates (lang, [rows]) across all languages -- every sliding AND
    # topk row per English sentence (not just the winner), each tagged
    # is_selected_best -- only populated when --save_all_candidates is set.
    both_candidates_corpus = []

    for lang in languages:
        try:
            if lang == "English":
                continue
            if lang not in split_sentences_dict or not split_sentences_dict[lang]:
                tqdm.write(f"⚠️ No sentences for {lang} — skipping.")
                continue

            # Collects every candidate pair examined for THIS language, across
            # the sliding-window stage and both mutual-top-k calls below.
            lang_complete_pairs = []

            # CHANGE 5: Manipuri's sliding method is controlled by
            # --manipuri_sliding_mode ("sonar_only" default, or "fused") rather
            # than a hardcoded branch. Fused LaBSE+SONAR was tried for Manipuri
            # once before and reverted to SONAR-only -- but that was before
            # several fixes since (drop_duplicates, min_span_chars, the
            # tolerance/confidence-override rework, display_similarity), so
            # it's worth being able to re-test fused now under a controlled
            # flag rather than guessing which one is actually better today.
            # Both functions share the same (aligned_pairs, full_units,
            # all_compared, resolved_threshold) return shape, so nothing else
            # downstream needs to know which one ran -- "stage" on each row
            # ("sliding_sonar" vs "sliding_fused") is what actually
            # distinguishes them for ids/columns/overrides.
            use_sonar_only = (lang == "Manipuri" and args.manipuri_sliding_mode == "sonar_only")
            if use_sonar_only:
                results_optimised, results_optimised_full, results_all_compared, sliding_fused_min = align_sentences_to_paragraph_twostage_unified_sonar_paragraph_aligned(
                    src_sentences=english_sents,
                    tgt_paragraph=texts[lang],
                    src_lang_tag="eng_Latn",
                    tgt_lang_tag=sonar_language_codes[lang],
                    sonar_model=text_embedder_seq,
                    lang=lang
                )
            else:
                results_optimised, results_optimised_full, results_all_compared, sliding_fused_min = align_sentences_to_paragraph_twostage_unified(
                    src_sentences=english_sents,
                    tgt_paragraph=texts[lang],
                    src_lang_tag="eng_Latn",
                    tgt_lang_tag=sonar_language_codes[lang],
                    labse_model=labse_model_seq,
                    sonar_model=text_embedder_seq,
                    lang=lang
                )

            lang_complete_pairs.extend([{**r, "stage": "sliding"} for r in results_all_compared])

            df_full = pd.DataFrame(results_optimised_full)

            # --- audio split ---
            # FIX #1: neither branch below previously filtered on
            # df_full["aligned"] at all -- both wrote df_full[lang], one deduped
            # and one not, so "aligned_audio.txt" and "unaligned_audio.txt" were
            # near-duplicates of each other regardless of actual alignment
            # status. Now actually split by the "aligned" column.
            #
            # FIX #2 / IMPORTANT: "unaligned_audio.txt" is fed DIRECTLY into MMS
            # forced-alignment by force_aligned_over_folders_parallel.py
            # (--text_filepath), which expects plain text, one sentence per
            # line -- NOT TSV. Both files are kept as plain text here (matching
            # "aligned_audio.txt" for consistency, even though only the
            # unaligned one is currently force-aligned); a sidecar id file is
            # written alongside each instead, same line order, so ids can be
            # attached positionally downstream without touching the text MMS
            # actually consumes.
            aligned_path = os.path.join(
                text_dir, lang, episode, f"{episode}_aligned_audio.txt"
            )
            unaligned_path = os.path.join(
                text_dir, lang, episode, f"{episode}_unaligned_audio.txt"
            )
            os.makedirs(os.path.dirname(aligned_path), exist_ok=True)

            aligned_span_df = df_full[df_full["aligned"] == True]      # noqa: E712
            unaligned_span_df = df_full[df_full["aligned"] == False]   # noqa: E712
            # FIX: this used to be `.drop_duplicates(subset=[lang])` -- silently
            # collapsing any TWO leftover gaps that happened to be word-for-word
            # identical text down to ONE line in unaligned_audio.txt, even
            # though df_full/full_units (built by walking the ENTIRE paragraph,
            # start to end, in align_sentences_to_paragraph_twostage_unified's
            # step 4) genuinely has both occurrences. That silently dropped real
            # paragraph content from the saved file -- aligned_audio.txt +
            # unaligned_audio.txt no longer reconstructed the full source
            # paragraph. Sentence ids don't need this dedup: identical text
            # simply gets the identical id at each occurrence (that's the
            # correct, intended behavior of make_sentence_id), so every
            # occurrence is now kept.

            aligned_sentences = aligned_span_df[lang].dropna().apply(lambda s: str(s).strip()).tolist()
            unaligned_sentences = unaligned_span_df[lang].dropna().apply(lambda s: str(s).strip()).tolist()

            with open(aligned_path, "w", encoding="utf-8") as f:
                for s in aligned_sentences:
                    f.write(f"{s}\n")
            save_sentence_id_sidecar(aligned_sentences, lang, episode, aligned_path)

            with open(unaligned_path, "w", encoding="utf-8") as f:
                for s in unaligned_sentences:
                    f.write(f"{s}\n")
            save_sentence_id_sidecar(unaligned_sentences, lang, episode, unaligned_path)

            # --- COMPLETE, ORDER-PRESERVING paragraph output ---
            # aligned_audio.txt and unaligned_audio.txt above are each filtered
            # subsets of df_full -- individually order-preserving, but SPLIT
            # into two files, so neither one alone (nor a naive concat of the
            # two) reconstructs the actual paragraph, since aligned/unaligned
            # segments are interleaved in the source text. For downstream
            # audio-splitting/forced-alignment that needs to walk the ENTIRE
            # continuous paragraph in its real order, that's not enough --
            # so this writes df_full completely unfiltered, in its natural
            # (already paragraph-ordered, since step 4 built it by walking
            # tgt_words[0:n_words] once) row order.
            full_paragraph_path = os.path.join(
                text_dir, lang, episode, f"{episode}_full_paragraph.txt"
            )
            full_segments = df_full[lang].apply(lambda s: str(s).strip()).tolist()
            with open(full_paragraph_path, "w", encoding="utf-8") as f:
                for s in full_segments:
                    f.write(f"{s}\n")

            # Richer sidecar than save_sentence_id_sidecar's plain idx/id/text --
            # this one also carries "aligned"/"English"/"similarity" per line,
            # same order as full_paragraph.txt, so downstream can tell which
            # lines are already-translated spans vs. untranslated gaps without
            # re-deriving that from the two separate filtered files.
            full_sidecar_path = full_paragraph_path + ".ids.tsv"
            with open(full_sidecar_path, "w", encoding="utf-8") as f:
                for i, row in df_full.reset_index(drop=True).iterrows():
                    text = str(row[lang]).strip()
                    sid = make_sentence_id(lang, episode, text)
                    aligned_flag = bool(row.get("aligned", False))
                    eng = row.get("English") or ""
                    sim = row.get("similarity")
                    sim_str = "" if pd.isna(sim) else str(sim)
                    f.write(f"{i}\t{sid}\t{aligned_flag}\t{sanitize_for_tsv_field(eng)}\t{sim_str}\t{sanitize_for_tsv_field(text)}\n")


            # --- normalize for corpus ---
            # Sliding's own accepted pairs. These are already-accepted (they're
            # in results_optimised precisely because they cleared fused_min/
            # sonar_min inside the sliding function), so aligned=True here is a
            # fact, not a guess. Tagged stage="sliding" -- see below, this
            # merges with the leftover-topk fill into one "sliding leg".
            sliding_dicts = [
                {**p, "aligned": True, "stage": "sliding", "threshold": sliding_fused_min}
                for p in results_optimised
            ]

            # ------------------------------------------------------------
            # THREE candidate sources per English sentence, in priority order:
            #   1) sliding's own result
            #   2) topk run ONLY on English sentences sliding missed ("remaining")
            #   3) topk run independently on the FULL list
            # (1) and (2) are folded into one combined "sliding leg" (both
            # tagged stage="sliding") since (2) exists purely to fill sliding's
            # gaps -- there's never a same-sentence conflict between them,
            # since (2) only ever runs on sentences (1) didn't cover. (3) stays
            # a fully separate, independent leg (stage="mutual_topk_cosine").
            # Priority sliding > leftover-topk > full-topk falls out naturally
            # from merging [sliding_leg (1+2)] ahead of [full-topk (3)] below.
            # ------------------------------------------------------------
            # FIX: this used to track "covered" English sentences by TEXT
            # (a set of strings) -- which silently collapsed two DIFFERENT
            # occurrences of the same exact sentence text (a repeated phrase
            # in the transcript) into one, since a Python set can't hold a
            # string twice. Now tracked by POSITION in english_sents instead,
            # so two occurrences of identical wording are treated as the two
            # distinct sentences they actually are.
            sliding_covered_indices = {p["english_idx"] for p in results_optimised if "english_idx" in p}
            remaining_indexed = [(i, s) for i, s in enumerate(english_sents) if i not in sliding_covered_indices]
            remaining_english_sents = [s for _, s in remaining_indexed]
            remaining_original_indices = [i for i, _ in remaining_indexed]  # maps leftover-topk's local j back to the true position

            if remaining_english_sents:
                topk_leftover_pairs, topk_leftover_all_compared, topk_leftover_threshold = align_sentences_mutual_topk(
                    split_sentences_dict[lang], remaining_english_sents, lang, "English",
                    args.threshold, args.top_k, return_all_compared=True,
                    max_usage=args.max_usage,
                    sentence2_original_indices=remaining_original_indices,
                )
            else:
                topk_leftover_pairs, topk_leftover_all_compared, topk_leftover_threshold = [], [], args.threshold

            lang_complete_pairs.extend([{**r, "stage": "sliding"} for r in topk_leftover_all_compared])

            topk_leftover_dicts = [
                {**p, "aligned": True, "stage": "sliding", "threshold": topk_leftover_threshold}
                for p in topk_leftover_pairs
            ]

            # Combined sliding leg: sliding's own accepts + leftover-topk fill,
            # both tagged stage="sliding". This is what {lang}_sliding_aligned.xlsx
            # saves, and what wins first in the final merge below.
            sliding_leg_dicts = sliding_dicts + topk_leftover_dicts

            if args.debug:
                df_sliding_leg = pd.DataFrame(sliding_leg_dicts)
                safe_write_alignment_excel(
                    df_sliding_leg, lang,
                    os.path.join(output, f"{lang}_sliding_aligned.xlsx"),
                    episode=episode, default_stage="sliding"
                )

            # Independent full-topk leg (unchanged): runs on the FULL sentence
            # list regardless of what sliding/leftover-topk already covered.
            topk_pairs, topk_all_compared, topk_effective_threshold = align_sentences_mutual_topk(
                split_sentences_dict[lang], english_sents, lang, "English",
                args.threshold, args.top_k, return_all_compared=True,
                max_usage=args.max_usage
            )
            lang_complete_pairs.extend(topk_all_compared)
            if args.debug:
                df_topk = pd.DataFrame(topk_pairs)
                safe_write_alignment_excel(
                    df_topk, lang,
                    os.path.join(output, f"{lang}_aligned_topk_only.xlsx"),
                    episode=episode, default_stage="mutual_topk_cosine"
                )

            topk_dicts = [
                {**p, "aligned": True, "stage": "mutual_topk_cosine", "threshold": topk_effective_threshold}
                for p in topk_pairs
            ]

            # Safety net (only remaining "fallback"): if BOTH legs produced
            # nothing at all for this language, retry full-topk once more with
            # a lower threshold.
            if not sliding_leg_dicts and not topk_dicts:
                tqdm.write(f"⚠️ Sliding + top-k both failed for {lang}, retrying top-k with a lower threshold")
                topk_fallback, topk_all_compared_fallback, topk_fallback_threshold = align_sentences_mutual_topk(
                    split_sentences_dict[lang], english_sents, lang, "English",
                    threshold=0.55, top_k=args.top_k, return_all_compared=True,
                    max_usage=args.max_usage
                )
                lang_complete_pairs.extend(topk_all_compared_fallback)
                topk_dicts = [
                    {**p, "aligned": True, "stage": "mutual_topk_cosine", "threshold": topk_fallback_threshold}
                    for p in topk_fallback
                ]

            # CHANGE 3 (real-output dedup): one best row per English sentence
            # across the sliding leg AND the full-topk leg. sliding_leg_dicts is
            # inserted FIRST, so it wins ties (matching the confirmed priority:
            # sliding > leftover-topk > full-topk) -- UNLESS
            # --allow_topk_override_weak_sliding is set and the sliding-leg row
            # is weak (below --low_confidence_margin above its own threshold),
            # in which case full-topk's independently-agreeing row wins instead.
            merged_dicts, both_candidates_rows = collapse_to_best_per_english(
                sliding_leg_dicts + topk_dicts, lang,
                allow_topk_override_weak_sliding=args.allow_topk_override_weak_sliding,
                low_confidence_margin=args.low_confidence_margin,
                return_all_candidates=args.save_all_candidates
            )

            final_pairs = normalize_final_pairs(merged_dicts, lang)

            if args.save_all_candidates and both_candidates_rows:
                both_candidates_corpus.append((lang, both_candidates_rows))
        
            if final_pairs:
                parallel_corpus.append((lang, final_pairs))
                tqdm.write(f"  ✓ {lang}: {len(final_pairs)} aligned pairs")
            else:
                tqdm.write(f"  ⚠️ {lang}: no aligned pairs produced")

            if lang_complete_pairs:
                # Both stages now emit exactly one all_compared row per English
                # sentence (sliding: one best span per source sentence; topk:
                # one best/accepted partner per column j), so collapsing with
                # the SAME "one row per English sentence" rule used for the
                # real output guarantees complete_pairs also has exactly
                # len(english_sents) rows -- a pure, threshold-independent
                # comparison table, not len(sentences1)+len(sentences2).
                lang_complete_pairs, _ = collapse_to_best_per_english(lang_complete_pairs, lang)
                complete_pairs_corpus.append((lang, lang_complete_pairs))
                tqdm.write(f"  ▸ {lang}: {len(lang_complete_pairs)} candidate pairs captured before thresholding (== {len(english_sents)} English sentences)")
        except Exception as e:
            # FIX: previously an exception in ANY single language's
            # processing (CUDA OOM on a long paragraph, an encoding
            # edge case, an unexpected empty/malformed source file --
            # far more likely to hit SOMETHING across a full 36-language
            # run than a 3-4 language test run) would propagate all the
            # way up and abort process_episode for the ENTIRE episode --
            # meaning the combined all_aligned/all_all_candidates
            # Excel+JSON at the end of this function never got reached,
            # even for languages that succeeded before the failure. Now
            # one bad language is logged and skipped; every other
            # language's results for this episode still make it through
            # to the final combined save.
            tqdm.write(f"❌ {lang} failed for episode {episode}, skipping this language: {e}")
            import traceback
            tqdm.write(traceback.format_exc())
            continue

    if not parallel_corpus:
        tqdm.write(f"⚠️ No alignments produced for episode {episode} — nothing to save.")
        return None

    merged_df = None
    for lang, pairs in parallel_corpus:
        df = pd.DataFrame(pairs).drop_duplicates(subset=[lang, "English"]).dropna(subset=[lang, "English", "similarity"])

        # sentence_id columns: deterministic, computed straight from the
        # text already in this row -- no lookup needed, and tagged with
        # THIS row's own stage (sliding/topk), so the id itself tells you
        # which stage produced this particular aligned pair.
        df[f"{lang}_sentence_id"] = df.apply(
            lambda r: make_sentence_id(lang, episode, r[lang]),
            axis=1
        )
        df["English_sentence_id"] = df.apply(
            lambda r: make_sentence_id("English", episode, r["English"]),
            axis=1
        )

        if "raw_similarity" not in df.columns:
            df["raw_similarity"] = df["similarity"]
        cols = [c for c in [lang, "English", "similarity", "raw_similarity", "aligned", "stage",
                             f"{lang}_sentence_id", "English_sentence_id"] if c in df.columns]
        df[cols].to_excel(os.path.join(output, f"{lang}_aligned.xlsx"), index=False)

        df_grouped = df.groupby("English")[lang].apply(lambda x: list(sorted(set(x)))).reset_index()
        # parallel list of ids, same order/length as the text list above
        df_grouped[f"{lang}_sentence_ids"] = df_grouped[lang].apply(
            lambda lst: [make_sentence_id(lang, episode, t) for t in lst]
        )
        merged_df = df_grouped if merged_df is None else pd.merge(merged_df, df_grouped, on="English", how="outer")

    # --- Save the PRE-THRESHOLD candidate pairs per language ---
    # Every row here is a genuine candidate pair that some alignment stage
    # actually proposed (one row per source sentence's best match, collapsed
    # across stages) -- NOT every possible sentence combination, and NOT
    # filtered by threshold. The "aligned" column records whether that
    # stage's own threshold accepted it; "similarity" is on that stage's
    # own scale (see collapse_to_best_per_sentence docstring: don't compare
    # similarity numbers across different "stage" values).
    for lang, pairs in complete_pairs_corpus:
        df_complete = pd.DataFrame(pairs)
        if df_complete.empty:
            continue
        cols = [c for c in [lang, "English", "similarity", "raw_similarity", "aligned", "stage"] if c in df_complete.columns]
        df_complete = df_complete[cols].drop_duplicates()

        if lang in df_complete.columns:
            df_complete[f"{lang}_sentence_id"] = df_complete.apply(
                lambda r: make_sentence_id(lang, episode, r[lang])
                if pd.notna(r[lang]) else None,
                axis=1
            )
        if "English" in df_complete.columns:
            df_complete["English_sentence_id"] = df_complete.apply(
                lambda r: make_sentence_id("English", episode, r["English"])
                if pd.notna(r["English"]) else None,
                axis=1
            )

        df_complete.to_excel(os.path.join(output, f"{lang}_all_pairs_before_threshold.xlsx"), index=False)
        tqdm.write(f"  ✓ {lang}: {len(df_complete)} candidate pairs saved -> {lang}_all_pairs_before_threshold.xlsx")

    # --- Save {lang}_aligned_both_candidates.xlsx (--save_all_candidates) ---
    # Purely additive: every sliding AND topk row per English sentence,
    # nothing discarded, each tagged is_selected_best. Does not touch
    # {lang}_aligned.xlsx, all_pairs_before_threshold, split_sentences, or
    # the aligned/unaligned audio files above.
    merged_df_candidates = None
    if args.save_all_candidates:
        for lang, rows in both_candidates_corpus:
            df_cand = pd.DataFrame(rows)
            if df_cand.empty:
                continue
            df_cand[f"{lang}_sentence_id"] = df_cand.apply(
                lambda r: make_sentence_id(lang, episode, r[lang]),
                axis=1
            )
            df_cand["English_sentence_id"] = df_cand.apply(
                lambda r: make_sentence_id("English", episode, r["English"]),
                axis=1
            )
            if "raw_similarity" not in df_cand.columns:
                df_cand["raw_similarity"] = df_cand["similarity"]
            cols = [c for c in [lang, "English", "similarity", "raw_similarity", "aligned", "stage", "is_selected_best",
                                 f"{lang}_sentence_id", "English_sentence_id"] if c in df_cand.columns]
            df_cand[cols].to_excel(os.path.join(output, f"{lang}_aligned_both_candidates.xlsx"), index=False)
            tqdm.write(f"  ✓ {lang}: {len(df_cand)} candidate rows (both stages) saved -> {lang}_aligned_both_candidates.xlsx")

            df_grouped_cand = df_cand.groupby("English")[lang].apply(lambda x: list(sorted(set(x)))).reset_index()
            df_grouped_cand[f"{lang}_sentence_ids"] = df_grouped_cand[lang].apply(
                lambda lst: [make_sentence_id(lang, episode, t) for t in lst]
            )
            merged_df_candidates = df_grouped_cand if merged_df_candidates is None else pd.merge(
                merged_df_candidates, df_grouped_cand, on="English", how="outer"
            )

    if merged_df is None or merged_df.empty:
        tqdm.write(f"⚠️ After processing, merged_df is empty for episode {episode}. Skipping final save.")
        return None

    # Added once here (not per-language) so it persists into every
    # downstream save of merged_df below, including the "without_null"
    # variant -- it's just another column on the same DataFrame object.
    merged_df["English_sentence_id"] = merged_df["English"].apply(
        lambda t: make_sentence_id("English", episode, t) if pd.notna(t) else None
    )

    combined_excel = os.path.join(output, "all_aligned.xlsx")
    merged_df.to_excel(combined_excel, index=False)
    tqdm.write(f"✅ Saved combined Excel: {combined_excel}")

    merged_df.to_json(os.path.join(output, "all_aligned.json"), orient="records", force_ascii=False, indent=0)

    for col in merged_df.columns:
        if col.endswith("_sentence_id") or col.endswith("_sentence_ids"):
            continue  # id columns aren't sentence text -- nothing to write as a text file
        out_dir = os.path.join(args.text_dir, col)
        os.makedirs(out_dir, exist_ok=True)
        save_list_column_to_lines(merged_df, col, os.path.join(out_dir, episode, f"{episode}_en_aligned.tsv"), episode)

    if args.save_all_candidates and merged_df_candidates is not None and not merged_df_candidates.empty:
        merged_df_candidates["English_sentence_id"] = merged_df_candidates["English"].apply(
            lambda t: make_sentence_id("English", episode, t) if pd.notna(t) else None
        )
        combined_candidates_excel = os.path.join(output, "all_all_candidates.xlsx")
        merged_df_candidates.to_excel(combined_candidates_excel, index=False)
        tqdm.write(f"✅ Saved combined candidates Excel: {combined_candidates_excel}")
        merged_df_candidates.to_json(os.path.join(output, "all_all_candidates.json"), orient="records", force_ascii=False, indent=0)

        for col in merged_df_candidates.columns:
            if col.endswith("_sentence_id") or col.endswith("_sentence_ids"):
                continue
            out_dir = os.path.join(args.text_dir, col)
            os.makedirs(out_dir, exist_ok=True)
            save_list_column_to_lines(merged_df_candidates, col, os.path.join(out_dir, episode, f"{episode}_en_all_candidates.tsv"), episode)

    # FIX: this used to be merged_df.dropna() with NO subset/thresh -- which
    # requires EVERY SINGLE language column to be non-null on a row to
    # survive. With 3-4 test languages that's plausible; with ~30 real
    # languages simultaneously producing data per episode, the odds of one
    # English sentence being matched by literally ALL of them at once
    # collapses toward zero -- producing an EMPTY "without_null" file for
    # most episodes at full 36-language scale, even though the same code
    # worked fine in smaller tests. This is not about a language being
    # absent from the episode (those never become columns at all here,
    # since merged_df only ever gains a column for languages that actually
    # appeared in parallel_corpus) -- it's the "must agree unanimously"
    # requirement itself not scaling.
    #
    # --min_lang_frac_for_without_null (default 0.5) requires at least
    # that FRACTION of the present language columns to be non-null,
    # instead of ALL of them.
    lang_cols = [c for c in merged_df.columns if c != "English" and not c.endswith("_sentence_id") and not c.endswith("_sentence_ids")]
    min_present = max(1, int(round(args.min_lang_frac_for_without_null * len(lang_cols)))) if lang_cols else 0
    merged_df = merged_df.dropna(subset=lang_cols, thresh=min_present) if lang_cols else merged_df.dropna()
    if not merged_df.empty:
        combined_clean_excel = os.path.join(output, "all_aligned_without_null.xlsx")
        merged_df.to_excel(combined_clean_excel, index=False)
        tqdm.write(f"✅ Saved cleaned Excel: {combined_clean_excel}")
        merged_df.to_json(os.path.join(output, "all_aligned_without_null.json"), orient="records", force_ascii=False, indent=0)
        for col in merged_df.columns:
            if col.endswith("_sentence_id") or col.endswith("_sentence_ids"):
                continue
            save_list_column_to_lines(merged_df, col, os.path.join(args.text_dir, col, episode, f"{episode}_all_aligned.tsv"), episode)
        tqdm.write(f"✅ Saved final cleaned aligned file for episode {episode}")

    # FIX: this used to be nested inside `if not merged_df.empty:` above --
    # coupling the candidates-without-null save to the ALIGNED dataframe's
    # emptiness, even though they're independent dataframes with
    # independent dropna outcomes. Now computed unconditionally.
    if args.save_all_candidates and merged_df_candidates is not None:
        cand_lang_cols = [c for c in merged_df_candidates.columns if c != "English" and not c.endswith("_sentence_id") and not c.endswith("_sentence_ids")]
        cand_min_present = max(1, int(round(args.min_lang_frac_for_without_null * len(cand_lang_cols)))) if cand_lang_cols else 0
        merged_df_candidates_clean = merged_df_candidates.dropna(subset=cand_lang_cols, thresh=cand_min_present) if cand_lang_cols else merged_df_candidates.dropna()
        if not merged_df_candidates_clean.empty:
            combined_candidates_clean_excel = os.path.join(output, "all_all_candidates_without_null.xlsx")
            merged_df_candidates_clean.to_excel(combined_candidates_clean_excel, index=False)
            tqdm.write(f"✅ Saved cleaned candidates Excel: {combined_candidates_clean_excel}")
            merged_df_candidates_clean.to_json(os.path.join(output, "all_all_candidates_without_null.json"), orient="records", force_ascii=False, indent=0)
            for col in merged_df_candidates_clean.columns:
                if col.endswith("_sentence_id") or col.endswith("_sentence_ids"):
                    continue
                save_list_column_to_lines(
                    merged_df_candidates_clean, col,
                    os.path.join(args.text_dir, col, episode, f"{episode}_all_all_candidates.tsv"), episode
                )
    if merged_df.empty:
        tqdm.write(f"⚠️ No non-null rows after cleaning for episode {episode}.")
    return merged_df

# -------------------------
# Orchestration: parallel and sequential entrypoints
# -------------------------
def extract_episode_number(folder_name):
    match = re.search(r'MKB_(\d+)', folder_name)
    return int(match.group(1)) if match else float('inf')
    
def run_sequential_mode(args):
    init_sequential_models(MAIN_DEVICE)

    text_dir = args.text_dir
    languages = sorted([l for l in os.listdir(text_dir) if os.path.isdir(os.path.join(text_dir, l))])
    if not languages:
        raise ValueError("No language folders found.")
    first_lang_dir = os.path.join(args.text_dir, languages[0])
    episode_folders = sorted(
        os.listdir(first_lang_dir),
        key=extract_episode_number,
        reverse=True
    ) if os.listdir(first_lang_dir) else []
    episode_folders = sorted(os.listdir(first_lang_dir), key=lambda x: int(re.search(r'\d+', x).group())) if os.listdir(first_lang_dir) else []
    episode_folders = episode_folders[16:]
    #episode_folders = episode_folders[31:-16]
    
    #episode_folders = ['MKB_17_February_2016']#,'MKB_119_February_2025','MKB_120_March_2025']

    for episode in tqdm(episode_folders, desc="Processing episodes (sequential)"):
        process_episode(episode, args, embed_fn=get_embeddings_sequential)


# -------------------------------------------------------------------------
# Parallel mode: worker-pool design.
#
# Earlier design: a small, fixed number of "GPU server" processes each held
# ONE shared model copy, and every worker/thread routed its embedding calls
# through a request queue to those servers, gated by a global
# GPU_SEMAPHORE(1). That semaphore allowed only ONE embedding job to run at
# a time, GPU-wide, no matter how many worker threads were spun up -- so
# extra workers just added queueing/IPC overhead without adding real
# concurrency. Sequential single-worker processing measured at ~8-9GB GPU
# memory, well under what an 80GB H100 can hold several copies of.
#
# New design: N independent OS processes, each loading its OWN full model
# copy (via init_sequential_models) directly onto its assigned GPU, then
# calling process_episode() the SAME way run_sequential_mode() does --
# fully sequential *within* each worker, but N workers genuinely running at
# once. Total concurrent workers = n_gpus * workers_per_gpu. On a single
# H100, workers_per_gpu around 8-10 (each ~8-9GB) is a reasonable starting
# point; watch nvidia-smi and back off if you see OOM, since actual peak
# memory depends on episode paragraph length (longer paragraphs -> more
# sliding-window spans embedded at once).
# -------------------------------------------------------------------------

def _pool_worker_init(gpu_id):
    """
    Runs once, automatically, when each worker process in the pool starts.
    Loads one full model copy onto this worker's assigned GPU and stores it
    in this worker's own globals (see init_sequential_models docstring).
    Everything process_episode() does after this point runs entirely
    inside this one worker process, using only its own model copy.
    """
    # FIX: this used to only pass device=cuda:{gpu_id} to init_sequential_
    # models(), which correctly placed THIS module's own models (SONAR/
    # LaBSE/GTE/BGE) on the right GPU -- but never changed the process's
    # actual DEFAULT CUDA device. Any OTHER library call that doesn't take
    # an explicit device argument -- Stanza's use_gpu=torch.cuda.is_
    # available() is exactly this -- silently defaults to PyTorch's
    # current device, which stayed cuda:0 for EVERY worker process
    # regardless of assigned GPU. With --workers_per_gpu 7 --n_gpus 4,
    # that meant all 28 workers' Stanza models piled onto GPU 0 on top of
    # the 7 that legitimately belonged there, exhausting it -- exactly
    # the OOM breakdown observed (7 processes at ~8.8GB = the real GPU-0
    # workers' full model sets, ~20 more at ~750MB-1GB = every other
    # worker's stray Stanza model). torch.cuda.set_device(gpu_id) here
    # makes gpu_id this process's actual default, so anything without an
    # explicit device argument lands on the correct GPU too.
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    init_sequential_models(device)
    print(f"[WORKER pid={os.getpid()}] ready on {device}")


def _pool_worker_process_episode(episode, args):
    try:
        return process_episode(episode, args, embed_fn=get_embeddings_sequential)
    except Exception as e:
        print(f"[worker pid={os.getpid()}] error processing {episode}: {e}")
        return None


def run_parallel_mode(args):
    n_gpus = max(1, min(
        args.n_gpus,
        torch.cuda.device_count() if torch.cuda.is_available() else 0 or 1
    ))
    if n_gpus <= 0:
        print("No GPUs detected; falling back to sequential mode.")
        return run_sequential_mode(args)

    workers_per_gpu = max(1, args.workers_per_gpu)
    total_workers = n_gpus * workers_per_gpu

    text_dir = args.text_dir
    languages = sorted([l for l in os.listdir(text_dir) if os.path.isdir(os.path.join(text_dir, l))])
    if not languages:
        raise ValueError("No language folders found.")
    first_lang_dir = os.path.join(args.text_dir, languages[0])
    episode_folders = sorted(
        os.listdir(first_lang_dir),
        key=extract_episode_number,
        reverse=True
    ) if os.listdir(first_lang_dir) else []
    #episode_folders = sorted(os.listdir(first_lang_dir), key=lambda x: int(re.search(r'\d+', x).group())) if os.listdir(first_lang_dir) else []
    episode_folders = episode_folders[:-16]
    '''episode_folders.extend([
        'MKB_76_April_2021','MKB_74_February_2021','MKB_94_October_2022',
        'MKB_90_June_2022','MKB_57_September_2019','MKB_77_May_2021','MKB_78_June_2021'
    ])
    episode_folders = sorted(
        episode_folders,
        key=lambda x: int(re.search(r'\d+', x).group())
    )'''

    print(f"[LAUNCHER] {n_gpus} GPU(s) x {workers_per_gpu} worker(s)/GPU = {total_workers} parallel workers")
    print(f"[LAUNCHER] {len(episode_folders)} episode(s) queued")

    # One process Pool per GPU so each worker's fixed gpu_id can be baked
    # in via initargs (a single Pool's initializer gets the same initargs
    # for every worker, so per-GPU assignment needs one Pool per GPU).
    pools = []
    # FIX: mp.Pool never recycles worker processes by default (no
    # maxtasksperchild) -- a worker stays alive for the entire run unless
    # it crashes outright. But a CUDA "unspecified launch failure" (or
    # several other CUDA fault classes) PERMANENTLY POISONS that worker's
    # CUDA context for the rest of the process's life -- it does not
    # self-heal. Since both this function's and process_episode's own
    # per-language exception handlers just log-and-continue (by design,
    # so one bad language/episode doesn't kill a healthy worker), a
    # poisoned worker just keeps getting fed more episodes by the Pool
    # and fails EVERY one of them silently, for the rest of the entire
    # run, without the pipeline ever visibly stopping. --maxtasksperchild
    # forces the Pool to kill and respawn a fresh worker (clean CUDA
    # context) every N episodes, bounding the damage from a poisoned
    # worker to at most N wasted episodes instead of unbounded. Trade-off:
    # each respawn re-runs _pool_worker_init (full model reload), so too
    # low a value adds real overhead -- default 20 balances that against
    # how much a single poisoned worker can silently waste.
    for gpu_id in range(n_gpus):
        pool = mp.Pool(processes=workers_per_gpu, initializer=_pool_worker_init, initargs=(gpu_id,),
                        maxtasksperchild=args.maxtasksperchild)
        pools.append(pool)

    async_results = []
    for idx, episode in enumerate(episode_folders):
        pool = pools[idx % n_gpus]  # round-robin episodes across GPUs
        async_results.append(pool.apply_async(_pool_worker_process_episode, (episode, args)))

    for pool in pools:
        pool.close()

    results = []
    for ar in async_results:
        try:
            results.append(ar.get())
        except Exception as e:
            print("Worker exception:", e)

    for pool in pools:
        pool.join()

    return results

def run_pipeline(args):
    if args.run_mode == "sequential":
        run_sequential_mode(args)
    else:
        run_parallel_mode(args)

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method", choices=["sliding","topk","all"], default="all")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--min_window", type=int, default=1)
    parser.add_argument("--max_window", type=int, default=350)
    parser.add_argument("--step", type=int, default=12)
    parser.add_argument("--dynamic_scale", type=float, default=1.5)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--max_usage", type=int, default=1)
    parser.add_argument("--languages", type=str, default=None,
                         help="Comma-separated list of language folder names to process "
                              "(English is always included). Default: process every "
                              "language folder found under --text_dir.")
    parser.add_argument("--low_resource_only", action="store_true",
                         help="Only process languages in LOW_RESOURCE_LANGS + English. "
                              "Ignored if --languages is given.")
    parser.add_argument("--use_simalign", action="store_true")
    parser.add_argument("--allow_topk_override_weak_sliding", action="store_true",
                         help="When sliding AND topk both accepted a candidate for the same "
                              "English sentence, and sliding's own margin above its threshold "
                              "is below --low_confidence_margin, let topk's row win instead of "
                              "the default 'sliding wins ties' rule. Off by default.")
    parser.add_argument("--low_confidence_margin", type=float, default=0.10,
                         help="Normalized margin -- (score - stage_threshold) / stage_threshold "
                              "-- below which an accepted sliding match is considered weak "
                              "enough for topk to override, if --allow_topk_override_weak_sliding "
                              "is set.")
    parser.add_argument("--save_all_candidates", action="store_true",
                         help="Additionally save {lang}_aligned_both_candidates.xlsx (every "
                              "sliding AND topk row per English sentence, with is_selected_best "
                              "marking the winner) plus {episode}_en_all_candidates.tsv / "
                              "{episode}_all_all_candidates.tsv. Purely additive -- doesn't "
                              "change {lang}_aligned.xlsx, all_pairs_before_threshold.xlsx, "
                              "split_sentences, or aligned/unaligned audio files.")
    parser.add_argument("--manipuri_sliding_mode", choices=["sonar_only", "fused"], default="sonar_only",
                         help="Which sliding function Manipuri uses. 'sonar_only' (default) uses "
                              "align_sentences_to_paragraph_twostage_unified_sonar_paragraph_aligned "
                              "(no LaBSE). 'fused' switches Manipuri onto the same LaBSE+SONAR "
                              "path every other language uses. Fused was tried for Manipuri once "
                              "before and reverted, but several fixes have landed since (min_span_chars, "
                              "the tolerance/confidence-override rework, display_similarity) -- this "
                              "flag lets you re-test fused under a controlled comparison instead of "
                              "guessing which is currently better. Every other language is unaffected "
                              "either way.")
    parser.add_argument("--min_lang_frac_for_without_null", type=float, default=0.5,
                         help="Fraction (0-1) of the present language columns that must be "
                              "non-null for a row to survive into all_aligned_without_null.xlsx/json "
                              "and all_all_candidates_without_null.xlsx/json. Used to be a hard "
                              "requirement that EVERY language column be non-null (dropna() with "
                              "no subset/thresh) -- fine for a 3-4 language test, but with ~30 "
                              "real languages simultaneously producing data per episode, requiring "
                              "unanimous agreement across all of them collapses toward zero rows "
                              "for most episodes. Default 0.5 requires at least half the present "
                              "languages to have matched a given English sentence.")
    parser.add_argument("--maxtasksperchild", type=int, default=20,
                         help="How many episodes a single worker process handles before the Pool "
                              "kills and respawns it with a fresh CUDA context. A CUDA "
                              "'unspecified launch failure' (or several other CUDA fault classes) "
                              "permanently poisons a worker's CUDA context -- it does not recover "
                              "within the same process, so every episode assigned to a poisoned "
                              "worker afterward silently fails, for the rest of the entire run, "
                              "with the pipeline never visibly stopping. This bounds that damage "
                              "to at most this many wasted episodes per poisoned worker. Lower "
                              "values recover faster but add more model-reload overhead (each "
                              "respawn re-runs the full model load); higher values do the reverse.")
    parser.add_argument("--run_mode", choices=["sequential","parallel"], default="sequential")
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--workers_per_gpu", type=int, default=1,
                         help="Number of independent worker PROCESSES to run per GPU in "
                              "--run_mode parallel. Each worker loads its own full model "
                              "copy (SONAR+LaBSE+GTE+BGE, ~8-9GB on an H100 for this "
                              "pipeline) and processes episodes fully independently -- "
                              "total concurrent workers = n_gpus * workers_per_gpu. On a "
                              "single 80GB H100, 8-10 workers is a reasonable starting "
                              "point; watch nvidia-smi and reduce if you see OOM, since "
                              "peak memory depends on episode paragraph length (longer "
                              "paragraphs -> more sliding-window spans embedded at once). "
                              "Default: 1 (no extra parallelism beyond --n_gpus).")
    args = parser.parse_args()
    run_pipeline(args)
