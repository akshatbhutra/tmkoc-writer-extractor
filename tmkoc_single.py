#!/usr/bin/env python3
"""
Single-episode TMKOC writer-credit extractor (v2, local, CPU-friendly).

Setup (inside your venv):
    pip install -U "yt-dlp[default]" easyocr opencv-python-headless numpy
    ffmpeg must be installed and on PATH.

Usage:
    python tmkoc_single.py "https://youtu.be/XXXXXXXXXXX"
    python tmkoc_single.py "<url>" --start 60 --end 74 --debug
    python tmkoc_single.py "<url>" --known known_writers.txt

Optional known_writers.txt: one correct writer name per line (Hindi or English).
OCR output is snapped to the closest known name (raw OCR is always kept too).
"""
import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")

import argparse, csv, os, re, sys, tempfile
from collections import Counter
from difflib import SequenceMatcher, get_close_matches

import cv2
import numpy as np
import easyocr
import yt_dlp
from yt_dlp.utils import download_range_func

CROP_TOP = 0.35          # ignore top 35% of frame
KW_MIN_CONF = 0.50       # keyword must be read with at least this confidence
LINE_MIN_CONF = 0.08     # name lines are noisy; keep low, rely on voting
UPSCALE = 3

# tolerant to OCR damage: लखक / लेखक / लेखिका ; "Written by" / "Wnitten by"
KEYWORD_RE = re.compile(r"(ल[ेै]?ख[कि]|w.{0,3}tt?[eo]n\s*b[yv]|\bwriters?\b)", re.I)
STOP_RE = re.compile(
    r"(directed|director|dialogue|screenplay|story|creative|produced|producer|"
    r"unit|post\s*production|"
    r"निर्देशक|निर्देशन|पटकथा|संवाद|कहानी|निर्माता|रचनात्मक|"
    r"डायरे|यूनिट|यानिट|यानट|क्रिएटिव|किएटिव|प्रोडक्शन|प्रडक्शन|सुपर|"
    r"presents|asit|sony|sab)", re.I)

DEFAULT_KNOWN = ["अशोक परमार", "Raju Odedra", "Rajen Upadhyay"]


# ---------------------------------------------------------------- helpers
def episode_number(title):
    m = re.search(r"\bep(?:isode)?\.?\s*(?:no\.?)?\s*[:\-]?\s*(\d{1,5})", title, re.I)
    return int(m.group(1)) if m else None


def clean(t):
    return re.sub(r"\s+", " ", t).strip(" .,:;|-_")


def geom(bbox):
    xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
    return min(xs), min(ys), max(xs), max(ys)


def sim(a, b):
    return SequenceMatcher(None, a, b).ratio()


def download_clip(url, start, end, workdir, cookies=None, proxy=None):
    opts = {
        "format": "bestvideo[height<=720]/best[height<=720]/worst",
        "outtmpl": os.path.join(workdir, "clip.%(ext)s"),
        "download_ranges": download_range_func(None, [(start, end)]),
        "force_keyframes_at_cuts": True,
        "geo_bypass_country": "IN",
        "quiet": True, "no_warnings": True, "noprogress": True,
    }
    if cookies:
        opts["cookiefile"] = cookies
    if proxy:
        opts["proxy"] = proxy
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info), info


def iter_frames(path, fps_sample):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, int(round(fps / fps_sample)))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            yield i / fps, frame
        i += 1
    cap.release()


# ------------------------------------------------------- stage 1: keyword
def find_keyword(results):
    """Return best keyword box dict or None."""
    best = None
    for bbox, text, conf in results:
        if conf < KW_MIN_CONF or not KEYWORD_RE.search(text):
            continue
        x0, y0, x1, y1 = geom(bbox)
        cand = dict(x0=x0, y0=y0, x1=x1, y1=y1, h=y1 - y0, text=text, conf=conf)
        if best is None or conf > best["conf"]:
            best = cand
    return best


# ------------------------------------------------------ stage 2: names
def preprocess_variants(region):
    """Return list of images (BGR/gray) to OCR."""
    big = cv2.resize(region, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
    variants = []

    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    variants.append(("gray", clahe.apply(gray)))

    # green channel: white text stays bright, magenta card goes dark
    g = big[:, :, 1]
    g = cv2.GaussianBlur(g, (3, 3), 0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = 255 - bw                                   # black text on white
    bw = cv2.copyMakeBorder(bw, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
    variants.append(("green_otsu", bw))

    # plain upscale
    variants.append(("plain", big))
    return variants


def lines_from_results(results):
    """Group OCR boxes into text lines. Returns [(text, conf)] top to bottom."""
    items = []
    for bbox, text, conf in results:
        x0, y0, x1, y1 = geom(bbox)
        items.append(dict(x0=x0, cy=(y0 + y1) / 2, h=y1 - y0, text=text, conf=conf))
    items.sort(key=lambda b: b["cy"])
    rows = []
    for it in items:
        if rows and abs(it["cy"] - rows[-1][-1]["cy"]) < max(it["h"], 1) * 0.6:
            rows[-1].append(it)
        else:
            rows.append([it])
    out = []
    for row in rows:
        row.sort(key=lambda b: b["x0"])
        text = clean(" ".join(b["text"] for b in row))
        conf = sum(b["conf"] for b in row) / len(row)
        out.append((text, conf))
    return out


def name_lines(lines):
    names = []
    for text, conf in lines:
        if KEYWORD_RE.search(text) and not names:
            # keyword itself or "Written by Name": keep any trailing name
            rest = clean(KEYWORD_RE.sub("", text, count=1))
            if len(rest) > 2 and re.search(r"[A-Za-z\u0900-\u097F]{3}", rest):
                names.append((rest, conf))
            continue
        if STOP_RE.search(text):
            break
        if conf < LINE_MIN_CONF or len(text) < 3:
            continue
        if not re.search(r"[A-Za-z\u0900-\u097F]{2}", text):
            continue
        names.append((text, conf))
        if len(names) >= 4:
            break
    return names


def read_names(reader, crop, kw, debug_tag=None):
    """OCR a tight region under the keyword with several preprocessings.
    Returns list of candidates; each candidate is [(text, conf), ...]."""
    H, W = crop.shape[:2]
    kw_w = kw["x1"] - kw["x0"]
    cx = (kw["x0"] + kw["x1"]) / 2
    pad = max(kw_w * 3, 150)
    x_lo = int(max(0, min(kw["x0"] - kw_w, cx - pad)))
    x_hi = int(min(W, max(kw["x1"] + pad, cx + pad)))
    y_lo = int(max(0, kw["y1"] - kw["h"] * 0.1))
    y_hi = int(min(H, kw["y1"] + kw["h"] * 5))
    if y_hi - y_lo < 8 or x_hi - x_lo < 8:
        return []
    region = crop[y_lo:y_hi, x_lo:x_hi]

    cands = []
    for vname, img in preprocess_variants(region):
        if debug_tag is not None:
            cv2.imwrite(f"debug_frames/{debug_tag}_{vname}.png", img)
        res = reader.readtext(img, paragraph=False)
        names = name_lines(lines_from_results(res))
        if names:
            cands.append(names)
            if debug_tag is not None:
                print(f"      [{vname}] ->", " | ".join(f"{t} ({c:.2f})" for t, c in names))
    return cands


# ----------------------------------------------------------- consensus
def consensus(cands):
    """cands: list of [(text, conf), ...]. Returns (list_of_names, avg_conf)."""
    cands = [c for c in cands if c]
    if not cands:
        return [], 0.0
    n = Counter(len(c) for c in cands).most_common(1)[0][0]
    same = [c for c in cands if len(c) == n]
    names, confs = [], []
    for i in range(n):
        pool = [(c[i][0], c[i][1] + 0.05) for c in same]   # +0.05: avoid zero weights
        best, best_score = None, -1
        for s, w in pool:
            score = sum(w2 * sim(s, s2) for s2, w2 in pool) + w
            if score > best_score:
                best, best_score = s, score
        names.append(best)
        confs.append(sum(w for _, w in pool) / len(pool))
    return names, sum(confs) / len(confs)


def snap_names(names, known):
    out = []
    for n in names:
        m = get_close_matches(n, known, n=1, cutoff=0.6)
        out.append(m[0] if m else n)
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--start", type=int, default=20, help="window start, seconds")
    ap.add_argument("--end", type=int, default=25, help="window end, seconds")
    ap.add_argument("--fps", type=float, default=2, help="frames per second to scan")
    ap.add_argument("--cookies", default=None)
    ap.add_argument("--proxy", default=None)
    ap.add_argument("--known", default="known_writers.txt",
                    help="optional text file of correct names, one per line")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("-o", "--out", default="single_result.csv")
    args = ap.parse_args()

    known = list(DEFAULT_KNOWN)
    if os.path.exists(args.known):
        known += [l.strip() for l in open(args.known, encoding="utf-8") if l.strip()]

    print(f"Downloading {args.start}s-{args.end}s ...")
    with tempfile.TemporaryDirectory() as wd:
        try:
            path, info = download_clip(args.url, args.start, args.end, wd,
                                       args.cookies, args.proxy)
        except Exception as e:
            print("\nDownload failed:", e)
            print("If this says 'not available in your country', run from an "
                  "Indian IP (no VPN, not Colab) or pass --proxy / --cookies.")
            sys.exit(1)

        title = info.get("title", "")
        print("Title:", title)
        print("Loading OCR model ...")
        reader = easyocr.Reader(["hi", "en"], gpu=args.gpu, verbose=False)

        if args.debug:
            os.makedirs("debug_frames", exist_ok=True)

        all_cands, first_ts = [], None
        hits = misses_after = 0
        for ts, frame in iter_frames(path, args.fps):
            real_ts = ts + args.start
            h = frame.shape[0]
            crop = frame[int(h * CROP_TOP):, :]
            res = reader.readtext(crop)
            kw = find_keyword(res)

            if args.debug:
                print(f"\n[{real_ts:6.1f}s] stage-1 OCR:")
                for _, text, conf in res:
                    print(f"    {conf:.2f}  {text}")

            if kw:
                hits += 1
                misses_after = 0
                if first_ts is None:
                    first_ts = real_ts
                print(f"[{real_ts:5.1f}s] keyword '{kw['text']}' ({kw['conf']:.2f}) -> reading names")
                tag = f"t{int(real_ts*10)}" if args.debug else None
                cands = read_names(reader, crop, kw, tag)
                all_cands.extend(cands)
            elif hits >= 2:
                misses_after += 1
                if misses_after >= 6:        # credit gone -> stop early
                    break

    names, conf = consensus(all_cands)
    snapped = snap_names(names, known) if names else []
    raw = " | ".join(names)
    final = " | ".join(snapped)
    review = "" if (names and conf >= 0.4 and len(all_cands) >= 3) else "yes"

    ep = episode_number(title)
    print("\n==== RESULT ====")
    print("Episode      :", ep)
    print("Writers      :", final or "(not found)")
    if names and raw != final:
        print("Raw OCR      :", raw, "  (snapped to known names)")
    print(f"Confidence   : {conf:.2f}   from {len(all_cands)} reading(s)")
    if first_ts is not None:
        print(f"Keyword first seen at {first_ts:.1f}s")
    if review:
        print("Needs manual review.")
    if not names:
        print("Tip: rerun with --debug; or widen the window, e.g. --start 0 --end 180")

    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["episode", "writers", "raw_ocr", "needs_review", "confidence",
                    "readings", "keyword_ts", "url", "title"])
        w.writerow([ep, final, raw, review, round(conf, 2), len(all_cands),
                    "" if first_ts is None else round(first_ts, 1), args.url, title])
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
