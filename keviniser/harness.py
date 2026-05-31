"""Corpus runner for the Keviniser.

Streams a corpus through the Keviniser line by line and logs the two numbers
doc 1 calls the headline metric:

  - word-count compression (raw)
  - tokeniser-count compression (what actually decides how much fits on-chip)

The two will not move by the same amount, which is exactly why both are logged.

The final model's tokeniser does not exist yet, so the token count uses a GPT-2
BPE proxy via tiktoken when it is installed. It is a stand-in for relative
comparison, not the on-chip budget itself. If tiktoken is missing the run still
completes and reports word stats only.

Usage:
    python -m keviniser.harness tinystories.txt -o tinystories.kevin.txt
    cat corpus.txt | python -m keviniser.harness > corpus.kevin.txt
    python -m keviniser.harness corpus.txt --no-objectify --keep-punct --stats stats.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from itertools import tee

from .keviniser import kevinise_stream


def _read_lines(path: str | None):
    """Yield stripped, non-empty lines from a file path, or stdin if path is None."""
    stream = sys.stdin if path is None else open(path, "r", encoding="utf-8")
    try:
        for line in stream:
            line = line.strip()
            if line:
                yield line
    finally:
        if stream is not sys.stdin:
            stream.close()


def _load_token_counter():
    """Return a function str -> token count, or None if tiktoken is unavailable."""
    try:
        import tiktoken
    except ImportError:
        return None
    enc = tiktoken.get_encoding("gpt2")
    # disallowed_special=() so corpus text containing a literal "<|endoftext|>"
    # is counted as ordinary tokens rather than raising.
    return lambda s: len(enc.encode(s, disallowed_special=()))


def run(
    in_path: str | None,
    out_path: str | None,
    *,
    lemmatise: bool = True,
    objectify: bool = True,
    keep_punct: bool = False,
    subs: dict | None = None,
    marker: str | None = None,
    n_process: int = 1,
    tokens: bool = True,
    stats_path: str | None = None,
    progress_every: int = 5000,
    quiet: bool = False,
) -> dict:
    """Kevinise a corpus, write the output, and return a stats dict.

    Single streaming pass, so memory stays flat on large corpora: the source is
    tee'd, one branch counts and interleaves markers, the other (markers
    filtered out) feeds a single nlp.pipe. n_process > 1 fans spaCy across cores.

    Lines equal to `marker` (e.g. TinyStories' "<|endoftext|>" story delimiter)
    are passed through to the output verbatim and excluded from the stats, so
    document boundaries survive into the Kevinised corpus.
    """
    count_tokens = _load_token_counter() if tokens else None

    def log(msg: str) -> None:
        if not quiet:
            sys.stderr.write(msg)
            sys.stderr.flush()

    if tokens and count_tokens is None:
        log("note: tiktoken not installed, reporting word counts only "
            "(pip install tiktoken for token compression)\n")

    words_before = words_after = 0
    tokens_before = tokens_after = 0
    lines = markers = 0
    start = time.perf_counter()

    src_count, src_kev = tee(_read_lines(in_path))
    content = (ln for ln in src_kev if marker is None or ln != marker)
    kevs = iter(kevinise_stream(
        content,
        lemmatise=lemmatise,
        objectify=objectify,
        keep_punct=keep_punct,
        subs=subs,
        n_process=n_process,
    ))

    out_ctx = (
        contextlib.nullcontext(sys.stdout)
        if out_path is None
        else open(out_path, "w", encoding="utf-8")
    )
    with out_ctx as out:
        for original in src_count:
            if marker is not None and original == marker:
                out.write(original + "\n")
                markers += 1
                continue

            kev = next(kevs)
            out.write(kev + "\n")

            words_before += len(original.split())
            words_after += len(kev.split())
            if count_tokens is not None:
                tokens_before += count_tokens(original)
                tokens_after += count_tokens(kev)

            lines += 1
            if progress_every and lines % progress_every == 0:
                el = time.perf_counter() - start
                rate = lines / el if el else 0
                log(f"\r  {lines:,} lines | {words_before:,} -> {words_after:,} words "
                    f"| {rate:,.0f} lines/s")

    elapsed = time.perf_counter() - start
    if progress_every:
        log("\r" + " " * 70 + "\r")  # clear the progress line

    stats = {
        "lines": lines,
        "markers": markers,
        "words_before": words_before,
        "words_after": words_after,
        "word_ratio": (words_after / words_before) if words_before else 0.0,
        "tokens_before": tokens_before if count_tokens else None,
        "tokens_after": tokens_after if count_tokens else None,
        "token_ratio": (
            (tokens_after / tokens_before) if (count_tokens and tokens_before) else None
        ),
        "seconds": round(elapsed, 2),
        "knobs": {
            "lemmatise": lemmatise,
            "objectify": objectify,
            "keep_punct": keep_punct,
        },
    }

    _print_summary(stats, log)

    if stats_path:
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        log(f"stats written to {stats_path}\n")

    return stats


def _print_summary(stats: dict, log) -> None:
    wb, wa = stats["words_before"], stats["words_after"]
    wr = stats["word_ratio"]
    log(f"\nKeviniser done: {stats['lines']:,} lines in {stats['seconds']}s\n")
    log(f"  words   {wb:,} -> {wa:,}  ({wr:.1%} of original)\n")
    if stats["token_ratio"] is not None:
        tb, ta = stats["tokens_before"], stats["tokens_after"]
        tr = stats["token_ratio"]
        log(f"  tokens  {tb:,} -> {ta:,}  ({tr:.1%} of original, gpt2 proxy)\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="keviniser.harness",
        description="Run the Keviniser over a corpus and log compression stats.",
    )
    p.add_argument(
        "input",
        nargs="?",
        default=None,
        help="input corpus file (one document or sentence per line). "
        "Reads stdin if omitted.",
    )
    p.add_argument(
        "-o", "--output", default=None,
        help="output file for Kevinised text. Writes stdout if omitted.",
    )
    p.add_argument(
        "--stats", default=None,
        help="write the stats summary to this path as JSON.",
    )
    p.add_argument(
        "--no-lemmatise", dest="lemmatise", action="store_false",
        help="keep original inflection instead of flattening it.",
    )
    p.add_argument(
        "--no-objectify", dest="objectify", action="store_false",
        help="leave subject pronouns alone instead of objectifying them.",
    )
    p.add_argument(
        "--keep-punct", dest="keep_punct", action="store_true",
        help="keep sentence-break punctuation.",
    )
    p.add_argument(
        "--subs", default=None, metavar="PATH",
        help='JSON object of word substitutions merged over the defaults, '
        'e.g. {"shin": "shine", "television": "tv"}. Applied as the final step '
        "to each kept word.",
    )
    p.add_argument(
        "--marker", default=None, metavar="STR",
        help='document delimiter to pass through verbatim and exclude from '
        'stats, e.g. "<|endoftext|>" for TinyStories.',
    )
    p.add_argument(
        "--nproc", type=int, default=1, metavar="N",
        help="fan spaCy across N processes (the parser is the slow part, so this "
        "scales well). Use for the full train split.",
    )
    p.add_argument(
        "--no-tokens", dest="tokens", action="store_false",
        help="skip the tiktoken token count (it is serial and caps --nproc "
        "scaling on huge corpora). Word compression is still reported.",
    )
    p.add_argument(
        "-q", "--quiet", action="store_true",
        help="suppress progress and the stats summary on stderr.",
    )
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    subs = None
    if args.subs:
        with open(args.subs, "r", encoding="utf-8") as f:
            subs = json.load(f)
    run(
        args.input,
        args.output,
        lemmatise=args.lemmatise,
        objectify=args.objectify,
        keep_punct=args.keep_punct,
        subs=subs,
        marker=args.marker,
        n_process=args.nproc,
        tokens=args.tokens,
        stats_path=args.stats,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
