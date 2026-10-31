"""prep_corpus.py — build ref/full_test/corpus.bin (raw UTF-8 byte stream).

Usage:
  python3 ref/prep_corpus.py --jsonl ~/train_clean.jsonl   # SFT data
  python3 ref/prep_corpus.py --text  ~/mybook.txt          # plain text
  python3 ref/prep_corpus.py                               # demo corpus
"""
import argparse, json, sys
from pathlib import Path

B = Path.home() / 'mmc_publish' / 'ref' / 'full_test'
B.mkdir(parents=True, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument('--jsonl', help='path to SFT jsonl')
ap.add_argument('--text', help='path to plain text file')
args = ap.parse_args()

docs = []
if args.jsonl:
    p = Path(args.jsonl).expanduser()
    if not p.exists():
        sys.exit(f"❌ not found: {p}")
    n_bad = 0
    for line in p.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            n_bad += 1
            continue
        if isinstance(d, dict):
            if isinstance(d.get('text'), str):
                docs.append(d['text'])
            elif isinstance(d.get('messages'), list):
                for msg in d['messages']:
                    if isinstance(msg, dict) and isinstance(msg.get('content'), str):
                        docs.append(msg['content'])
            elif isinstance(d.get('content'), str):
                docs.append(d['content'])
        elif isinstance(d, str):
            docs.append(d)
    print(f"jsonl: {len(docs)} docs  ({n_bad} bad lines skipped)")
elif args.text:
    p = Path(args.text).expanduser()
    if not p.exists():
        sys.exit(f"❌ not found: {p}")
    docs = [p.read_text(encoding='utf-8')]
else:
    print("no source given — writing DEMO corpus (bilingual, ~120KB)")
    en = "The quick brown fox jumps over the lazy dog."
    my = "မြန်မာစာ စမ်းသပ်မှု စာသား။"
    docs = [f"{en} {my} sample line {i}" for i in range(1500)]

data = ("\n".join(docs) + "\n").encode('utf-8')
out = B / 'corpus.bin'
out.write_bytes(data)

uniq = len(set(data))
steps_per_epoch = len(data) // 128
print(f"✅ wrote {len(data):,} bytes -> {out}")
print(f"   unique bytes: {uniq}/256")
print(f"   1 epoch @ SEQ=128 ≈ {steps_per_epoch:,} steps (batch=1)")
if len(data) < 2_000_000:
    print("   ⚠️  corpus < 2MB — model will overfit/memorize quickly;")
    print("      consider a larger corpus for meaningful language learning")
if len(data) < 130:
    sys.exit("❌ corpus too small (need >= 130 bytes)")
