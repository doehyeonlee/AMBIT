"""
Download OpenAI batch results to correct filenames.
Run from project root: python scripts/download_openai_results.py
"""
from openai import OpenAI
import os

client = OpenAI()
out_dir = "outputs/one_batch/openai"
os.makedirs(out_dir, exist_ok=True)

# JobFair: chunk015 (newest) → chunk000 (oldest)
jobfair_outputs = [
    "file-KtTcvrj4Wa5nCoecMUrD3y",   # chunk015
    "file-M6zpCL5hGdym1PZgDdy6oV",   # chunk014
    "file-E3GwxyL8zM6sDn3JDoSS8L",   # chunk013
    "file-3aFohAiEV4orPUJiXh46sY",   # chunk012
    "file-EhFEshqE4KcMn4mxiLEzGt",   # chunk011
    "file-KvCU5oJxq8DosGYfDvi4Tz",   # chunk010
    "file-MNU3Fxi2bqvqh71NiAsNjb",   # chunk009
    "file-5ux48ojp4RXHD8P6FYQ8hG",   # chunk008
    "file-9mLaABXbiEg1qWnEwRaAze",   # chunk007
    "file-V8uuD5Udu79uSM5UwHBJgs",   # chunk006
    "file-DdCdVEtzKh9X8wo4Uch77h",   # chunk005
    "file-RZHNJ1fvGHDUmddZYGYLWH",   # chunk004
    "file-81DXhmpWceVawVdwcjpf9B",   # chunk003
    "file-K41YbrQcAtSLWf7b6xG4da",   # chunk002
    "file-NWLykQJvqgkygczWT6BPMM",   # chunk001
    "file-1kzMhRZS3LDfqEs44aJt7r",   # chunk000
]

# LBOX: chunk015 (newest) → chunk000 (oldest)
lbox_outputs = [
    "file-KB3SEqonyWarVGdhvJnRpW",   # chunk015
    "file-Lj5XWeco8gZDou9KHgWp6u",   # chunk014
    "file-UnHeT2QGXJ4iRqpEePWkTx",   # chunk013
    "file-1gvtvJS2Jak5LwEGgBpWk9",   # chunk012
    "file-NezUDUqsnp76HsU3pNj6zM",   # chunk011
    "file-6gHidUbBtuBEZHw2weDrdD",   # chunk010
    "file-8yKnQ4pXr4x5ZuiS8RHTMN",   # chunk009
    "file-EvPA3cvB9iPDNKs1CNnXbn",   # chunk008
    "file-Vh97ZDfQBPVRLHpC1uEaP1",   # chunk007
    "file-KepF7Dtn1eujPS7hfkbawG",   # chunk006
    "file-EBL8HkBefSZLNpgBNo35vT",   # chunk005
    "file-Dkzz9gawZWuPNgTcd992T3",   # chunk004
    "file-PraPe9Vi6TN9cJR7kzf13M",   # chunk003
    "file-Fhg88dyv4iHCA48aarNVGE",   # chunk002
    "file-M3w7irpnqD7q9DkAoiHyhm",   # chunk001
    "file-Su9n4vqxiBvDVdHBeSujPH",   # chunk000
]

def download(file_ids, dataset):
    n = len(file_ids)
    for i, fid in enumerate(file_ids):
        chunk_num = n - 1 - i  # reverse: first in list = highest chunk
        fname = f"results_{dataset}_gpt-4.1-mini_chunk{chunk_num:03d}.jsonl"
        path = os.path.join(out_dir, fname)
        if os.path.exists(path):
            print(f"  SKIP (exists): {fname}")
            continue
        print(f"  Downloading {fname}...", end=" ", flush=True)
        content = client.files.content(fid).text
        with open(path, "w") as f:
            f.write(content)
        print(f"OK ({len(content.splitlines())} lines)")

print("=== JobFair ===")
download(jobfair_outputs, "jobfair")

print("\n=== LBOX ===")
download(lbox_outputs, "lbox")

print("\nDone. Verify:")
print(f"  ls {out_dir}/results_jobfair_gpt-4.1-mini_*.jsonl | wc -l  # should be 16")
print(f"  ls {out_dir}/results_lbox_gpt-4.1-mini_*.jsonl | wc -l     # should be 16")