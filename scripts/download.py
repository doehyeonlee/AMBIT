# 1. 결과 파일 복구 (Dashboard에서 output file ID로)

from openai import OpenAI
client = OpenAI()
import json

for b in client.batches.list(limit=100):
    if b.status != 'completed' or not b.output_file_id:
        continue
    # input file에서 원래 chunk 이름 추출
    inp = client.files.retrieve(b.input_file_id)
    fname = inp.filename  # e.g. jobfair_gpt-4.1-mini_chunk003.jsonl
    
    content = client.files.content(b.output_file_id).text
    out_path = f'outputs/one_batch/openai/results_{fname}'
    with open(out_path, 'w') as f:
        f.write(content)
    print(f'Saved: {out_path} ({len(content.splitlines())} lines)')