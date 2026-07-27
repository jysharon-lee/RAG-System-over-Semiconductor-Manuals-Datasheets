import json
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

chunks = json.load(open("data/processed_chunks/TPS61030.json"))
print("=== Page 4 Tables ===")
for c in chunks:
    if c['type'] == 'table_row' and c['page_number'] == 4:
        print(c['content'])

print("\n=== Page 5 Tables (first 15 rows) ===")
count = 0
for c in chunks:
    if c['type'] == 'table_row' and c['page_number'] == 5:
        print(c['content'])
        count += 1
        if count >= 15: break
