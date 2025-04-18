import lancedb
import pyarrow as pa
import os
import glob
import hashlib
from sentence_transformers import SentenceTransformer
import torch
import time

def generate_id(file_name, start_seconds, end_seconds, text):
    unique_str = f"{file_name}_{start_seconds}_{end_seconds}_{text}"
    return hashlib.md5(unique_str.encode('utf-8')).hexdigest()

def parse_srt_file(srt_path):
    entries = []
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = content.strip().split('\n\n')
    
    for block in blocks:
        lines = block.split('\n')
        if len(lines) < 3:
            continue

        try:
            timecode = lines[1].strip()
            start_time, end_time = timecode.split(' --> ')
            start_seconds = srt_time_to_seconds(start_time)
            end_seconds = srt_time_to_seconds(end_time)

            if end_seconds <= start_seconds:
                continue

            text = ' '.join(line.strip() for line in lines[2:])
            entries.append({
                'text': text,
                'start_time': start_time,
                'end_time': end_time,
                'start_seconds': start_seconds,
                'end_seconds': end_seconds
            })
        except (ValueError, IndexError) as e:
            print(f"Skipping invalid block in {srt_path}: {e}")
            continue

    return entries


def srt_time_to_seconds(time_str):
    hms, ms = time_str.split(',') if ',' in time_str else (time_str, '000')
    ms = ms.ljust(3, '0')[:3]
    hours, minutes, seconds = hms.split(':')
    return int(hours)*3600 + int(minutes)*60 + int(seconds) + int(ms)/1000


def main():
    srt_directory = "../yt-download"
    model_path = "/Users/dave/AI/models/gte-modernbert-base"

    db = lancedb.connect("./transcripts_lancedb")

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")

    model = SentenceTransformer(model_path).to(device)
    batch_size = 16

    srt_files = glob.glob(os.path.join(srt_directory, '*.srt'))

    embedding_dimension = 768
    transcripts_schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("file_name", pa.string()),
        pa.field("video_url", pa.string()),
        pa.field("text", pa.string()),
        pa.field("start_time", pa.string()),
        pa.field("end_time", pa.string()),
        pa.field("start_seconds", pa.float64()),
        pa.field("end_seconds", pa.float64()),
        pa.field("embedding", pa.list_(pa.float32(), embedding_dimension))
    ])

    if "transcripts" not in db.table_names():
        transcripts_table = db.create_table("transcripts", schema=transcripts_schema)
        transcripts_table.create_scalar_index("id")
    else:
        transcripts_table = db.open_table("transcripts")

    video_map = {}
    with open('../yt-download/video_links.txt', 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                video_map[parts[0]] = parts[1]

    for srt_file in srt_files:
        entries = parse_srt_file(srt_file)
        if not entries:
            continue

        texts = [entry['text'] for entry in entries]
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            device=device,
            show_progress_bar=False,
            convert_to_tensor=True,
            normalize_embeddings=True
        ).cpu().numpy().tolist()

        file_name = os.path.splitext(os.path.basename(srt_file))[0]
        video_url = video_map.get(file_name, "")

        records = []
        for entry, embedding in zip(entries, embeddings):
            record_id = generate_id(file_name, entry['start_seconds'], entry['end_seconds'], entry['text'])
            records.append({
                'id': record_id,
                'file_name': file_name,
                'video_url': video_url,
                'text': entry['text'],
                'start_time': entry['start_time'],
                'end_time': entry['end_time'],
                'start_seconds': entry['start_seconds'],
                'end_seconds': entry['end_seconds'],
                'embedding': embedding
            })

        table = pa.Table.from_pylist(records, schema=transcripts_schema)

        transcripts_table.merge_insert("id") \
            .when_matched_update_all() \
            .when_not_matched_insert_all() \
            .execute(records)

        transcripts_table.optimize()
        print(f"Processed {len(records)} records from {srt_file}")

    print("All files processed.")

if __name__ == '__main__':
    main()
