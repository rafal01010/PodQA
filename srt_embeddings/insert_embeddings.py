import lancedb
import pyarrow as pa
import pyarrow.compute as pc
import os
import glob
from sentence_transformers import SentenceTransformer
import torch
import time
import gc

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
                
            entries.append({
                'text': ' '.join(line.strip() for line in lines[2:]),
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
    """Convert SRT time format (HH:MM:SS,mmm) to total seconds."""
    hms, ms = time_str.split(',') if ',' in time_str else (time_str, '000')
    ms = ms.ljust(3, '0')[:3]
    hours, minutes, seconds = hms.split(':')
    return (
        int(hours) * 3600 + 
        int(minutes) * 60 + 
        int(seconds) + 
        int(ms) / 1000
    )

def main():
    srt_directory = "../yt-download"

    # Change model_path to local path or HF path
    model_path = "/Users/dave/AI/models/gte-modernbert-base"

    db = lancedb.connect("./transcripts_lancedb")

    if not os.path.isdir(srt_directory):
        print(f"Error: {srt_directory} is not a valid directory")
        return

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")
    
    batch_size = 16
    model = SentenceTransformer(model_path).to(device)

    srt_files = glob.glob(os.path.join(srt_directory, '*.srt'))
    
    if not srt_files:
        print(f"No SRT files found in {srt_directory}")
        return
    
    embedding_dimension = 768
    transcripts_schema = pa.schema([
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
        db.create_table("transcripts", schema=transcripts_schema)

    
    video_map = {}
    with open('../yt-download/video_links.txt', 'r') as file:
        for line in file:
            stripped_line = line.strip()
            if not stripped_line:
                continue
            
            parts = stripped_line.split('\t')
            if len(parts) != 2:
                print(f"Skipping invalid line: {stripped_line}")
                continue
            
            file_name, video_url = parts
            video_map[file_name] = video_url

    for srt_file in srt_files:

        entries = parse_srt_file(srt_file)
        if not entries:
            print(f"No valid entries found in {srt_file}")
            continue

        texts = [entry['text'] for entry in entries]

        print(f"Starting embedding {srt_file}")
        start_time = time.time()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            device=device,
            show_progress_bar=False,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        embeddings = embeddings.cpu().numpy().tolist()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time for {srt_file}: {elapsed_time} seconds")

        file_name = os.path.splitext(os.path.basename(srt_file))[0]
        video_url = video_map.get(file_name, "")

        records = []
        for entry, embedding in zip(entries, embeddings):
            records.append({
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

        transcripts_table = db.open_table("transcripts")
        transcripts_table.add(table)
        print(f"Inserted {len(records)} records from {srt_file}")

    print("All files processed.")

if __name__ == '__main__':
    main()