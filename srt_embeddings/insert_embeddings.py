import lancedb
import pyarrow as pa
import os
import glob
import argparse
import torch
import numpy as np

from srt_embeddings.utils import parse_srt_file, generate_id
from pylate import models
from sentence_transformers import SentenceTransformer

def main():
    parser = argparse.ArgumentParser(description='Process SRT files for embeddings')
    parser.add_argument('--path', type=str, help='Path to a specific SRT file or directory of SRT files')
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    default_srt_directory = os.path.join(parent_dir, "yt-download")

    model_path = "/Users/dave/AI/models/gte-modernbert-base"
    modern_colbert_path = "/Users/dave/AI/models/GTE-ModernColBERT-v1"

    db = lancedb.connect(os.path.join(current_dir, "transcripts_lancedb"))

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")

    model = SentenceTransformer(model_path).to(device)
    batch_size = 4
    colbert_model = models.ColBERT(
        model_name_or_path=modern_colbert_path,
        document_length=1500,
        device=device
    )

    embedding_dimension = 768
    multivector_dimension = 128
    transcripts_schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("file_name", pa.string()),
        pa.field("video_url", pa.string()),
        pa.field("text", pa.string()),
        pa.field("start_time", pa.string()),
        pa.field("end_time", pa.string()),
        pa.field("start_seconds", pa.float64()),
        pa.field("end_seconds", pa.float64()),
        pa.field("embedding", pa.list_(pa.float32(), embedding_dimension)),
        pa.field("multivector_embedding", pa.list_(pa.list_(pa.float32(), multivector_dimension)))
    ])
    
    table_name = "transcripts"
    if table_name not in db.table_names():
        transcripts_table = db.create_table(table_name, schema=transcripts_schema)
        transcripts_table.create_scalar_index("id")
    else:
        transcripts_table = db.open_table(table_name)
    
    video_map = {}
    video_links_path = os.path.join(parent_dir, "yt-download", "video_links.txt")
    with open(video_links_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                video_map[parts[0]] = parts[1]
    
    if args.path:
        if os.path.isfile(args.path):
            srt_files = [args.path]
            print(f"Processing single file: {args.path}")
        elif os.path.isdir(args.path):
            srt_files = glob.glob(os.path.join(args.path, '*.srt'))
            print(f"Processing {len(srt_files)} files from directory: {args.path}")
        else:
            print(f"Error: The path '{args.path}' does not exist or is not accessible.")
            return
    else:
        srt_files = glob.glob(os.path.join(default_srt_directory, '*.srt'))
        print(f"Processing {len(srt_files)} files from default directory: {default_srt_directory}")

    for srt_file in srt_files:
        entries = parse_srt_file(srt_file)
        if not entries:
            continue

        file_name = os.path.splitext(os.path.basename(srt_file))[0]
        texts = [f"{file_name}\n\n{entry['text']}" for entry in entries]
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            device=device,
            show_progress_bar=False,
            convert_to_tensor=True,
            normalize_embeddings=True
        ).cpu().numpy().tolist()

        multivector_embeddings = colbert_model.encode(
            texts,
            batch_size=batch_size,
            is_query=False,
            show_progress_bar=False,
        )

        
        video_url = video_map.get(file_name, "")

        records = []
        for entry, embedding, multivector in zip(entries, embeddings, multivector_embeddings):
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
                'embedding': embedding,
                'multivector_embedding': multivector.tolist()
            })

        transcripts_table.merge_insert("id") \
            .when_matched_update_all() \
            .when_not_matched_insert_all() \
            .execute(records)

        transcripts_table.optimize()
        print(f"Processed {len(records)} records from {srt_file}")

    print("All files processed.")

if __name__ == '__main__':
    main()