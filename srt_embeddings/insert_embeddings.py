import lancedb
import pyarrow as pa
import os
import glob
import argparse

from models.config import CONFIG
from srt_embeddings.utils import parse_srt_file
from models.embedding_model_factory import EmbeddingModelFactory

def main():
    parser = argparse.ArgumentParser(description='Process SRT files for embeddings')
    parser.add_argument('--path', type=str, help='Path to a specific SRT file or directory of SRT files')
    args = parser.parse_args()

    embedding_type = CONFIG["embedding_type"]
    db_config = CONFIG["databases"][embedding_type]
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    default_srt_directory = os.path.join(parent_dir, "yt-download")
    
    db_path = os.path.join(current_dir, db_config["path"])
    db = lancedb.connect(db_path)
    
    embedding_model = EmbeddingModelFactory.get_model(embedding_type, db_config)
    
    table_schema = embedding_model.get_schema()
    
    table_name = "transcripts"
    if table_name not in db.table_names():
        transcripts_table = db.create_table(table_name, schema=table_schema)
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
        
        embeddings = embedding_model.encode(texts)
        
        video_url = video_map.get(file_name, "")
        
        records = embedding_model.prepare_records(file_name, entries, embeddings, video_url)
        
        transcripts_table.merge_insert("id") \
            .when_matched_update_all() \
            .when_not_matched_insert_all() \
            .execute(records)
        
        transcripts_table.optimize()
        print(f"Processed {len(records)} records from {srt_file}")
    
    print(f"All files processed using '{embedding_type}' embedding model.")
    print(f"Database stored at: {db_path}")

if __name__ == '__main__':
    main()