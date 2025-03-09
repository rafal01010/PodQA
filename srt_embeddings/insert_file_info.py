import sqlite3

conn = sqlite3.connect('transcripts.db')
cursor = conn.cursor()

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
        cursor.execute('''
            INSERT INTO source_files (file_name, video_url)
            VALUES (?, ?)
        ''', (file_name, video_url))

conn.commit()
conn.close()