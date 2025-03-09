import sqlite3
import os
import glob

def parse_srt_file(srt_path):
    """Parse an SRT file and return a list of subtitle entries."""
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

    if not os.path.isdir(srt_directory):
        print(f"Error: {srt_directory} is not a valid directory")
        return

    conn = sqlite3.connect("transcripts.db")
    cur = conn.cursor()

    # Find all SRT files in the directory
    srt_files = glob.glob(os.path.join(srt_directory, '*.srt'))
    
    if not srt_files:
        print(f"No SRT files found in {srt_directory}")
        return

    for srt_file in srt_files:
        base_name = os.path.splitext(os.path.basename(srt_file))[0]
        cur.execute('SELECT id FROM source_files WHERE file_name = ?', (base_name,))
        result = cur.fetchone()
        
        if not result:
            print(f"Skipping {srt_file} - no matching entry in source_files")
            continue
            
        source_file_id = result[0]
        entries = parse_srt_file(srt_file)
        
        for entry in entries:
            cur.execute('''INSERT INTO transcripts (
                text, start_time, end_time, 
                start_seconds, end_seconds, source_file_id
            ) VALUES (?, ?, ?, ?, ?, ?)''', (
                entry['text'],
                entry['start_time'],
                entry['end_time'],
                entry['start_seconds'],
                entry['end_seconds'],
                source_file_id
            ))
        
        print(f"Inserted {len(entries)} entries from {os.path.basename(srt_file)}")
    
    conn.commit()
    conn.close()

if __name__ == '__main__':
    main()