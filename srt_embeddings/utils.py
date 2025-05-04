import hashlib

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