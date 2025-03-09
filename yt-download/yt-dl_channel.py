import yt_dlp
import os
import time

def download_mp3_from_channel(channel_url, num_videos, audio_quality='192'):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': audio_quality,
        }],
        'noplaylist': False,
        'playlistend': None if num_videos == 'all' else num_videos,
        'outtmpl': '%(title)s.%(ext)s',
        'ignoreerrors': True,
        'nooverwrites': True,
        'noprogress': True,
        'retries': 5,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(channel_url, download=False)
        video_entries = info_dict.get('entries', [])

        end_index = None if num_videos == 'all' else num_videos
        with open('video_links.txt', 'a', encoding='utf-8') as f:
            for entry in video_entries[:end_index]:
                video_title = entry.get('title', 'unknown').replace("|", "｜").replace("?", "？")
                video_url = entry.get('webpage_url', 'unknown')
                output_filename = f"{video_title}.mp3"
                
                if os.path.exists(output_filename):
                    print(f"File '{output_filename}' already exists. Skipping...")
                    continue
                
                ydl.download([entry['webpage_url']])
                f.write(f"{video_title}\t{video_url}\n")

if __name__ == "__main__":
    channel_url = input("Enter the YouTube channel URL: ")
    num_videos_input = input("Enter the number of videos to download (or 'all' to download all videos): ")
    audio_quality_input = input("Enter the audio quality (e.g., 320 for 320 kbps) [default: 192]: ")

    if num_videos_input.lower() == 'all':
        num_videos = 'all'
    else:
        num_videos = int(num_videos_input)

    if not audio_quality_input.strip():
        audio_quality = '192'
    else:
        audio_quality = audio_quality_input.strip()

    download_mp3_from_channel(channel_url, num_videos, audio_quality)