import yt_dlp
import os

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
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(channel_url, download=False)
        video_entries = info_dict.get('entries', [])

        for entry in video_entries[:num_videos]:
            video_title = entry.get('title', 'unknown')
            output_filename = f"{video_title}.mp3".replace("|","｜").replace("?","？")
            
            if os.path.exists(output_filename):
                print(f"File '{output_filename}' already exists. Skipping...")
                continue
            
            ydl.download([entry['webpage_url']])

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