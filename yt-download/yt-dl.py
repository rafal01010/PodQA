import yt_dlp
import os
from yt_dlp.utils import sanitize_filename

def download_mp3_from_channel(channel_url, num_videos, audio_quality='192'):
    ydl_opts = {
        'format': 'bestaudio/best',  # Get the best audio quality available
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': audio_quality,  # Set the audio quality
        }],
        'noplaylist': False,  # Download all videos from the playlist or channel
        'playlistend': num_videos,  # Limit the number of videos to download
        'outtmpl': '%(title)s.%(ext)s',  # Output filename template
        'ignoreerrors': True,  # Ignore errors to ensure the script continues
        'nooverwrites': True,  # Do not overwrite existing files
        'noprogress': True,  # Do not show the progress bar (optional)
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # Fetch the playlist information to get the list of video URLs
        info_dict = ydl.extract_info(channel_url, download=False)
        video_entries = info_dict.get('entries', [])

        for entry in video_entries[:num_videos]:
            video_title = entry.get('title', 'unknown')
            output_filename = f"{video_title}.mp3".replace("|","｜").replace("?","？")
            
            
            # Check if the output file already exists
            if os.path.exists(output_filename):
                print(f"File '{output_filename}' already exists. Skipping...")
                continue
            
            # Download the video and perform post-processing
            ydl.download([entry['webpage_url']])

if __name__ == "__main__":
    channel_url = input("Enter the YouTube channel URL: ")
    num_videos = int(input("Enter the number of videos to download: "))
    audio_quality = input("Enter the audio quality (e.g., 320 for 320 kbps): ")
    download_mp3_from_channel(channel_url, num_videos, audio_quality)