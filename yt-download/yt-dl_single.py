import yt_dlp
import os

def download_youtube_video_as_mp3(youtube_url, audio_quality):
    ydl_opts = {
        'format': f'bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': audio_quality,
        }],
        'outtmpl': '%(title)s.%(ext)s',
        'quiet': False,
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        video_title = info_dict.get('title', 'unknown_video').replace("|","｜").replace("?","？")
        output_filename = f"{video_title}.mp3"

        with open('video_links.txt', 'a', encoding='utf-8') as f:
            output_filename = f"{video_title}.mp3"

            if os.path.exists(output_filename):
                print(f"The file '{output_filename}' already exists. Skipping download.")
                return

            ydl.download([youtube_url])
            f.write(f"{video_title}\t{youtube_url}\n")
            print(f"Downloaded '{output_filename}' successfully.")

if __name__ == "__main__":
    youtube_url = input("Enter the YouTube URL: ")
    audio_quality = input("Enter the desired audio quality (e.g., 192): ")

    download_youtube_video_as_mp3(youtube_url, audio_quality)