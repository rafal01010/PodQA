import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os
import time

device = "mps" if torch.backends.mps.is_available() else "cpu"
torch_dtype = torch.float16 if torch.backends.mps.is_available() else torch.float32

# model_id = "openai/whisper-large-v3-turbo"

# Can change to local model path
model_id = "/Users/dave/AI/models/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    return_timestamps=True,
    chunk_length_s=30,
)

audio_directory = "../yt-download"

# Function to convert timestamps to SRT format
def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

# Process each audio file in the directory
for audio_filename in os.listdir(audio_directory):
    if audio_filename.endswith(('.mp3', '.wav', '.flac')):  # Add more formats if needed
        audio_path = os.path.join(audio_directory, audio_filename)
        
        # Construct the SRT file path
        audio_base_name = os.path.splitext(audio_filename)[0]
        srt_file_path = os.path.join(audio_directory, f"{audio_base_name}.srt")

        # Check if SRT file already exists
        if os.path.exists(srt_file_path):
            print(f"SRT file already exists for {audio_filename}. Skipping transcription.")
            continue

        print(f"Processing {audio_filename}")
        start_time = time.time()

        result = pipe(audio_path)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time for {audio_filename}: {elapsed_time} seconds")

        # Prepare the SRT content
        srt_content = []
        for i, segment in enumerate(result["chunks"], start=1):
            start_time = format_timestamp(segment["timestamp"][0])
            end_time = format_timestamp(segment["timestamp"][1])
            
            text = segment["text"].strip()
            srt_content.append(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")

        # Save the SRT content to a file
        with open(srt_file_path, "w") as srt_file:
            srt_file.writelines(srt_content)

        print(f"SRT file saved to {srt_file_path}")