import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os
import time

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.backends.mps.is_available() or torch.cuda.is_available()  else torch.float32

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

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    remaining_after_hours = seconds % 3600
    minutes = int(remaining_after_hours // 60)
    remaining_after_minutes = remaining_after_hours % 60
    seconds_int = int(remaining_after_minutes)
    milliseconds = int(round((remaining_after_minutes - seconds_int) * 1000))
    return f"{hours:02d}:{minutes:02d}:{seconds_int:02d},{milliseconds:03d}"

reprocess_list = []

for audio_filename in os.listdir(audio_directory):
    reprocess = False
    if audio_filename.endswith(('.mp3', '.wav', '.flac')):
        audio_path = os.path.join(audio_directory, audio_filename)
        
        audio_base_name = os.path.splitext(audio_filename)[0]
        srt_file_path = os.path.join(audio_directory, f"{audio_base_name}.srt")

        if os.path.exists(srt_file_path):
            print(f"SRT file already exists for {audio_filename}. Skipping transcription.")
            continue

        print(f"Processing {audio_filename}")
        start_time = time.time()
        result = pipe(audio_path)
        

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time for {audio_filename}: {elapsed_time} seconds")

        srt_content = []
        current_id = 1

        for segment in result["chunks"]:
            text = segment["text"].strip()
            if not text:
                continue

            if len(text) > 10000:
                reprocess= True
                break
            
            start_timestamp = segment["timestamp"][0] if segment["timestamp"][0] is not None else segment["timestamp"][1]
            end_timestamp = segment["timestamp"][1] if segment["timestamp"][1] is not None else segment["timestamp"][0]
            start_time = format_timestamp(start_timestamp)
            end_time = format_timestamp(end_timestamp)
            
            srt_content.append(f"{current_id}\n{start_time} --> {end_time}\n{text}\n\n")
            current_id += 1
        if reprocess:
            reprocess_list.append(audio_filename)
            continue
        with open(srt_file_path, "w") as srt_file:
            srt_file.writelines(srt_content)
        print(f"SRT file saved to {srt_file_path}")

if reprocess_list:

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps="word",
        chunk_length_s=30,
    )

    for audio_filename in reprocess_list:
        if audio_filename.endswith(('.mp3', '.wav', '.flac')):
            audio_path = os.path.join(audio_directory, audio_filename)
            
            audio_base_name = os.path.splitext(audio_filename)[0]
            srt_file_path = os.path.join(audio_directory, f"{audio_base_name}.srt")

            if os.path.exists(srt_file_path):
                print(f"SRT file already exists for {audio_filename}. Skipping transcription.")
                continue

            print(f"Processing {audio_filename}")
            start_time = time.time()

            result = pipe(audio_path)

            srt_content = []
            current_id = 1

            current_segment_start = None
            current_segment_end = None
            current_segment_text = []

            for segment in result["chunks"]:
                word = segment["text"].strip()
                start = segment["timestamp"][0] if segment["timestamp"][0] is not None else segment["timestamp"][1]
                end = segment["timestamp"][1] if segment["timestamp"][1] is not None else segment["timestamp"][0]

                if not word:
                    continue

                if current_segment_start is None:
                    current_segment_start = start
                    current_segment_end = end
                    current_segment_text.append(word)
                else:
                    potential_duration = end - current_segment_start
                    if potential_duration <= 15:
                        current_segment_end = end
                        current_segment_text.append(word)
                    else:
                        srt_content.append(
                            f"{current_id}\n"
                            f"{format_timestamp(current_segment_start)} --> {format_timestamp(current_segment_end)}\n"
                            f"{' '.join(current_segment_text)}\n\n"
                        )
                        current_id += 1
                        current_segment_start = start
                        current_segment_end = end
                        current_segment_text = [word]

            if current_segment_text:
                srt_content.append(
                    f"{current_id}\n"
                    f"{format_timestamp(current_segment_start)} --> {format_timestamp(current_segment_end)}\n"
                    f"{' '.join(current_segment_text)}\n\n"
                )

            with open(srt_file_path, "w") as srt_file:
                srt_file.writelines(srt_content)
            print(f"SRT file saved to {srt_file_path}")