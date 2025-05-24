import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os
import time

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if torch.backends.mps.is_available() or torch.cuda.is_available()  else torch.float32

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
    return_timestamps="word",
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

        # Store all segments first
        all_segments = []
        for segment in result["chunks"]:
            word = segment["text"].strip()
            start = segment["timestamp"][0] if segment["timestamp"][0] is not None else segment["timestamp"][1]
            end = segment["timestamp"][1] if segment["timestamp"][1] is not None else segment["timestamp"][0]
            
            if word:
                all_segments.append({
                    "word": word,
                    "start": start,
                    "end": end,
                    "ends_with_punct": word.endswith(('.', '!', '?'))
                })
        
        srt_content = []
        current_id = 1
        
        if not all_segments:
            print(f"No segments found for {audio_filename}")
            continue
            
        # Initialize first segment
        current_segment_start = all_segments[0]["start"]
        current_segment_text = []
        segment_index = 0
        
        while segment_index < len(all_segments):
            # Build up current segment
            current_segment_text = []
            current_segment_start = all_segments[segment_index]["start"]
            current_segment_end = None
            segment_duration = 0
            
            # Build segment until we hit the duration limit (220 seconds)
            while segment_index < len(all_segments) and segment_duration <= 220:
                segment = all_segments[segment_index]
                current_segment_text.append(segment["word"])
                current_segment_end = segment["end"]
                segment_duration = current_segment_end - current_segment_start
                
                # Check if we should end this segment
                ends_with_punct = segment["ends_with_punct"]
                if ends_with_punct and segment_duration >= 210:
                    segment_index += 1  # Move to next word for the next segment
                    break
                
                segment_index += 1
            
            # If we ran out of segments or didn't find a good end point
            if segment_index >= len(all_segments) or segment_duration > 220:
                # Just end at the last processed word
                pass
            
            # Add this segment to SRT file
            srt_content.append(
                f"{current_id}\n"
                f"{format_timestamp(current_segment_start)} --> {format_timestamp(current_segment_end)}\n"
                f"{' '.join(current_segment_text)}\n\n"
            )
            current_id += 1
            
            # Calculate where the next segment should start (with overlap)
            if segment_index < len(all_segments):
                # Find the timestamp that's about 12 seconds before the end of current segment
                overlap_start_time = current_segment_end - 12
                
                # Find the word that corresponds to this timestamp
                overlap_index = segment_index - 1  # Start from the last word
                
                # Go backwards to find the right starting point for the overlap
                while overlap_index >= 0:
                    if all_segments[overlap_index]["start"] <= overlap_start_time:
                        break
                    overlap_index -= 1
                
                # Make sure we found a valid overlap point
                if overlap_index >= 0:
                    # But also check that we don't have more than 18 seconds overlap
                    max_overlap_time = current_segment_end - 18
                    
                    if all_segments[overlap_index]["start"] < max_overlap_time:
                        # Too much overlap, find a better point
                        temp_index = overlap_index
                        while temp_index < segment_index - 1:
                            temp_index += 1
                            if all_segments[temp_index]["start"] >= max_overlap_time:
                                overlap_index = temp_index
                                break
                    
                    # Set the index for the next segment to start at the overlap point
                    segment_index = overlap_index
                    
                # If we couldn't find a good overlap point, next segment will start from current position
                # segment_index is already at the right position
        
        # Write the SRT file
        with open(srt_file_path, "w") as srt_file:
            srt_file.writelines(srt_content)
        print(f"SRT file saved to {srt_file_path}")