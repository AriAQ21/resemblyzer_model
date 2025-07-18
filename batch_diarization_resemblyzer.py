import argparse
import os
import time
import csv
import io
import sys
import librosa
import numpy as np
from diarization import main  

def batch_process(input_folder, output_folder, n_speakers=2, chunk_length=1.5, num_files=None):
    os.makedirs(output_folder, exist_ok=True)

    audio_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.wav')])
    if num_files:
        audio_files = audio_files[:num_files]

    metrics_file = os.path.join(output_folder, "metrics.csv")
    with open(metrics_file, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['filename', 'time_taken_s', 'num_segments', 'audio_duration_s', 'avg_segment_duration_s']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        total_start = time.time()

        for filename in audio_files:
            audio_path = os.path.join(input_folder, filename)
            print(f"Processing {filename}...")

            start_time = time.time()

            # Capture diarization printed output
            old_stdout = sys.stdout
            sys.stdout = mystdout = io.StringIO()

            main(audio_path, n_speakers=n_speakers, chunk_length=chunk_length)

            sys.stdout = old_stdout
            diarization_text = mystdout.getvalue()

            # Save diarization output text file
            out_txt_path = os.path.join(output_folder, filename.replace('.wav', '.txt'))
            with open(out_txt_path, "w", encoding="utf-8") as f:
                f.write(diarization_text)

            # Parse diarization_text for segment info
            num_segments, total_segment_duration = parse_segments(diarization_text)

            # Get audio duration using librosa
            wav, sr = librosa.load(audio_path, sr=None)
            audio_duration = len(wav) / sr

            elapsed = time.time() - start_time
            avg_segment_duration = total_segment_duration / num_segments if num_segments > 0 else 0

            print(f"Finished {filename} in {elapsed:.2f}s, {num_segments} segments, audio length: {audio_duration:.2f}s")

            writer.writerow({
                'filename': filename,
                'time_taken_s': f"{elapsed:.2f}",
                'num_segments': num_segments,
                'audio_duration_s': f"{audio_duration:.2f}",
                'avg_segment_duration_s': f"{avg_segment_duration:.2f}"
            })

        total_elapsed = time.time() - total_start
        print(f"\nTotal processing time: {total_elapsed:.2f}s for {len(audio_files)} files")

def parse_segments(text):
    """
    Parses diarization output text like:
    Speaker 1: 0.0s - 1.5s
    Speaker 2: 1.5s - 3.0s
    Returns number of segments and total duration.
    """
    num_segments = 0
    total_duration = 0.0
    for line in text.splitlines():
        if line.startswith("Speaker"):
            try:
                parts = line.split(":")[1].strip().split(" - ")
                start = float(parts[0].replace("s", ""))
                end = float(parts[1].replace("s", ""))
                total_duration += (end - start)
                num_segments += 1
            except Exception:
                continue
    return num_segments, total_duration

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch diarization with Resemblyzer")
    parser.add_argument("input_folder", type=str, help="Folder with WAV audio files")
    parser.add_argument("output_folder", type=str, help="Folder for diarization results and metrics")
    parser.add_argument("--n_speakers", type=int, default=2, help="Number of speakers")
    parser.add_argument("--chunk_length", type=float, default=1.5, help="Chunk length in seconds")
    parser.add_argument("--num_files", type=int, default=None, help="Number of files to process")
    args = parser.parse_args()
    batch_process(args.input_folder, args.output_folder, args.n_speakers, args.chunk_length, args.num_files)
