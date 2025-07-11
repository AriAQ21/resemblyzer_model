import argparse
import numpy as np
import librosa
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import AgglomerativeClustering
import soundfile as sf

def get_audio_chunks(wav, sr, chunk_length=1.5):
    """Split audio into fixed-length chunks (seconds)."""
    chunk_samples = int(chunk_length * sr)
    chunks = []
    for start in range(0, len(wav), chunk_samples):
        chunk = wav[start:start + chunk_samples]
        if len(chunk) == chunk_samples:
            chunks.append(chunk)
    return chunks

def main(audio_path, n_speakers=2, chunk_length=1.5):
    wav, sr = librosa.load(audio_path, sr=16000)
    print(f"Loaded audio {audio_path} at {sr} Hz")

    chunks = get_audio_chunks(wav, sr, chunk_length)
    print(f"Split audio into {len(chunks)} chunks of {chunk_length} seconds")

    encoder = VoiceEncoder()
    embeddings = []

    for i, chunk in enumerate(chunks):
        # Resemblyzer expects a wav file or numpy array, preprocess accordingly
        emb = encoder.embed_utterance(chunk)
        embeddings.append(emb)

    embeddings = np.vstack(embeddings)

    clustering = AgglomerativeClustering(n_clusters=n_speakers)
    labels = clustering.fit_predict(embeddings)

    # Print chunk start time and assigned speaker
    for i, label in enumerate(labels):
        start_time = i * chunk_length
        end_time = start_time + chunk_length
        print(f"Speaker {label + 1}: {start_time:.1f}s - {end_time:.1f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple diarization with Resemblyzer")
    parser.add_argument("audio", type=str, help="Path to audio WAV file (16 kHz mono)")
    parser.add_argument("--n_speakers", type=int, default=2, help="Number of speakers to cluster")
    parser.add_argument("--chunk_length", type=float, default=1.5, help="Chunk length in seconds")
    args = parser.parse_args()

    main(args.audio, args.n_speakers, args.chunk_length)
