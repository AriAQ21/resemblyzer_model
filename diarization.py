import argparse
import numpy as np
from resemblyzer import VoiceEncoder
from sklearn.cluster import AgglomerativeClustering
from pyannote.audio import Pipeline
from pyannote.core import Segment
import torchaudio
import os

def main(audio_path, n_speakers=2, chunk_length=None):
    # Authenticate with Hugging Face
    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN not found in environment. Please set it.")

    # Load VAD pipeline
    vad = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token=token)
    speech_regions = vad(audio_path).get_timeline()

    # Load Resemblyzer encoder
    encoder = VoiceEncoder()
    signal, sr = torchaudio.load(audio_path)
    assert sr == 16000, "Audio must be sampled at 16 kHz"

    embeddings = []
    segments = []

    for region in speech_regions:
        start_sample = int(region.start * sr)
        end_sample = int(region.end * sr)
        chunk = signal[:, start_sample:end_sample]
        if chunk.size(1) < int(0.5 * sr):  # Skip very short chunks
            continue

        emb = encoder.embed_utterance(chunk.squeeze().numpy())
        embeddings.append(emb)
        segments.append((region.start, region.end))

    if len(embeddings) < 2:
        print("⚠️ Not enough segments for clustering.")
        return

    embeddings = np.vstack(embeddings)

    clustering = AgglomerativeClustering(n_clusters=n_speakers)
    labels = clustering.fit_predict(embeddings)

    # Output diarization
    print("Diarization results:")
    for (start, end), label in zip(segments, labels):
        print(f"Speaker {label + 1}: {start:.2f}s - {end:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resemblyzer Diarization with Pyannote VAD")
    parser.add_argument("audio", type=str, help="Path to 16kHz mono WAV file")
    parser.add_argument("--n_speakers", type=int, default=2, help="Number of speakers to cluster")
    args = parser.parse_args()

    main(args.audio, args.n_speakers)
