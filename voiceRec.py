
import librosa
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import warnings
from transformers.utils import logging
#trynna surpress terminal bloat
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.set_verbosity_error()
#these ones i can alter
audio_path = "audio.mp3"
tolerance = 0.4
context = "paint in feet"
#these are setting the models and the transformers
embedder = SentenceTransformer("all-MiniLM-L6-v2")
context_vec = embedder.encode([context])
processor = AutoProcessor.from_pretrained("openai/whisper-base")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-base")

audio, sr = librosa.load(audio_path, sr=16000)
inputs = processor(
    audio,
    sampling_rate=16000,
    return_tensors="pt"
)
#thus generating the transcripts
with torch.no_grad():
    generated_ids = model.generate(inputs["input_features"])
#this decoder
transcription = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True
)

print(transcription[0])

chunk_duration = 10  # seconds
chunk_samples = 16000 * chunk_duration

chunks = [
    audio[i:i + chunk_samples]
    for i in range(0, len(audio), chunk_samples)
]

drift_history = []

for chunk in chunks:
    inputs = processor(chunk, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        ids = model.generate(inputs.input_features)

    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    if len(text.strip()) < 10:
        continue

    chunk_vec = embedder.encode([text])

    similarity = cosine_similarity(context_vec, chunk_vec)[0][0]
    drift = 1 - similarity
    drift_history.append(drift)
    if len(drift_history) > 3:
        drift_history.pop(0)

    drift_avg = sum(drift_history) / len(drift_history)

    print("Text: ", text)
    print("Drift: ",drift_avg)
    
    if drift_avg > tolerance:
        print("Topic drift detected")