import pyaudio
import wave
import os
import torch
import numpy as np
import librosa
import soundfile as sf
from collections import Counter

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment

# Optional color codes for console printing
NEON_GREEN = "\033[92m"
RESET_COLOR = "\033[0m"

def record_chunk(p, stream, file_path, chunk_length=1):
    """
    Records a short audio chunk (chunk_length in seconds)
    and saves it to 'file_path' as a WAV file.
    """
    frames = []
    for _ in range(0, int(16000 / 1024 * chunk_length)):
        data = stream.read(1024)
        frames.append(data)

    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()

def diarize_audio(pipeline, wav_path):
    """
    Runs speaker diarization on the WAV file and returns
    a pyannote.core.Annotation object containing speaker segments.
    """
    return pipeline(wav_path)

def transcribe_audio(whisper_model, wav_path):
    """
    Transcribes the audio using faster-whisper and returns a list of segments.
    Each segment has {start, end, text}.
    """
    segments, info = whisper_model.transcribe(wav_path, beam_size=7)
    result = []
    for seg in segments:
        # seg.start, seg.end, seg.text
        result.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip()
        })
    return result

def classify_speaker_type(audio_file, start_time, end_time):
    """
    Analyzes a section of audio to determine speaker demographic category.
    Returns "Man", "Woman", or "Child" based on voice characteristics.
    
    Uses multiple methods to increase reliability:
    1. Fundamental frequency analysis
    2. Spectral features
    """
    try:
        # Load audio file - explicitly set mono and normalized format
        y, sr = librosa.load(audio_file, sr=16000, mono=True)
        
        # Convert timestamps to samples
        start_sample = max(0, int(start_time * sr))
        end_sample = min(len(y), int(end_time * sr))
        
        # Check if we have a valid segment
        if start_sample >= end_sample or end_sample - start_sample < sr * 0.5:
            print(f"Segment too short or invalid: {start_time}-{end_time}")
            return "Unknown"
        
        # Extract the segment we want to analyze
        segment = y[start_sample:end_sample]
        
        # Normalize the segment
        segment = librosa.util.normalize(segment)
        
        # Method 1: Fundamental frequency using librosa's yin algorithm (more robust than pyin)
        f0, voiced_flag, voiced_probs = librosa.pyin(segment, 
                                                   fmin=60,      # minimum frequency in Hz
                                                   fmax=500,     # maximum frequency in Hz  
                                                   sr=sr)
        
        # Filter out NaN values and get only the voiced parts
        valid_f0 = f0[~np.isnan(f0)]
        
        # Method 2: Use spectral features
        # Spectral centroid (center of mass of the spectrum) - typically higher for women/children
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr))
        
        # Spectral rolloff (frequency below which 85% of the spectral energy lies)
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=segment, sr=sr))
        
        # Spectral flatness (measure of how noise-like vs. tone-like the sound is)
        flatness = np.mean(librosa.feature.spectral_flatness(y=segment))
        
        print(f"Audio analysis - F0 median: {np.median(valid_f0) if len(valid_f0) > 0 else 'N/A'}, " 
              f"Spectral centroid: {spectral_centroid}, Rolloff: {rolloff}")
        
        # Classification logic - combining multiple features for better accuracy
        if len(valid_f0) > 0:
            # We have reliable fundamental frequency data
            median_f0 = np.median(valid_f0)
            
            # Use multiple features for classification
            if median_f0 < 150 and spectral_centroid < 2000:
                return "Man"  # Low pitch and lower spectral centroid
            elif median_f0 > 250 or (spectral_centroid > 3000 and rolloff > 5000):
                return "Child"  # Very high pitch or bright spectral characteristics
            elif median_f0 >= 150:
                return "Woman"  # Medium-high pitch
            else:
                # Default to using just spectral characteristics
                if spectral_centroid < 2000:
                    return "Man"
                elif spectral_centroid > 3000:
                    return "Child"
                else:
                    return "Woman"
        else:
            # No reliable F0, use only spectral features
            if spectral_centroid < 2000:
                return "Man"
            elif spectral_centroid > 3000:
                return "Child"
            else:
                return "Woman"
                
    except Exception as e:
        print(f"Error in voice classification: {e}")
        return "Unknown"

def align_speakers_to_transcription(diarization_annotation, transcription_segments, audio_file):
    """
    Matches each transcription segment to the speaker from the diarization.
    Uses voice characteristics to determine speaker type (Man, Woman, Child).
    Returns a list of strings like: "[Man] text".
    """
    labeled_segments = []
    
    # Dictionary to cache speaker classifications
    speaker_types = {}

    # First, pre-process all speaker segments to identify clean samples for classification
    print("Pre-processing speaker segments for classification")
    all_speaker_segments = {}
    
    for seg, _, spk in diarization_annotation.itertracks(yield_label=True):
        if spk not in all_speaker_segments:
            all_speaker_segments[spk] = []
        all_speaker_segments[spk].append(seg)
    
    # Classify each speaker once using their longest segments
    for speaker_id, segments in all_speaker_segments.items():
        # Sort segments by duration (longest first)
        segments.sort(key=lambda s: s.duration, reverse=True)
        
        # Try to classify using up to 3 longest segments to improve accuracy
        classifications = []
        for i, seg in enumerate(segments[:3]):
            if seg.duration > 1.0:  # Only use segments longer than 1 second
                print(f"Classifying speaker {speaker_id} using segment {i+1} ({seg.start:.2f}-{seg.end:.2f}, duration: {seg.duration:.2f}s)")
                speaker_type = classify_speaker_type(audio_file, seg.start, seg.end)
                if speaker_type != "Unknown":
                    classifications.append(speaker_type)
        
        # Take the most common classification or default to "Unknown"
        if classifications:
            # Count occurrences of each classification
            from collections import Counter
            counter = Counter(classifications)
            speaker_types[speaker_id] = counter.most_common(1)[0][0]
            print(f"Speaker {speaker_id} classified as {speaker_types[speaker_id]} (votes: {counter})")
        else:
            speaker_types[speaker_id] = "Unknown"
            print(f"Speaker {speaker_id} could not be classified reliably")

    # Process each transcription segment
    for tseg in transcription_segments:
        t_start = tseg["start"]
        t_end = tseg["end"]

        # We'll represent the transcription segment as a pyannote Segment
        trans_segment = Segment(t_start, t_end)

        # Find overlapping diarization segments
        overlapping_speakers = diarization_annotation.crop(
            trans_segment, mode="intersection"
        )

        if len(overlapping_speakers) == 0:
            # No speaker found (maybe silence?), just label as "Unknown"
            speaker_label = "Unknown"
        else:
            # Find the speaker with largest overlap
            speaker_id = None
            max_overlap = 0.0

            for speech_turn, _, spk_id in overlapping_speakers.itertracks(yield_label=True):
                # speech_turn is a pyannote.core.Segment for the overlap
                overlap_duration = speech_turn.duration
                if overlap_duration > max_overlap:
                    max_overlap = overlap_duration
                    speaker_id = spk_id
            
            # If we have a speaker ID, use our pre-classified type
            if speaker_id is not None and speaker_id in speaker_types:
                speaker_label = speaker_types[speaker_id]
            else:
                speaker_label = "Unknown"

        # Build the final text line
        line = f"[{speaker_label}] {tseg['text']}"
        labeled_segments.append(line)

    return labeled_segments

def extract_speaker_audio(audio_path, diarization, speaker_id, output_folder="speaker_samples"):
    """
    Extract audio samples for a specific speaker and save them to separate files.
    This helps with debugging voice classification issues.
    """
    # Create output folder if needed
    os.makedirs(output_folder, exist_ok=True)
    
    # Load the full audio
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    # Find all segments for this speaker
    speaker_segments = []
    for segment, _, label in diarization.itertracks(yield_label=True):
        if label == speaker_id:
            speaker_segments.append(segment)
    
    # Extract and save up to 3 longest segments
    speaker_segments.sort(key=lambda seg: seg.duration, reverse=True)
    for i, segment in enumerate(speaker_segments[:3]):
        if segment.duration < 1.0:  # Skip very short segments
            continue
            
        # Convert start/end times to sample indices
        start_sample = int(segment.start * sr)
        end_sample = int(segment.end * sr)
        
        # Make sure we don't exceed audio bounds
        if start_sample >= len(audio) or end_sample > len(audio):
            continue
            
        # Extract the segment
        segment_audio = audio[start_sample:end_sample]
        
        # Save to file
        output_path = f"{output_folder}/speaker_{speaker_id}_segment_{i+1}.wav"
        sf.write(output_path, segment_audio, sr)
        print(f"Saved speaker {speaker_id} sample to {output_path}")
        
    return speaker_segments[:3]

def main():
    """
    Repeatedly:
    1) Records a 15-second audio chunk to 'temp_chunk.wav'
    2) Diarizes that chunk to find speaker segments
    3) Transcribes that chunk with faster-whisper
    4) Classifies each speaker as Man, Woman, or Child
    5) Prints and appends the speaker-labeled text to 'log.txt'
    """

    # 1) Initialize Models
    whisper_model = WhisperModel(
        "medium",          # or "small", "large-v2", etc.
        device="cuda",     # or "cpu" if no GPU
        compute_type="float16"
    )

    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.getenv("HF_API_KEY", "YOUR_HF_API_KEY")
    )
    # Move to GPU if available
    diarization_pipeline.to(torch.device("cuda"))

    # 2) Set Up Audio Input
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=1024
    )

    accumulated_transcription = ""

    # Reusable temp files (we'll overwrite them each loop)
    wav_file = "temp_chunk.wav"
    
    # Enable debug mode (set to False to disable verbose output)
    debug_mode = True

    try:
        while True:
            print("\n--- New Recording Session ---")
            # (A) Record a 15-second chunk
            print("Recording audio...")
            record_chunk(p, stream, wav_file, chunk_length=30)

            # (B) Diarize
            print("Diarizing speakers...")
            diarization_annotation = diarize_audio(diarization_pipeline, wav_file)
            
            # Optional: Extract debug samples for speaker classification
            if debug_mode:
                print("Extracting speaker samples for analysis...")
                speaker_ids = set()
                for _, _, speaker in diarization_annotation.itertracks(yield_label=True):
                    speaker_ids.add(speaker)
                
                for speaker_id in speaker_ids:
                    extract_speaker_audio(wav_file, diarization_annotation, speaker_id)

            # (C) Transcribe
            print("Transcribing audio...")
            transcription_segments = transcribe_audio(whisper_model, wav_file)

            # (D) Align speakers -> labeled text lines with speaker classification
            print("Classifying speakers and aligning transcription...")
            labeled_transcript_lines = align_speakers_to_transcription(
                diarization_annotation,
                transcription_segments,
                wav_file
            )

            # Join them into a block for printing
            block_text = "\n".join(labeled_transcript_lines)
            print("\nTranscript with speaker labels:")
            print(NEON_GREEN + block_text + RESET_COLOR)

            # Append to our in-memory transcript
            accumulated_transcription += block_text + "\n"

    except KeyboardInterrupt:
        print("Stopping...")

        # (E) Final: Write all accumulated text to log.txt
        with open("log.txt", "w", encoding="utf-8") as log_file:
            log_file.write(accumulated_transcription)

    finally:
        print("LOG:\n" + accumulated_transcription)
        # Clean up PyAudio
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
