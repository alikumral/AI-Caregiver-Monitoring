import os
import numpy as np
import torch
import torch.nn as nn
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import time
import concurrent.futures
from collections import Counter, defaultdict
import traceback
import argparse
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import AgglomerativeClustering
import webrtcvad
from pathlib import Path
import wave
import contextlib
import struct

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
from resemblyzer import VoiceEncoder, preprocess_wav

# Color codes for printing
NEON_GREEN = "\033[92m"
RESET_COLOR = "\033[0m"
RED = "\033[91m"
BLUE = "\033[94m"

# Hugging Face API key
HF_API_KEY = os.getenv("HF_API_KEY", "YOUR_HF_API_KEY")  # Replace with your key

# Define the Audeering model for age and gender recognition
class ModelHead(nn.Module):
    """Classification head."""
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class AgeGenderModel(Wav2Vec2PreTrainedModel):
    """Age and gender classifier."""
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)
        return hidden_states, logits_age, logits_gender

def load_audeering_model(device='cpu'):
    """Load the Audeering age-gender model."""
    print("Loading Audeering age-gender model...")
    model_name = 'audeering/wav2vec2-large-robust-24-ft-age-gender'
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = AgeGenderModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    return processor, model

def process_audio_segment(processor, model, audio, sampling_rate=16000, device='cpu'):
    """
    Process an audio segment through the Audeering model to get age and gender predictions.
    """
    # Ensure audio is float32 and properly shaped
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    # Normalize if needed
    if np.abs(audio).max() > 1.0:
        audio = audio / np.abs(audio).max()
    
    # run through processor to normalize signal
    y = processor(audio, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    y = y.reshape(1, -1)
    y = torch.from_numpy(y).to(device)

    # run through model
    with torch.no_grad():
        hidden_states, age_preds, gender_preds = model(y)
        
    # Convert to numpy
    age = age_preds.detach().cpu().numpy()[0][0]  # 0-1 scale (0-100 years)
    gender = gender_preds.detach().cpu().numpy()[0]  # [female, male, child] probabilities
    
    return {
        'age': age * 100,  # Convert to years (0-100)
        'gender_probs': {
            'female': gender[0],
            'male': gender[1],
            'child': gender[2]
        }
    }

def convert_mp3_to_wav(mp3_path, wav_path=None):
    """
    Convert MP3 to WAV format for processing.
    """
    # If no wav_path is specified, create one based on the mp3_path
    if wav_path is None:
        wav_path = os.path.splitext(mp3_path)[0] + ".wav"
    
    # Check if WAV already exists to avoid re-conversion
    if os.path.exists(wav_path):
        print(f"WAV file already exists at: {wav_path}")
        return wav_path
    
    print(f"Converting MP3 to WAV: {mp3_path} -> {wav_path}")
    
    try:
        # Load the MP3 file
        y, sr = librosa.load(mp3_path, sr=None)
        
        # Write to WAV
        sf.write(wav_path, y, sr)
        print(f"Conversion successful. WAV file saved at: {wav_path}")
        return wav_path
    except Exception as e:
        print(f"Error converting MP3 to WAV: {e}")
        traceback.print_exc()
        return None

def diarize_with_pyannote(audio_path):
    """
    Runs speaker diarization on the audio file using pyannote.
    """
    start_time = time.time()
    print("Running diarization with Pyannote...")
    
    try:
        # Load pyannote speaker diarization model
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_API_KEY
        )
        
        if torch.cuda.is_available():
            diarization_pipeline.to(torch.device("cuda"))
        
        # Run diarization
        result = diarization_pipeline(audio_path)
        print(f"Pyannote diarization completed in {time.time() - start_time:.2f} seconds")
        return result
    
    except Exception as e:
        print(f"Error in Pyannote diarization: {e}")
        traceback.print_exc()
        return None

class VoiceActivityDetector:
    """VAD based on WebRTC to detect voice activity in audio."""
    
    def __init__(self, sample_rate=16000, frame_duration=30, aggressive=3):
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration  # in milliseconds
        self.vad = webrtcvad.Vad(aggressive)
        self.frame_size = int(sample_rate * frame_duration / 1000)
    
    def is_speech(self, audio_frame):
        """Check if a frame contains speech."""
        try:
            return self.vad.is_speech(audio_frame, self.sample_rate)
        except:
            return False
    
    def detect_speech_segments(self, audio, min_speech_duration=0.3):
        """Detect speech segments in audio."""
        # Convert to int16 for VAD
        if audio.dtype != np.int16:
            if audio.dtype == np.float32 or audio.dtype == np.float64:
                # Normalize and convert to int16
                audio = (audio * 32767).astype(np.int16)
            else:
                audio = audio.astype(np.int16)
        
        # Process audio in frames
        frames = []
        for i in range(0, len(audio) - self.frame_size + 1, self.frame_size):
            frames.append(audio[i:i + self.frame_size])
        
        # Check each frame for speech
        is_speech = []
        for frame in frames:
            # Pack frame into bytes for WebRTC VAD
            packed_frame = struct.pack("h" * len(frame), *frame)
            is_speech.append(self.is_speech(packed_frame))
        
        # Combine consecutive speech frames into segments
        speech_segments = []
        in_speech = False
        start = 0
        
        for i, speech in enumerate(is_speech):
            if speech and not in_speech:
                in_speech = True
                start = i
            elif not speech and in_speech:
                in_speech = False
                duration = (i - start) * self.frame_duration / 1000
                if duration >= min_speech_duration:
                    speech_segments.append((
                        start * self.frame_duration / 1000,
                        i * self.frame_duration / 1000
                    ))
        
        # Handle the case where the audio ends during speech
        if in_speech:
            duration = (len(is_speech) - start) * self.frame_duration / 1000
            if duration >= min_speech_duration:
                speech_segments.append((
                    start * self.frame_duration / 1000,
                    len(is_speech) * self.frame_duration / 1000
                ))
        
        return speech_segments

def extract_segments(audio, sr, speech_segments, margin=0.1):
    """Extract audio segments with a margin."""
    segments = []
    for start, end in speech_segments:
        # Add margin
        seg_start = max(0, start - margin)
        seg_end = min(len(audio) / sr, end + margin)
        
        # Extract segment
        start_sample = int(seg_start * sr)
        end_sample = int(seg_end * sr)
        segment = audio[start_sample:end_sample]
        
        segments.append((segment, seg_start, seg_end))
    
    return segments

def diarize_with_resemblyzer(audio_path, whisper_segments=None, num_speakers=None):
    """
    Runs speaker diarization on the audio file using Resemblyzer.
    
    Args:
        audio_path: Path to the audio file
        whisper_segments: Optional list of segment dictionaries from Whisper with 'start' and 'end' fields
        num_speakers: Optional number of speakers to cluster
        
    Returns:
        An Annotation object similar to pyannote output
    """
    start_time = time.time()
    print("Running diarization with Resemblyzer...")
    
    try:
        # Load the audio file
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        duration = len(audio) / sr
        
        # Initialize Resemblyzer
        encoder = VoiceEncoder()
        
        # If we have Whisper segments, use those as our initial segmentation
        if whisper_segments:
            print("Using Whisper segments for initial segmentation...")
            speech_segments = [(seg['start'], seg['end']) for seg in whisper_segments]
        else:
            # Use VAD to detect speech segments
            print("Using WebRTC VAD for initial segmentation...")
            vad = VoiceActivityDetector()
            speech_segments = vad.detect_speech_segments(audio)
        
        # Extract audio segments
        extracted_segments = extract_segments(audio, sr, speech_segments)
        
        if not extracted_segments:
            print("No speech segments detected!")
            return None
        
        # Extract speaker embeddings for each segment
        embeddings = []
        valid_segments = []
        
        print(f"Processing {len(extracted_segments)} segments...")
        for segment, start, end in extracted_segments:
            # Skip very short segments
            if len(segment) < sr * 0.5:  # skip segments shorter than 0.5s
                continue
                
            # Process with Resemblyzer
            segment_wav = preprocess_wav(segment, source_sr=sr)
            embedding = encoder.embed_utterance(segment_wav)
            
            embeddings.append(embedding)
            valid_segments.append((start, end))
        
        if not embeddings:
            print("No valid embeddings extracted!")
            return None
        
        # Determine number of speakers through clustering if not provided
        if num_speakers is None:
            # Try to estimate using clustering
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.7,  # Adjust this threshold
                affinity='cosine',
                linkage='average'
            )
        else:
            clustering = AgglomerativeClustering(
                n_clusters=num_speakers,
                affinity='cosine',
                linkage='average'
            )
        
        # Convert embeddings to numpy array for clustering
        embeddings_array = np.array(embeddings)
        labels = clustering.fit_predict(embeddings_array)
        
        # Create a pyannote-compatible Annotation object
        diarization = Annotation()
        
        for (start, end), label in zip(valid_segments, labels):
            segment = Segment(start, end)
            speaker_id = f"SPEAKER_{label:02d}"
            diarization[segment] = speaker_id
        
        print(f"Resemblyzer diarization completed in {time.time() - start_time:.2f} seconds")
        print(f"Detected {len(set(labels))} speakers")
        
        return diarization
    
    except Exception as e:
        print(f"Error in Resemblyzer diarization: {e}")
        traceback.print_exc()
        return None

def transcribe_with_whisper(audio_path, model_size="small", language=None):
    """
    Transcribes the audio using faster-whisper with speed optimizations.
    """
    start_time = time.time()
    print("Transcribing audio with Whisper...")
    
    try:
        # Use compute_type=int8 for faster transcription if available
        compute_type = "int8" if torch.cuda.is_available() else "float32"
        
        # Initialize Whisper model
        whisper_model = WhisperModel(
            model_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type=compute_type
        )
        
        # Transcribe
        segments, info = whisper_model.transcribe(
            audio_path, 
            beam_size=5,
            language=language,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        # Format results
        result = []
        for seg in segments:
            result.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip()
            })
        print(f"Whisper transcription completed in {time.time() - start_time:.2f} seconds")
        return result
    
    except Exception as e:
        print(f"Error in Whisper transcription: {e}")
        traceback.print_exc()
        return []

def extract_speaker_segments(audio_path, diarization, min_duration=1.0, output_dir=None):
    """
    Extracts audio segments for each speaker.
    """
    if output_dir is None:
        output_dir = "speaker_samples"
        
    start_time = time.time()
    print(f"Extracting speaker segments to {output_dir}...")
    
    try:
        # Load the full audio
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Dictionary to store speaker audio
        speaker_segments = {}
        speaker_audio = {}  # Store audio segments for each speaker
        
        # Process each speaker
        for segment, _, speaker_id in diarization.itertracks(yield_label=True):
            if segment.duration < min_duration:
                continue  # Skip short segments
                
            if speaker_id not in speaker_segments:
                speaker_segments[speaker_id] = []
            speaker_segments[speaker_id].append(segment)
        
        # Process segments for each speaker
        for speaker_id, segments in speaker_segments.items():
            # Sort segments by duration (longest first)
            segments.sort(key=lambda s: s.duration, reverse=True)
            
            # Take the top 5 longest segments
            valid_segments = [s for s in segments[:5] if s.duration >= min_duration]
            
            if not valid_segments:
                continue
                
            # Create a directory for saving speaker samples
            os.makedirs(output_dir, exist_ok=True)
            
            # Collect audio segments for this speaker
            all_audio_segments = []
            
            for i, segment in enumerate(valid_segments[:3]):  # Limit to 3 samples per speaker
                start_sample = int(segment.start * sr)
                end_sample = int(segment.end * sr)
                
                if start_sample >= len(audio) or end_sample > len(audio):
                    continue
                    
                segment_audio = audio[start_sample:end_sample]
                all_audio_segments.append(segment_audio)
                
                # Save the segment for model usage
                output_path = f"{output_dir}/{speaker_id}_segment_{i+1}.wav"
                sf.write(output_path, segment_audio, sr)
            
            # Store concatenated audio for this speaker
            if all_audio_segments:
                speaker_audio[speaker_id] = np.concatenate(all_audio_segments)
                print(f"Speaker {speaker_id} audio extracted - {len(speaker_audio[speaker_id])/sr:.2f}s")
        
        print(f"Speaker extraction completed in {time.time() - start_time:.2f} seconds")
        return speaker_audio
    
    except Exception as e:
        print(f"Error extracting speaker segments: {e}")
        traceback.print_exc()
        return {}

def classify_speakers_with_audeering(speaker_audio, processor, model, device='cpu'):
    """
    Classify speakers using the Audeering age-gender model.
    Returns classifications and detailed results.
    """
    print("\nClassifying speakers with Audeering age-gender model...")
    
    classifications = {}
    detailed_results = {}
    confidence_scores = {}
    
    for speaker_id, audio in speaker_audio.items():
        # Process audio through the model
        result = process_audio_segment(processor, model, audio, device=device)
        
        # Get age and gender probabilities
        age = result['age']
        female_prob = result['gender_probs']['female']
        male_prob = result['gender_probs']['male']
        child_prob = result['gender_probs']['child']
        
        # Determine the most likely classification
        if child_prob > max(female_prob, male_prob) and child_prob > 0.4:
            label = "Child"
            confidence = child_prob
        elif female_prob > male_prob:
            label = "Woman"  # Using "Woman" instead of "Female" for consistency
            confidence = female_prob
        else:
            label = "Man"  # Using "Man" instead of "Male" for consistency
            confidence = male_prob
        
        # Store classification and detailed results
        classifications[speaker_id] = label
        confidence_scores[speaker_id] = {
            "Man": male_prob,
            "Woman": female_prob,
            "Child": child_prob
        }
        detailed_results[speaker_id] = {
            'classification': label,
            'age_estimate': age,
            'gender_probabilities': {
                'female': female_prob,
                'male': male_prob,
                'child': child_prob
            },
            'confidence': confidence
        }
        
        print(f"Speaker {speaker_id} â†’ {label} " +
              f"(Age: {age:.1f} years, Confidence: {confidence:.2f}, " +
              f"Probabilities: M={male_prob:.2f}, F={female_prob:.2f}, C={child_prob:.2f})")
    
    return classifications, confidence_scores, detailed_results

def extract_text_by_speaker(transcription, diarization):
    """
    Extract all text spoken by each speaker.
    """
    speaker_text = {}
    segment_mapping = []
    
    for segment in transcription:
        # Find which speaker was talking during this segment
        t_start, t_end = segment['start'], segment['end']
        trans_segment = Segment(t_start, t_end)
        
        # Find overlapping speaker
        overlapping_speakers = diarization.crop(trans_segment, mode="intersection")
        if not overlapping_speakers:
            # No speaker found
            segment_mapping.append({
                'segment': segment,
                'speaker_id': "Unknown",
                'overlap_duration': 0
            })
            continue
            
        # Get most overlapping speaker
        speaker_id = None
        max_overlap = 0
        
        for speech_turn, _, spk_id in overlapping_speakers.itertracks(yield_label=True):
            overlap_duration = speech_turn.duration
            if overlap_duration > max_overlap:
                max_overlap = overlap_duration
                speaker_id = spk_id
        
        if speaker_id:
            # Store segment mapping
            segment_mapping.append({
                'segment': segment,
                'speaker_id': speaker_id,
                'overlap_duration': max_overlap
            })
            
            # Add to speaker text collection
            if speaker_id not in speaker_text:
                speaker_text[speaker_id] = []
            speaker_text[speaker_id].append(segment['text'])
    
    # Combine all text for each speaker
    for speaker_id in speaker_text:
        speaker_text[speaker_id] = " ".join(speaker_text[speaker_id])
    
    return speaker_text, segment_mapping

def format_timestamp(seconds):
    """
    Formats seconds as HH:MM:SS
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def create_transcript(segment_mapping, speaker_classifications, confidence_scores, include_timestamps=True):
    """
    Creates a transcript from the segment mapping and speaker classifications.
    """
    labeled_segments = []
    
    for mapping in segment_mapping:
        segment = mapping['segment']
        speaker_id = mapping['speaker_id']
        
        t_start = segment["start"]
        t_end = segment["end"]
        
        # Format timestamps as HH:MM:SS
        if include_timestamps:
            start_formatted = format_timestamp(t_start)
            end_formatted = format_timestamp(t_end)
            timestamp_str = f"[{start_formatted} - {end_formatted}] "
        else:
            timestamp_str = ""
        
        # Get speaker label and confidence
        if speaker_id in speaker_classifications:
            speaker_label = speaker_classifications[speaker_id]
            confidence = confidence_scores.get(speaker_id, {}).get(speaker_label, 0.5)
        else:
            speaker_label = "Unknown"
            confidence = 0.0
        
        # Build the final text line with confidence score
        confidence_indicator = f"({confidence:.2f})" if confidence > 0 else ""
        line = f"{timestamp_str}[{speaker_label}{confidence_indicator}] {segment['text']}"
        labeled_segments.append(line)
    
    return labeled_segments

def visualize_speaker_classification(detailed_results, method_name=""):
    """
    Create a visualization of speaker classifications and probabilities.
    """
    if not detailed_results:
        print("No speaker data to visualize")
        return
    
    # Prepare data for plotting
    speaker_ids = list(detailed_results.keys())
    ages = [detailed_results[spk]['age_estimate'] for spk in speaker_ids]
    man_probs = [detailed_results[spk]['gender_probabilities']['male'] for spk in speaker_ids]
    woman_probs = [detailed_results[spk]['gender_probabilities']['female'] for spk in speaker_ids]
    child_probs = [detailed_results[spk]['gender_probabilities']['child'] for spk in speaker_ids]
    classifications = [detailed_results[spk]['classification'] for spk in speaker_ids]
    
    # Set up plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot 1: Probability Stacked Bar Chart
    bar_width = 0.6
    indices = np.arange(len(speaker_ids))
    
    # Create stacked bar chart
    ax1.bar(indices, man_probs, bar_width, color='blue', alpha=0.7, label='Man')
    ax1.bar(indices, woman_probs, bar_width, bottom=man_probs, color='red', alpha=0.7, label='Woman')
    
    # Calculate bottom for child scores
    bottoms = [m+w for m, w in zip(man_probs, woman_probs)]
    ax1.bar(indices, child_probs, bar_width, bottom=bottoms, color='green', alpha=0.7, label='Child')
    
    # Mark the final classification with stars
    for i, cls in enumerate(classifications):
        color = 'blue' if cls == 'Man' else 'red' if cls == 'Woman' else 'green'
        ax1.plot(indices[i], 1.05, marker='*', color=color, markersize=10)
    
    # Set up the axes for probability chart
    ax1.set_title(f"{method_name} Speaker Classification Probabilities")
    ax1.set_xlabel('Speaker ID')
    ax1.set_ylabel('Probability')
    ax1.set_xticks(indices)
    ax1.set_xticklabels(speaker_ids, rotation=45)
    ax1.set_ylim(0, 1.2)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Plot 2: Age Estimates
    colors = ['blue' if cls == 'Man' else 'red' if cls == 'Woman' else 'green' for cls in classifications]
    ax2.bar(indices, ages, color=colors, alpha=0.7)
    
    # Add age labels on top of bars
    for i, age in enumerate(ages):
        ax2.text(indices[i], age + 2, f"{age:.1f}", ha='center')
    
    # Set up the axes for age chart
    ax2.set_title(f"{method_name} Estimated Speaker Ages")
    ax2.set_xlabel('Speaker ID')
    ax2.set_ylabel('Age (years)')
    ax2.set_xticks(indices)
    ax2.set_xticklabels(speaker_ids, rotation=45)
    ax2.set_ylim(0, max(ages) * 1.2)
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # Add a legend for age colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.7, label='Man'),
        Patch(facecolor='red', alpha=0.7, label='Woman'),
        Patch(facecolor='green', alpha=0.7, label='Child')
    ]
    ax2.legend(handles=legend_elements)
    
    plt.tight_layout()
    
    # Save with method name
    filename = f"speaker_classification_{method_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=150)
    print(f"Speaker classification visualization saved to {filename}")

def create_detailed_report(output_txt, method_name, speaker_classifications, confidence_scores, detailed_results):
    """
    Create a detailed report of speaker classifications.
    """
    report_path = output_txt.replace(".txt", f"_{method_name.lower().replace(' ', '_')}_report.txt")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"SPEAKER CLASSIFICATION DETAILED REPORT - {method_name}\n")
        f.write("=" * (42 + len(method_name)) + "\n\n")
        
        # Get all unique speaker IDs
        all_speakers = sorted(list(detailed_results.keys()))
        
        f.write("SUMMARY\n")
        f.write("-------\n")
        counts = Counter(speaker_classifications.values())
        f.write(f"Total speakers identified: {len(speaker_classifications)}\n")
        for speaker_type in ["Man", "Woman", "Child", "Unknown"]:
            count = counts.get(speaker_type, 0)
            f.write(f"  {speaker_type}: {count}\n")
        f.write("\n")
        
        # Write a section for each speaker
        for speaker in all_speakers:
            f.write(f"SPEAKER: {speaker}\n")
            f.write("-" * 40 + "\n")
            
            # Show classification
            speaker_data = detailed_results[speaker]
            f.write(f"Classification: {speaker_data['classification']}\n")
            f.write(f"Age estimate: {speaker_data['age_estimate']:.1f} years\n")
            f.write(f"Confidence: {speaker_data['confidence']:.2f}\n\n")
            
            f.write("Gender Probabilities:\n")
            for gender, prob in speaker_data['gender_probabilities'].items():
                f.write(f"  {gender.capitalize()}: {prob:.4f}\n")
            
            f.write("\n\n")
        
    print(f"Detailed classification report saved to {report_path}")
    return report_path

def visualize_diarization_comparison(pyannote_segments, resemblyzer_segments, audio_duration):
    """
    Create a visualization comparing the two diarization methods.
    
    Args:
        pyannote_segments: List of (start, end, speaker_id) tuples from pyannote
        resemblyzer_segments: List of (start, end, speaker_id) tuples from resemblyzer
        audio_duration: Duration of the audio in seconds
    """
    # Set up plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    # Plot Pyannote diarization
    for start, end, speaker_id in pyannote_segments:
        # Extract speaker number from ID
        if "_" in speaker_id:
            speaker_num = int(speaker_id.split("_")[1])
            color = plt.cm.tab10(speaker_num % 10)
        else:
            speaker_num = 0
            color = 'gray'
        
        ax1.barh(y=speaker_id, width=end-start, left=start, height=0.8, color=color, alpha=0.7)
    
    ax1.set_title("Pyannote Diarization")
    ax1.set_ylabel("Speaker ID")
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Plot Resemblyzer diarization
    for start, end, speaker_id in resemblyzer_segments:
        # Extract speaker number from ID
        if "_" in speaker_id:
            speaker_num = int(speaker_id.split("_")[1])
            color = plt.cm.tab10(speaker_num % 10)
        else:
            speaker_num = 0
            color = 'gray'
        
        ax2.barh(y=speaker_id, width=end-start, left=start, height=0.8, color=color, alpha=0.7)
    
    ax2.set_title("Resemblyzer Diarization")
    ax2.set_ylabel("Speaker ID")
    ax2.set_xlabel("Time (seconds)")
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # Set x-axis limits
    ax1.set_xlim(0, audio_duration)
    
    plt.tight_layout()
    plt.savefig("diarization_comparison.png", dpi=150)
    print("Diarization comparison visualization saved to diarization_comparison.png")

def compare_diarization_methods(
    audio_path, 
    output_prefix="transcript", 
    include_timestamps=True,
    model_size="small",
    language=None,
    expected_child_count=None,
    num_speakers=None
):
    """
    Compare Pyannote and Resemblyzer diarization methods on the same audio file.
    
    Args:
        audio_path: Path to the audio file
        output_prefix: Prefix for output transcripts and reports
        include_timestamps: Whether to include timestamps in the transcripts
        model_size: Size of the Whisper model to use
        language: Language code for transcription
        expected_child_count: Expected number of children in the recording
        num_speakers: Optional number of speakers to cluster in Resemblyzer
    """
    total_start_time = time.time()
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Check if input is MP3 and convert if needed
    if audio_path.lower().endswith('.mp3'):
        print("MP3 file detected, converting to WAV for processing...")
        wav_path = convert_mp3_to_wav(audio_path)
        if wav_path is None:
            print("Error converting MP3 to WAV. Aborting.")
            return None
        audio_path = wav_path
    
    # Get audio duration for visualization
    with contextlib.closing(wave.open(audio_path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        audio_duration = frames / float(rate)
    
    # 1. Load the Audeering age-gender model
    processor, model = load_audeering_model(device)
    
    # 2. Run transcription with Whisper (used by both methods)
    whisper_segments = transcribe_with_whisper(audio_path, model_size, language)
    if not whisper_segments:
        print("Transcription failed. Cannot continue.")
        return None
    
    # 3. Run diarization with Pyannote
    pyannote_diarization = diarize_with_pyannote(audio_path)
    if pyannote_diarization is None:
        print("Pyannote diarization failed.")
        pyannote_results = None
    else:
        # Extract speaker segments for Pyannote
        pyannote_speaker_audio = extract_speaker_segments(
            audio_path, pyannote_diarization, output_dir="pyannote_speakers"
        )
        
        # Classify speakers using the Audeering model
        pyannote_classifications, pyannote_confidence, pyannate_details = classify_speakers_with_audeering(
            pyannote_speaker_audio, processor, model, device
        )
        
        # Extract text by speaker and create transcript
        pyannote_speaker_text, pyannote_segment_mapping = extract_text_by_speaker(
            whisper_segments, pyannote_diarization
        )
        
        # Create transcript
        pyannote_transcript = create_transcript(
            pyannote_segment_mapping, pyannote_classifications, 
            pyannote_confidence, include_timestamps
        )
        
        # Save transcript
        pyannote_output = f"{output_prefix}_pyannote.txt"
        with open(pyannote_output, "w", encoding="utf-8") as f:
            f.write("\n".join(pyannote_transcript))
        print(f"Pyannote transcript saved to {pyannote_output}")
        
        # Create visualization and report
        visualize_speaker_classification(pyannate_details, "Pyannote")
        create_detailed_report(pyannote_output, "Pyannate", 
                              pyannote_classifications, pyannote_confidence, 
                              pyannate_details)
        
        # Store results for comparison
        pyannote_results = {
            'diarization': pyannote_diarization,
            'speaker_audio': pyannote_speaker_audio,
            'classifications': pyannote_classifications,
            'confidence': pyannote_confidence,
            'details': pyannate_details,
            'transcript': pyannote_transcript,
            'output_file': pyannote_output
        }
    
    # 4. Run diarization with Resemblyzer
    resemblyzer_diarization = diarize_with_resemblyzer(
        audio_path, whisper_segments, num_speakers
    )
    if resemblyzer_diarization is None:
        print("Resemblyzer diarization failed.")
        resemblyzer_results = None
    else:
        # Extract speaker segments for Resemblyzer
        resemblyzer_speaker_audio = extract_speaker_segments(
            audio_path, resemblyzer_diarization, output_dir="resemblyzer_speakers"
        )
        
        # Classify speakers using the Audeering model
        resemblyzer_classifications, resemblyzer_confidence, resemblyzer_details = classify_speakers_with_audeering(
            resemblyzer_speaker_audio, processor, model, device
        )
        
        # Extract text by speaker and create transcript
        resemblyzer_speaker_text, resemblyzer_segment_mapping = extract_text_by_speaker(
            whisper_segments, resemblyzer_diarization
        )
        
        # Create transcript
        resemblyzer_transcript = create_transcript(
            resemblyzer_segment_mapping, resemblyzer_classifications, 
            resemblyzer_confidence, include_timestamps
        )
        
        # Save transcript
        resemblyzer_output = f"{output_prefix}_resemblyzer.txt"
        with open(resemblyzer_output, "w", encoding="utf-8") as f:
            f.write("\n".join(resemblyzer_transcript))
        print(f"Resemblyzer transcript saved to {resemblyzer_output}")
        
        # Create visualization and report
        visualize_speaker_classification(resemblyzer_details, "Resemblyzer")
        create_detailed_report(resemblyzer_output, "Resemblyzer", 
                              resemblyzer_classifications, resemblyzer_confidence, 
                              resemblyzer_details)
        
        # Store results for comparison
        resemblyzer_results = {
            'diarization': resemblyzer_diarization,
            'speaker_audio': resemblyzer_speaker_audio,
            'classifications': resemblyzer_classifications,
            'confidence': resemblyzer_confidence,
            'details': resemblyzer_details,
            'transcript': resemblyzer_transcript,
            'output_file': resemblyzer_output
        }
    
    # 5. Create comparison visualization if both methods worked
    if pyannote_results and resemblyzer_results:
        # Extract segments for visualization
        pyannote_segments = []
        for segment, _, speaker_id in pyannote_diarization.itertracks(yield_label=True):
            pyannote_segments.append((segment.start, segment.end, speaker_id))
        
        resemblyzer_segments = []
        for segment, _, speaker_id in resemblyzer_diarization.itertracks(yield_label=True):
            resemblyzer_segments.append((segment.start, segment.end, speaker_id))
        
        # Create comparison visualization
        visualize_diarization_comparison(
            pyannote_segments, resemblyzer_segments, audio_duration
        )
        
        # Write comparison summary
        with open(f"{output_prefix}_comparison.txt", "w", encoding="utf-8") as f:
            f.write("DIARIZATION METHOD COMPARISON\n")
            f.write("============================\n\n")
            
            # Speaker counts
            f.write("Speaker Counts:\n")
            if pyannote_results:
                f.write(f"  Pyannote: {len(pyannote_results['classifications'])} speakers\n")
            if resemblyzer_results:
                f.write(f"  Resemblyzer: {len(resemblyzer_results['classifications'])} speakers\n")
            f.write("\n")
            
            # Speaker types
            f.write("Speaker Types:\n")
            if pyannote_results:
                pyannote_counts = Counter(pyannote_results['classifications'].values())
                f.write("  Pyannote:\n")
                for speaker_type in ["Man", "Woman", "Child", "Unknown"]:
                    count = pyannote_counts.get(speaker_type, 0)
                    f.write(f"    {speaker_type}: {count}\n")
            
            if resemblyzer_results:
                resemblyzer_counts = Counter(resemblyzer_results['classifications'].values())
                f.write("  Resemblyzer:\n")
                for speaker_type in ["Man", "Woman", "Child", "Unknown"]:
                    count = resemblyzer_counts.get(speaker_type, 0)
                    f.write(f"    {speaker_type}: {count}\n")
            
            f.write("\nRecommendation:\n")
            f.write("Use the transcript that provides better speaker separation based on your knowledge of the audio.\n")
            f.write("Check the diarization_comparison.png file to see the differences between methods.\n")
            
        print(f"Comparison summary saved to {output_prefix}_comparison.txt")
    
    # Print summary
    total_time = time.time() - total_start_time
    print(f"\nProcessing completed in {total_time:.2f} seconds")
    
    # Print preview of both transcripts
    if pyannote_results:
        print(f"\n{BLUE}Pyannote Transcript Preview:{RESET_COLOR}")
        print(BLUE + "\n".join(pyannote_results['transcript'][:5]) + RESET_COLOR)
    
    if resemblyzer_results:
        print(f"\n{RED}Resemblyzer Transcript Preview:{RESET_COLOR}")
        print(RED + "\n".join(resemblyzer_results['transcript'][:5]) + RESET_COLOR)
    
    print("\nReview both transcripts and their visualizations to determine which diarization method works better for your audio.")
    
    return {
        'pyannote': pyannote_results,
        'resemblyzer': resemblyzer_results
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Pyannote and Resemblyzer diarization methods")
    parser.add_argument("audio_file", help="Path to the audio file to process (MP3 or WAV)")
    parser.add_argument("-o", "--output", default="transcript", help="Prefix for output files")
    parser.add_argument("--no-timestamps", action="store_true", help="Exclude timestamps")
    parser.add_argument("--model", default="small", choices=["tiny", "base", "small", "medium"], 
                        help="Whisper model size (smaller is faster)")
    parser.add_argument("--language", help="Specify language code (e.g., 'en' for English)")
    parser.add_argument("--api-key", help="Hugging Face API key")
    parser.add_argument("--child-count", type=int, help="Expected number of children in the recording")
    parser.add_argument("--num-speakers", type=int, help="Number of speakers (if known) for Resemblyzer")
    
    args = parser.parse_args()
    
    # Set API key if provided
    if args.api_key:
        HF_API_KEY = args.api_key
    
    compare_diarization_methods(
        args.audio_file, 
        args.output, 
        include_timestamps=not args.no_timestamps,
        model_size=args.model,
        language=args.language,
        expected_child_count=args.child_count,
        num_speakers=args.num_speakers
    )
