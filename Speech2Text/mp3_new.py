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

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

# Color codes for printing
NEON_GREEN = "\033[92m"
RESET_COLOR = "\033[0m"

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
    Returns:
        - Embeddings (optional)
        - Age prediction (0-1 scale representing 0-100 years)
        - Gender probabilities (female, male, child)
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

def diarize_audio(audio_path):
    """
    Runs speaker diarization on the audio file.
    """
    start_time = time.time()
    print("Running diarization...")
    
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
        print(f"Diarization completed in {time.time() - start_time:.2f} seconds")
        return result
    
    except Exception as e:
        print(f"Error in diarization: {e}")
        traceback.print_exc()
        return None

def transcribe_audio(audio_path, model_size="small", language=None):
    """
    Transcribes the audio using faster-whisper with speed optimizations.
    """
    start_time = time.time()
    print("Transcribing audio...")
    
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
        print(f"Transcription completed in {time.time() - start_time:.2f} seconds")
        return result
    
    except Exception as e:
        print(f"Error in transcription: {e}")
        traceback.print_exc()
        return []

def extract_speaker_segments(audio_path, diarization, min_duration=1.0):
    """
    Extracts audio segments for each speaker.
    """
    start_time = time.time()
    print("Extracting speaker segments...")
    
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
            os.makedirs("speaker_samples", exist_ok=True)
            
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
                output_path = f"speaker_samples/speaker_{speaker_id}_segment_{i+1}.wav"
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
    
    for segment in transcription:
        # Find which speaker was talking during this segment
        t_start, t_end = segment['start'], segment['end']
        trans_segment = Segment(t_start, t_end)
        
        # Find overlapping speaker
        overlapping_speakers = diarization.crop(trans_segment, mode="intersection")
        if not overlapping_speakers:
            continue
            
        # Get most overlapping speaker
        speaker_id = None
        max_overlap = 0
        for speech_turn, _, spk_id in overlapping_speakers.itertracks(yield_label=True):
            if speech_turn.duration > max_overlap:
                max_overlap = speech_turn.duration
                speaker_id = spk_id
        
        if speaker_id:
            if speaker_id not in speaker_text:
                speaker_text[speaker_id] = []
            speaker_text[speaker_id].append(segment['text'])
    
    # Combine all text for each speaker
    for speaker_id in speaker_text:
        speaker_text[speaker_id] = " ".join(speaker_text[speaker_id])
    
    return speaker_text

def format_timestamp(seconds):
    """
    Formats seconds as HH:MM:SS
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def align_speakers_to_transcription(diarization, transcription_segments, speaker_classifications, confidence_scores, include_timestamps=True):
    """
    Matches each transcription segment to the speaker from the diarization.
    """
    start_time = time.time()
    labeled_segments = []
    
    for tseg in transcription_segments:
        t_start = tseg["start"]
        t_end = tseg["end"]
        
        # We'll represent the transcription segment as a pyannote Segment
        trans_segment = Segment(t_start, t_end)
        
        # Find overlapping diarization segments
        overlapping_speakers = diarization.crop(
            trans_segment, mode="intersection"
        )
        
        if len(overlapping_speakers) == 0:
            # No speaker found, check for any close speaker within 0.5s
            extended_segment = Segment(max(0, t_start - 0.5), t_end + 0.5)
            nearby_speakers = diarization.crop(extended_segment, mode="intersection")
            
            if len(nearby_speakers) > 0:
                # Find closest speaker
                speaker_id = None
                for speech_turn, _, spk_id in nearby_speakers.itertracks(yield_label=True):
                    speaker_id = spk_id
                    break
                    
                if speaker_id is not None and speaker_id in speaker_classifications:
                    speaker_label = speaker_classifications[speaker_id]
                    confidence = confidence_scores.get(speaker_id, {}).get(speaker_label, 0.5)
                else:
                    speaker_label = "Unknown"
                    confidence = 0.0
            else:
                speaker_label = "Unknown"
                confidence = 0.0
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
            if speaker_id is not None and speaker_id in speaker_classifications:
                speaker_label = speaker_classifications[speaker_id]
                confidence = confidence_scores.get(speaker_id, {}).get(speaker_label, 0.5)
            else:
                speaker_label = "Unknown"
                confidence = 0.0
        
        # Format timestamps as HH:MM:SS
        if include_timestamps:
            start_formatted = format_timestamp(t_start)
            end_formatted = format_timestamp(t_end)
            timestamp_str = f"[{start_formatted} - {end_formatted}] "
        else:
            timestamp_str = ""
        
        # Build the final text line with confidence score
        confidence_indicator = f"({confidence:.2f})" if confidence > 0 else ""
        line = f"{timestamp_str}[{speaker_label}{confidence_indicator}] {tseg['text']}"
        labeled_segments.append(line)
    
    print(f"Alignment completed in {time.time() - start_time:.2f} seconds")
    return labeled_segments

def visualize_speaker_classification(detailed_results):
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
    ax1.set_title("Speaker Classification Probabilities")
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
    ax2.set_title("Estimated Speaker Ages")
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
    plt.savefig("speaker_classification_results.png", dpi=150)
    print("Speaker classification visualization saved to speaker_classification_results.png")

def create_detailed_report(output_txt, speaker_classifications, confidence_scores, detailed_results):
    """
    Create a detailed report of speaker classifications.
    """
    report_path = output_txt.replace(".txt", "_detailed_report.txt")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("SPEAKER CLASSIFICATION DETAILED REPORT\n")
        f.write("====================================\n\n")
        
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

def process_mp3_with_audeering(
    audio_path, 
    output_txt="transcript.txt", 
    include_timestamps=True,
    model_size="small",
    language=None,
    expected_child_count=None
):
    """
    Process audio file using the Audeering age-gender model for speaker classification.
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
    
    # 1. Load the Audeering age-gender model
    processor, model = load_audeering_model(device)
    
    # 2. Run diarization to get speaker segments
    diarization = diarize_audio(audio_path)
    if diarization is None:
        print("Diarization failed. Cannot continue.")
        return None
    
    # 3. Transcribe audio
    transcription = transcribe_audio(audio_path, model_size, language)
    if not transcription:
        print("Transcription failed. Cannot continue.")
        return None
    
    # 4. Extract speaker segments
    speaker_audio = extract_speaker_segments(audio_path, diarization)
    if not speaker_audio:
        print("Speaker extraction failed. Cannot continue.")
        return None
    
    # 5. Classify speakers using the Audeering model
    speaker_classifications, confidence_scores, detailed_results = classify_speakers_with_audeering(
        speaker_audio, processor, model, device
    )
    
    # 6. Apply expected child count if provided
    if expected_child_count is not None and expected_child_count > 0:
        print(f"\nAdjusting for expected child count of {expected_child_count}...")
        
        # Count current children
        current_child_count = sum(1 for label in speaker_classifications.values() if label == "Child")
        
        if current_child_count < expected_child_count:
            # Find speakers to reclassify as children
            candidates = []
            
            for speaker_id, label in speaker_classifications.items():
                if label != "Child":
                    # Use child probability as the score
                    child_score = confidence_scores[speaker_id]["Child"]
                    candidates.append((speaker_id, child_score))
            
            # Sort by child-likeness score (highest first)
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Reclassify as many as needed
            reclassified = 0
            for i in range(min(len(candidates), expected_child_count - current_child_count)):
                speaker_id = candidates[i][0]
                speaker_classifications[speaker_id] = "Child"
                
                # Update confidence scores and detailed results
                confidence_scores[speaker_id]["Child"] = 0.7
                confidence_scores[speaker_id]["Man"] = 0.15
                confidence_scores[speaker_id]["Woman"] = 0.15
                
                detailed_results[speaker_id]["classification"] = "Child"
                detailed_results[speaker_id]["confidence"] = 0.7
                detailed_results[speaker_id]["gender_probabilities"]["child"] = 0.7
                detailed_results[speaker_id]["gender_probabilities"]["male"] = 0.15
                detailed_results[speaker_id]["gender_probabilities"]["female"] = 0.15
                
                reclassified += 1
            
            if reclassified > 0:
                print(f"Reclassified {reclassified} speakers as Child based on expected count")
    
    # 7. Create visualizations and reports
    visualize_speaker_classification(detailed_results)
    report_path = create_detailed_report(output_txt, speaker_classifications, 
                                        confidence_scores, detailed_results)
    
    # 8. Align transcription with speaker labels
    labeled_transcript = align_speakers_to_transcription(
        diarization, transcription, speaker_classifications, confidence_scores, include_timestamps
    )
    
    # 9. Save the transcript
    transcript_text = "\n".join(labeled_transcript)
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(transcript_text)
    
    total_time = time.time() - total_start_time
    print(f"\nProcessing completed in {total_time:.2f} seconds")
    print(f"Transcript saved to {output_txt}")
    print(f"Detailed report saved to {report_path}")
    
    print("\nTranscript Preview:")
    print(NEON_GREEN + "\n".join(labeled_transcript[:10]) + RESET_COLOR)
    
    return {
        'transcript': transcript_text,
        'speaker_classifications': speaker_classifications,
        'confidence_scores': confidence_scores,
        'detailed_results': detailed_results,
        'report_path': report_path
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MP3 transcript generator with Audeering age-gender model")
    parser.add_argument("audio_file", help="Path to the audio file to process (MP3 or WAV)")
    parser.add_argument("-o", "--output", default="transcript.txt", help="Output file for the transcript")
    parser.add_argument("--no-timestamps", action="store_true", help="Exclude timestamps")
    parser.add_argument("--model", default="small", choices=["tiny", "base", "small", "medium"], 
                        help="Whisper model size (smaller is faster)")
    parser.add_argument("--language", help="Specify language code (e.g., 'en' for English)")
    parser.add_argument("--api-key", help="Hugging Face API key")
    parser.add_argument("--child-count", type=int, help="Expected number of children in the recording")
    
    args = parser.parse_args()
    
    # Set API key if provided
    if args.api_key:
        HF_API_KEY = args.api_key
    
    process_mp3_with_audeering(
        args.audio_file, 
        args.output, 
        include_timestamps=not args.no_timestamps,
        model_size=args.model,
        language=args.language,
        expected_child_count=args.child_count
    )
