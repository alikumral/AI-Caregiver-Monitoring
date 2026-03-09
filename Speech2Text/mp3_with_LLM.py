import os
import torch
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import time
import concurrent.futures
import requests
import json
from collections import Counter

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment

# Color codes for printing
NEON_GREEN = "\033[92m"
RESET_COLOR = "\033[0m"

# Hugging Face API configuration
HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
HF_API_KEY = os.getenv("HF_API_KEY", "YOUR_HF_API_KEY")  # Replace with your key
HF_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

def diarize_audio(pipeline, audio_path):
    """
    Runs speaker diarization on the audio file.
    """
    start_time = time.time()
    print("Running diarization...")
    result = pipeline(audio_path)
    print(f"Diarization completed in {time.time() - start_time:.2f} seconds")
    return result

def transcribe_audio(whisper_model, audio_path, language=None):
    """
    Transcribes the audio using faster-whisper with speed optimizations.
    """
    start_time = time.time()
    print("Transcribing audio...")
    segments, info = whisper_model.transcribe(
        audio_path, 
        beam_size=5,
        language=language,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    
    result = []
    for seg in segments:
        result.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip()
        })
    print(f"Transcription completed in {time.time() - start_time:.2f} seconds")
    return result

def extract_acoustic_features(audio_path, diarization, min_duration=1.0):
    """
    Extracts acoustic features for each speaker.
    """
    start_time = time.time()
    print("Extracting acoustic features...")
    # Load the full audio
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    # Dictionary to store speaker features
    speaker_features = {}
    speaker_segments = {}
    
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
            
        # Extract features for each valid segment
        f0_values = []
        spectral_centroids = []
        spectral_rolloffs = []
        
        # Create a directory for saving speaker samples
        os.makedirs("speaker_samples", exist_ok=True)
        
        for i, segment in enumerate(valid_segments[:3]):  # Limit to 3 samples per speaker
            start_sample = int(segment.start * sr)
            end_sample = int(segment.end * sr)
            
            if start_sample >= len(audio) or end_sample > len(audio):
                continue
                
            segment_audio = audio[start_sample:end_sample]
            
            # Save the segment for manual inspection
            output_path = f"speaker_samples/speaker_{speaker_id}_segment_{i+1}.wav"
            sf.write(output_path, segment_audio, sr)
            
            # Fundamental frequency using YIN algorithm
            try:
                f0, voiced_flag, _ = librosa.pyin(segment_audio, 
                                                 fmin=60, 
                                                 fmax=500,
                                                 sr=sr)
                valid_f0 = f0[~np.isnan(f0)]
                if len(valid_f0) > 0:
                    f0_values.append(np.median(valid_f0))
            except Exception as e:
                print(f"Error extracting F0 for speaker {speaker_id}, segment {i}: {e}")
            
            # Spectral features
            try:
                centroid = np.mean(librosa.feature.spectral_centroid(y=segment_audio, sr=sr))
                spectral_centroids.append(centroid)
                
                rolloff = np.mean(librosa.feature.spectral_rolloff(y=segment_audio, sr=sr))
                spectral_rolloffs.append(rolloff)
            except Exception as e:
                print(f"Error extracting spectral features for speaker {speaker_id}, segment {i}: {e}")
        
        # Compute average features for this speaker
        features = {}
        if f0_values:
            features['f0_median'] = np.median(f0_values)
        if spectral_centroids:
            features['spectral_centroid'] = np.mean(spectral_centroids)
        if spectral_rolloffs:
            features['spectral_rolloff'] = np.mean(spectral_rolloffs)
        
        if features:
            speaker_features[speaker_id] = features
            print(f"Speaker {speaker_id} features extracted successfully")
    
    print(f"Feature extraction completed in {time.time() - start_time:.2f} seconds")
    return speaker_features

def classify_speakers_acoustic(speaker_features):
    """
    Classifies speakers based on acoustic features with confidence scores.
    """
    if not speaker_features:
        return {}, {}
        
    # Print all features for debugging
    print("\nAcoustic Features Summary:")
    for speaker_id, features in speaker_features.items():
        feature_str = ", ".join([f"{k}: {v:.2f}" if isinstance(v, (float, int)) else f"{k}: [...]" 
                               for k, v in features.items()])
        print(f"Speaker {speaker_id}: {feature_str}")
    
    # Classifications and confidence scores
    classifications = {}
    confidence_scores = {}
    
    for speaker_id, features in speaker_features.items():
        if 'f0_median' in features:
            f0 = features['f0_median']
            centroid = features.get('spectral_centroid', 0)
            
            # Calculate scores for each class
            man_score = 0.0
            woman_score = 0.0
            child_score = 0.0
            
            # Man classification logic
            if f0 < 145:  # Strong indicator
                man_score = 0.95
            elif f0 < 165:  # Likely
                man_score = 0.85 - (f0 - 145) * 0.01
            else:  # Unlikely
                man_score = max(0.05, 0.30 - (f0 - 165) * 0.01)
            
            # Child classification logic
            if f0 > 280 and centroid > 3500:  # Strong indicator
                child_score = 0.95
            elif f0 > 245:  # Likely
                child_score = 0.70 + (f0 - 245) * 0.01
            else:  # Unlikely
                child_score = max(0.05, 0.30 - (245 - f0) * 0.01)
            
            # Woman classification logic
            if 165 < f0 < 245:  # Ideal range
                woman_score = 0.90 - abs(205 - f0) * 0.004
            else:  # Less likely
                woman_score = max(0.05, 0.40 - min(abs(165 - f0), abs(245 - f0)) * 0.01)
            
            # Normalize scores
            total = man_score + woman_score + child_score
            man_score /= total
            woman_score /= total
            child_score /= total
            
            # Get highest scoring class
            scores = {
                "Man": man_score,
                "Woman": woman_score,
                "Child": child_score
            }
            best_class = max(scores.items(), key=lambda x: x[1])
            
            classifications[speaker_id] = best_class[0]
            confidence_scores[speaker_id] = {
                "Man": man_score,
                "Woman": woman_score,
                "Child": child_score
            }
        else:
            # If no pitch data, use spectral centroid as fallback
            centroid = features.get('spectral_centroid', 0)
            if centroid < 2000:
                classifications[speaker_id] = "Man"
                confidence_scores[speaker_id] = {"Man": 0.7, "Woman": 0.2, "Child": 0.1}
            elif centroid > 3000:
                classifications[speaker_id] = "Child"
                confidence_scores[speaker_id] = {"Man": 0.1, "Woman": 0.2, "Child": 0.7}
            else:
                classifications[speaker_id] = "Woman"
                confidence_scores[speaker_id] = {"Man": 0.2, "Woman": 0.7, "Child": 0.1}
    
    # Print acoustic classifications with confidence
    print("\nAcoustic Classification Results:")
    for speaker, label in classifications.items():
        confidence = confidence_scores[speaker][label]
        print(f"Speaker {speaker} â†’ {label} (confidence: {confidence:.2f})")
    
    return classifications, confidence_scores

def extract_text_by_speaker(transcription, diarization):
    """
    Extract all text spoken by each speaker for LLM analysis.
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

def classify_with_huggingface(speaker_texts):
    """
    Use Hugging Face Inference API to classify speakers based on their text.
    Returns classifications and confidence scores.
    """
    print("\nClassifying speakers with Hugging Face API...")
    classifications = {}
    confidence_scores = {}
    
    for speaker_id, text in speaker_texts.items():
        # Ensure we have enough text to analyze (at least 20 words)
        if len(text.split()) < 20:
            print(f"Not enough text for speaker {speaker_id} to use LLM classification")
            continue
        
        # Prepare API request for zero-shot classification
        try:
            payload = {
                "inputs": text[:1024],  # Limit text length
                "parameters": {
                    "candidate_labels": ["man", "woman", "child"]
                }
            }
            
            print(f"Sending API request for speaker {speaker_id}...")
            response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload)
            
            if response.status_code != 200:
                print(f"API error: {response.status_code} - {response.text}")
                continue
                
            result = response.json()
            
            # Get classifications with scores
            if "labels" in result and "scores" in result:
                # Extract labels and scores
                labels = [label.capitalize() for label in result["labels"]]
                scores = result["scores"]
                
                # Get highest scoring label
                best_idx = scores.index(max(scores))
                best_label = labels[best_idx]
                
                classifications[speaker_id] = best_label
                confidence_scores[speaker_id] = {
                    labels[i]: scores[i] for i in range(len(labels))
                }
                
                print(f"LLM classified speaker {speaker_id} as {best_label} with confidence {scores[best_idx]:.2f}")
            else:
                print(f"Unexpected API response format: {result}")
                
        except Exception as e:
            print(f"Error using Hugging Face API for speaker {speaker_id}: {e}")
    
    return classifications, confidence_scores

def combine_classifications(acoustic_classes, acoustic_confidence, llm_classes, llm_confidence):
    """
    Combine acoustic and LLM classifications, with LLM taking precedence in case of disagreement
    unless acoustic confidence is very high.
    """
    final_classifications = {}
    final_confidence = {}
    
    # Process each speaker
    for speaker_id in acoustic_classes.keys():
        # Get acoustic classification and confidence
        acoustic_label = acoustic_classes[speaker_id]
        acoustic_conf = acoustic_confidence[speaker_id][acoustic_label]
        
        # If no LLM classification available, use acoustic
        if speaker_id not in llm_classes:
            final_classifications[speaker_id] = acoustic_label
            final_confidence[speaker_id] = acoustic_conf
            print(f"Speaker {speaker_id}: Using acoustic classification only: {acoustic_label} ({acoustic_conf:.2f})")
            continue
        
        # Get LLM classification and confidence
        llm_label = llm_classes[speaker_id]
        llm_conf = llm_confidence[speaker_id][llm_label]
        
        # If they agree, we're very confident
        if acoustic_label == llm_label:
            # Average the confidence scores for a boost
            combined_conf = (acoustic_conf + llm_conf) / 2
            final_classifications[speaker_id] = acoustic_label
            final_confidence[speaker_id] = min(0.99, combined_conf * 1.1)  # Boost confidence but cap at 0.99
            print(f"Speaker {speaker_id}: Acoustic and LLM agree: {acoustic_label} ({final_confidence[speaker_id]:.2f})")
        
        # If they disagree, but acoustic is very confident (>0.9), still use acoustic
        elif acoustic_conf > 0.9:
            final_classifications[speaker_id] = acoustic_label
            final_confidence[speaker_id] = acoustic_conf
            print(f"Speaker {speaker_id}: High acoustic confidence overriding LLM: {acoustic_label} ({acoustic_conf:.2f})")
        
        # In all other cases of disagreement, use LLM (as requested)
        else:
            final_classifications[speaker_id] = llm_label
            final_confidence[speaker_id] = llm_conf
            print(f"Speaker {speaker_id}: Using LLM over acoustic: {llm_label} ({llm_conf:.2f}) vs {acoustic_label} ({acoustic_conf:.2f})")
    
    return final_classifications, final_confidence

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
                    confidence = confidence_scores.get(speaker_id, 0.5)
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
                confidence = confidence_scores.get(speaker_id, 0.5)
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

def visualize_classifications(acoustic_classes, acoustic_confidence, llm_classes, llm_confidence, final_classes):
    """
    Visualize the classification process and confidence levels
    """
    # Get all speaker IDs
    all_speakers = set(acoustic_classes.keys())
    all_speakers.update(llm_classes.keys())
    all_speakers = sorted(list(all_speakers))
    
    if not all_speakers:
        print("No speakers to visualize")
        return
    
    # Set up the plot
    plt.figure(figsize=(14, 8))
    
    # Set up bar positions
    bar_width = 0.25
    index = np.arange(len(all_speakers))
    
    # Track which classes were used for each speaker
    chosen_classes = []
    
    # Plot bars for each speaker
    for i, speaker in enumerate(all_speakers):
        acoustic_label = acoustic_classes.get(speaker, "Unknown")
        llm_label = llm_classes.get(speaker, "Unknown")
        final_label = final_classes.get(speaker, "Unknown")
        
        # Get confidence scores
        acoustic_conf = acoustic_confidence.get(speaker, {}).get(acoustic_label, 0)
        llm_conf = llm_confidence.get(speaker, {}).get(llm_label, 0)
        
        # Plot the bars
        plt.bar(index[i] - bar_width, acoustic_conf, bar_width, 
                color='blue', alpha=0.7, label='Acoustic' if i == 0 else "")
        plt.bar(index[i], llm_conf, bar_width, 
                color='green', alpha=0.7, label='LLM' if i == 0 else "")
        
        # Mark which one was chosen
        marker_x = index[i] - bar_width if final_label == acoustic_label else index[i]
        plt.plot(marker_x, 1.05, 'r*', markersize=10)
        
        # Store the chosen classification
        chosen_classes.append(final_label)
    
    # Set up the axes
    plt.xlabel('Speaker ID')
    plt.ylabel('Confidence Score')
    plt.title('Classification Confidence by Method')
    plt.xticks(index, all_speakers)
    plt.ylim(0, 1.1)
    
    # Add a second axis for the classification labels
    ax2 = plt.twinx()
    ax2.set_ylim(0, 1.1)
    ax2.set_yticks([])
    
    # Add text labels for the final classifications
    for i, cls in enumerate(chosen_classes):
        plt.text(index[i], 1.1, cls, ha='center', fontweight='bold')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig("classification_confidence.png")
    print("Classification visualization saved to classification_confidence.png")

def process_with_llm(
    audio_path, 
    output_txt="transcript.txt", 
    force_child=False,
    include_timestamps=True,
    model_size="small",
    language=None
):
    """
    Process audio with combined acoustic and LLM speaker classification.
    """
    total_start_time = time.time()
    
    # 1. Initialize models
    print(f"Loading models (size: {model_size})...")
    model_start = time.time()
    
    # Use compute_type=int8 for faster transcription if available
    compute_type = "int8" if torch.cuda.is_available() else "float32"
    
    whisper_model = WhisperModel(
        model_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
        compute_type=compute_type
    )
    
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.getenv("HF_API_KEY", "YOUR_HF_API_KEY")  # Replace with your token
    )
    if torch.cuda.is_available():
        diarization_pipeline.to(torch.device("cuda"))
    
    print(f"Models loaded in {time.time() - model_start:.2f} seconds")
    
    # 2. Run diarization and transcription
    print("Processing audio...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        diarization_future = executor.submit(diarize_audio, diarization_pipeline, audio_path)
        transcription_future = executor.submit(transcribe_audio, whisper_model, audio_path, language)
        
        diarization = diarization_future.result()
        transcription = transcription_future.result()
    
    # 3. Extract features and perform classifications
    
    # 3a. Acoustic classification
    acoustic_features = extract_acoustic_features(audio_path, diarization)
    acoustic_classes, acoustic_confidence = classify_speakers_acoustic(acoustic_features)
    
    # 3b. LLM-based classification
    speaker_texts = extract_text_by_speaker(transcription, diarization)
    llm_classes, llm_confidence = classify_with_huggingface(speaker_texts)
    
    # 3c. Combine classifications with LLM preference
    final_classes, final_confidence = combine_classifications(
        acoustic_classes, acoustic_confidence, 
        llm_classes, llm_confidence
    )
    
    # 4. Visualize the classification process
    visualize_classifications(
        acoustic_classes, acoustic_confidence,
        llm_classes, llm_confidence,
        final_classes
    )
    
    # 5. Align transcription with final speaker labels
    labeled_transcript = align_speakers_to_transcription(
        diarization, transcription, final_classes, final_confidence, include_timestamps
    )
    
    # 6. Save the transcript
    transcript_text = "\n".join(labeled_transcript)
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(transcript_text)
    
    total_time = time.time() - total_start_time
    print(f"\nProcessing completed in {total_time:.2f} seconds")
    print(f"Transcript saved to {output_txt}")
    
    print("\nTranscript Preview:")
    print(NEON_GREEN + "\n".join(labeled_transcript[:10]) + RESET_COLOR)
    
    # Create a detailed report
    with open(output_txt.replace(".txt", "_report.txt"), "w", encoding="utf-8") as f:
        f.write("SPEAKER CLASSIFICATION REPORT\n")
        f.write("===========================\n\n")
        
        f.write("ACOUSTIC CLASSIFICATION:\n")
        for speaker, label in acoustic_classes.items():
            scores = acoustic_confidence[speaker]
            score_str = ", ".join([f"{k}: {v:.2f}" for k, v in scores.items()])
            f.write(f"Speaker {speaker}: {label} ({score_str})\n")
        
        f.write("\nLLM CLASSIFICATION:\n")
        for speaker, label in llm_classes.items():
            scores = llm_confidence[speaker]
            score_str = ", ".join([f"{k}: {v:.2f}" for k, v in scores.items()])
            f.write(f"Speaker {speaker}: {label} ({score_str})\n")
        
        f.write("\nFINAL CLASSIFICATION:\n")
        for speaker, label in final_classes.items():
            conf = final_confidence[speaker]
            f.write(f"Speaker {speaker}: {label} (confidence: {conf:.2f})\n")
    
    return {
        'transcript': transcript_text,
        'acoustic_classes': acoustic_classes,
        'acoustic_confidence': acoustic_confidence,
        'llm_classes': llm_classes,
        'llm_confidence': llm_confidence,
        'final_classes': final_classes,
        'final_confidence': final_confidence
    }

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM-enhanced speaker classification")
    parser.add_argument("audio_file", help="Path to the audio file to process")
    parser.add_argument("-o", "--output", default="transcript.txt", help="Output file for the transcript")
    parser.add_argument("--no-timestamps", action="store_true", help="Exclude timestamps")
    parser.add_argument("--model", default="small", choices=["tiny", "base", "small", "medium"], 
                        help="Whisper model size (smaller is faster)")
    parser.add_argument("--language", help="Specify language code (e.g., 'en' for English)")
    parser.add_argument("--api-key", help="Hugging Face API key")
    
    args = parser.parse_args()
    
    # Set API key if provided
    if args.api_key:
        HF_API_KEY = args.api_key
        HF_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}
    
    process_with_llm(
        args.audio_file, 
        args.output, 
        include_timestamps=not args.no_timestamps,
        model_size=args.model,
        language=args.language
    )
