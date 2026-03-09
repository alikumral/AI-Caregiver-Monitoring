import os
import torch
import numpy as np
import librosa
import soundfile as sf
import time
import concurrent.futures
import matplotlib.pyplot as plt
from collections import Counter
import json
import subprocess
import sys
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Optional imports - will be handled gracefully if not available
try:
    import xgboost as xgb
    HAVE_XGBOOST = True
except ImportError:
    HAVE_XGBOOST = False
    print("XGBoost not installed. Will use RandomForest instead.")

try:
    import torchaudio
    from speechbrain.pretrained import EncoderClassifier
    HAVE_SPEECHBRAIN = True
except ImportError:
    HAVE_SPEECHBRAIN = False
    print("SpeechBrain not installed. Neural embeddings will not be available.")

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment

# Global constants
AUDIO_SAMPLE_RATE = 16000
MIN_SEGMENT_DURATION = 1.0  # seconds
FEATURE_CACHE_DIR = "feature_cache"
MODEL_DIR = "models"
SPEAKER_SAMPLES_DIR = "speaker_samples"

# Color codes for printing
NEON_GREEN = "\033[92m"
RESET_COLOR = "\033[0m"

# Ensure required directories exist
os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SPEAKER_SAMPLES_DIR, exist_ok=True)

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
    Transcribes the audio using faster-whisper.
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

def extract_speaker_features(audio_path, diarization):
    """
    Extracts acoustic features for each speaker.
    """
    print("Extracting speaker features...")
    start_time = time.time()
    
    # Check if features are already cached
    cache_file = os.path.join(FEATURE_CACHE_DIR, os.path.basename(audio_path) + ".features.joblib")
    if os.path.exists(cache_file):
        print(f"Loading cached features from {cache_file}")
        return joblib.load(cache_file)
    
    # Load the full audio
    audio, sr = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE, mono=True)
    
    # Group segments by speaker
    speaker_segments = {}
    for segment, _, speaker_id in diarization.itertracks(yield_label=True):
        if segment.duration < MIN_SEGMENT_DURATION:
            continue
        if speaker_id not in speaker_segments:
            speaker_segments[speaker_id] = []
        speaker_segments[speaker_id].append(segment)
    
    # Process each speaker
    all_features = {}
    
    # Try to load SpeechBrain classifier if available
    have_classifier = False
    if HAVE_SPEECHBRAIN:
        try:
            classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb", 
                savedir="pretrained_models/spkrec-ecapa-voxceleb"
            )
            have_classifier = True
            print("Successfully loaded SpeechBrain classifier")
        except Exception as e:
            print(f"Could not load SpeechBrain classifier: {e}")
    
    for speaker_id, segments in speaker_segments.items():
        print(f"Processing speaker {speaker_id}...")
        
        # Sort segments by duration (longest first)
        segments.sort(key=lambda s: s.duration, reverse=True)
        
        # Take the top 5 longest segments
        valid_segments = [s for s in segments[:5] if s.duration >= MIN_SEGMENT_DURATION]
        
        if not valid_segments:
            continue
        
        # Extract features from each segment
        segment_features = []
        embeddings = []
        
        for i, segment in enumerate(valid_segments[:3]):
            start_idx = int(segment.start * sr)
            end_idx = min(len(audio), int(segment.end * sr))
            
            if start_idx >= end_idx:
                continue
                
            segment_audio = audio[start_idx:end_idx]
            
            # Save for manual inspection
            output_path = os.path.join(SPEAKER_SAMPLES_DIR, f"speaker_{speaker_id}_segment_{i+1}.wav")
            sf.write(output_path, segment_audio, sr)
            
            # Extract acoustic features
            features = {}
            
            # Pitch features
            try:
                f0, voiced_flag, _ = librosa.pyin(
                    segment_audio, 
                    fmin=60, 
                    fmax=600,
                    sr=sr
                )
                
                # Filter out NaN values
                valid_f0 = f0[~np.isnan(f0)]
                
                if len(valid_f0) > 0:
                    features['f0_median'] = np.median(valid_f0)
                    features['f0_mean'] = np.mean(valid_f0)
                    features['f0_std'] = np.std(valid_f0)
            except Exception as e:
                print(f"Error extracting pitch for speaker {speaker_id}: {e}")
            
            # Spectral features
            try:
                # Spectral centroid (center of mass of the spectrum)
                cent = librosa.feature.spectral_centroid(y=segment_audio, sr=sr)[0]
                features['spectral_centroid'] = np.mean(cent)
                
                # Spectral rolloff (frequency below which X% of energy is contained)
                rolloff = librosa.feature.spectral_rolloff(y=segment_audio, sr=sr)[0]
                features['spectral_rolloff'] = np.mean(rolloff)
                
                # Zero crossing rate (higher for more noisy signals)
                zcr = librosa.feature.zero_crossing_rate(segment_audio)[0]
                features['zero_crossing_rate'] = np.mean(zcr)
            except Exception as e:
                print(f"Error extracting spectral features for speaker {speaker_id}: {e}")
            
            # Harmonics features
            try:
                # Separate harmonic and percussive components
                y_harmonic, y_percussive = librosa.effects.hpss(segment_audio)
                
                # Calculate ratio of harmonic to percussive energy
                harmonic_energy = np.sum(y_harmonic**2)
                percussive_energy = np.sum(y_percussive**2)
                if percussive_energy > 0:
                    features['harmonic_ratio'] = harmonic_energy / percussive_energy
            except Exception as e:
                print(f"Error extracting harmonic features for speaker {speaker_id}: {e}")
            
            # Voice quality features
            try:
                # RMS energy
                rms = librosa.feature.rms(y=segment_audio)[0]
                features['rms_energy'] = np.mean(rms)
                
                # Estimate speech rate from energy peaks
                onset_env = librosa.onset.onset_strength(y=segment_audio, sr=sr)
                peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10)
                features['speech_rate'] = len(peaks) / segment.duration
            except Exception as e:
                print(f"Error extracting voice quality features for speaker {speaker_id}: {e}")
            
            # Neural embeddings if available
            if have_classifier and HAVE_SPEECHBRAIN:
                try:
                    # Convert to tensor
                    waveform = torch.FloatTensor(segment_audio).unsqueeze(0)
                    # Extract embedding using SpeechBrain
                    emb = classifier.encode_batch(waveform)
                    embedding = emb.squeeze().cpu().numpy()
                    embeddings.append(embedding)
                except Exception as e:
                    print(f"Error extracting embeddings for speaker {speaker_id}: {e}")
            
            # Add all features from this segment
            if features:
                segment_features.append(features)
        
        # Aggregate features across segments
        if segment_features:
            # Average the features across all segments
            avg_features = {}
            for feature_name in segment_features[0].keys():
                values = [features[feature_name] for features in segment_features if feature_name in features]
                if values:
                    avg_features[feature_name] = np.mean(values)
            
            # Add embeddings if available
            if embeddings:
                avg_embedding = np.mean(embeddings, axis=0)
                # Only keep a few components to avoid too many features
                for i in range(min(5, len(avg_embedding))):
                    avg_features[f'embedding_{i}'] = avg_embedding[i]
            
            # Store the averaged features
            all_features[speaker_id] = avg_features
    
    # Add relative features comparing speakers to each other
    add_relative_features(all_features)
    
    # Save to cache
    joblib.dump(all_features, cache_file)
    
    print(f"Feature extraction completed in {time.time() - start_time:.2f} seconds")
    return all_features

def add_relative_features(feature_dict):
    """
    Add features that compare speakers to each other.
    """
    if len(feature_dict) < 2:
        return
    
    # Calculate statistics across all speakers
    f0_values = [features.get('f0_median', 0) for features in feature_dict.values() 
                if 'f0_median' in features and features['f0_median'] > 0]
    
    if not f0_values:
        return
    
    global_min_f0 = min(f0_values)
    global_max_f0 = max(f0_values)
    global_median_f0 = np.median(f0_values)
    
    # Add relative position features for each speaker
    for speaker_id, features in feature_dict.items():
        if 'f0_median' in features and features['f0_median'] > 0:
            # Normalized position in the F0 range (0 = lowest, 1 = highest)
            if global_max_f0 > global_min_f0:
                features['f0_relative_position'] = (features['f0_median'] - global_min_f0) / (global_max_f0 - global_min_f0)
            
            # Difference from global median
            features['f0_diff_from_median'] = features['f0_median'] - global_median_f0

def detect_child_content(transcription):
    """
    Analyze transcript content for clues about speaker demographics.
    """
    # Child-related keywords and phrases
    child_indicators = [
        "school", "homework", "play", "toys", "mommy", "daddy", 
        "playground", "games", "mom", "dad", "teacher",
        "when i grow up", "big kid", "being young"
    ]
    
    # Join all text
    full_text = " ".join([seg["text"].lower() for seg in transcription])
    
    # Count matches
    matches = [word for word in child_indicators if word in full_text]
    child_likelihood = len(matches)
    
    # Estimate child count
    estimated_child_count = 0
    if child_likelihood > 0:
        if child_likelihood >= 5:
            estimated_child_count = 2  # Likely multiple children
        else:
            estimated_child_count = 1  # Likely one child
    
    if matches:
        print(f"Child content detected: {', '.join(matches)}")
        print(f"Estimated child count from content: at least {estimated_child_count}")
    
    return {
        'is_child_content': bool(matches),
        'estimated_child_count': estimated_child_count,
        'matches': matches
    }

def train_classifier(feature_dict, manual_labels):
    """
    Train a classifier using the provided features and labels.
    """
    print("Training speaker classifier...")
    
    # Prepare data for training
    X = []
    y = []
    feature_names = []
    speaker_ids = []
    
    # Identify common features across all speakers
    common_features = set()
    for features in feature_dict.values():
        if not common_features:
            common_features = set(features.keys())
        else:
            common_features &= set(features.keys())
    
    # Ensure we have all required speakers in the labels
    missing_speakers = set(feature_dict.keys()) - set(manual_labels.keys())
    if missing_speakers:
        print(f"Warning: Missing labels for speakers: {missing_speakers}")
        print("These speakers will be excluded from training")
    
    for speaker_id, features in feature_dict.items():
        if speaker_id not in manual_labels:
            continue
            
        # Extract feature vector
        feature_vector = []
        feature_names = []
        
        # Use only features available for all speakers
        for name in common_features:
            feature_vector.append(features[name])
            feature_names.append(name)
        
        X.append(feature_vector)
        y.append(manual_labels[speaker_id])
        speaker_ids.append(speaker_id)
    
    if len(X) < 2:
        print("Not enough labeled data for training")
        return None
    
    # Normalize features
    X = StandardScaler().fit_transform(X)
    
    # Train a classifier
    if HAVE_XGBOOST:
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            objective='multi:softprob',
            random_state=42
        )
    else:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=3,
            random_state=42
        )
    
    model.fit(X, y)
    print(f"Trained classifier with {len(X)} samples")
    
    # Save the model and feature names
    model_data = {
        'model': model,
        'feature_names': feature_names
    }
    model_path = os.path.join(MODEL_DIR, "speaker_classifier.joblib")
    joblib.dump(model_data, model_path)
    print(f"Saved model to {model_path}")
    
    return model_data

def classify_speakers(feature_dict, model_data=None, child_count_hint=None, force_child=False):
    """
    Classify speakers using a model or rules.
    """
    # If no model provided, try to load one
    if model_data is None:
        model_path = os.path.join(MODEL_DIR, "speaker_classifier.joblib")
        if os.path.exists(model_path):
            print(f"Loading classifier from {model_path}")
            try:
                model_data = joblib.load(model_path)
            except Exception as e:
                print(f"Error loading model: {e}")
                model_data = None
    
    # If we have a model, use it
    if model_data is not None and 'model' in model_data:
        return classify_with_model(feature_dict, model_data, child_count_hint, force_child)
    else:
        # Fallback to rule-based classification
        return classify_with_rules(feature_dict, child_count_hint, force_child)

def classify_with_model(feature_dict, model_data, child_count_hint=None, force_child=False):
    """
    Classify speakers using a trained model.
    """
    model = model_data['model']
    feature_names = model_data['feature_names']
    
    # Prepare data for classification
    X = []
    speaker_ids = []
    
    for speaker_id, features in feature_dict.items():
        # Extract features in the correct order
        feature_vector = []
        for name in feature_names:
            feature_vector.append(features.get(name, 0))
        
        X.append(feature_vector)
        speaker_ids.append(speaker_id)
    
    if not X:
        return {}
    
    # Normalize features
    X = StandardScaler().fit_transform(X)
    
    # Get predictions
    y_pred = model.predict(X)
    
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(X)
    else:
        probas = None
    
    # Map predictions to speaker IDs
    classifications = {}
    for i, speaker_id in enumerate(speaker_ids):
        classifications[speaker_id] = y_pred[i]
    
    # Handle child count hint
    if child_count_hint and child_count_hint > 0:
        current_child_count = sum(1 for label in classifications.values() if label == "Child")
        
        if current_child_count < child_count_hint:
            # Need to reclassify some speakers as children
            if probas is not None:
                # Use probabilities to find the most child-like speakers
                child_idx = list(model.classes_).index("Child") if "Child" in model.classes_ else -1
                
                if child_idx >= 0:
                    # Find non-child speakers with highest child probability
                    candidates = []
                    for i, speaker_id in enumerate(speaker_ids):
                        if classifications[speaker_id] != "Child":
                            candidates.append((speaker_id, probas[i][child_idx]))
                    
                    # Sort by probability (highest first)
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    
                    # Reclassify as many as needed
                    for i in range(min(len(candidates), child_count_hint - current_child_count)):
                        classifications[candidates[i][0]] = "Child"
                        print(f"Reclassified speaker {candidates[i][0]} as Child (probability: {candidates[i][1]:.2f})")
            else:
                # Fallback to using pitch
                non_child_speakers = [(spk_id, features.get('f0_median', 0)) 
                                    for spk_id, features in feature_dict.items() 
                                    if classifications[spk_id] != "Child" and 'f0_median' in features]
                
                # Sort by pitch (highest first)
                non_child_speakers.sort(key=lambda x: x[1], reverse=True)
                
                # Reclassify as many as needed
                for i in range(min(len(non_child_speakers), child_count_hint - current_child_count)):
                    classifications[non_child_speakers[i][0]] = "Child"
                    print(f"Reclassified speaker {non_child_speakers[i][0]} as Child (highest pitch)")
    
    # Handle force_child
    if force_child and "Child" not in classifications.values():
        # Find highest-pitched non-man speaker
        non_men = [(spk_id, features.get('f0_median', 0)) 
                  for spk_id, features in feature_dict.items() 
                  if classifications[spk_id] != "Man" and 'f0_median' in features]
        
        if non_men:
            # Sort by pitch (highest first)
            non_men.sort(key=lambda x: x[1], reverse=True)
            # Reclassify highest-pitched non-man as child
            classifications[non_men[0][0]] = "Child"
            print(f"Forced classification: Speaker {non_men[0][0]} as Child (highest non-male pitch)")
    
    print("\nModel Classification Results:")
    for speaker, label in classifications.items():
        print(f"Speaker {speaker} â†’ {label}")
    
    return classifications

def classify_with_rules(feature_dict, child_count_hint=None, force_child=False):
    """
    Classify speakers using rule-based approach.
    """
    print("\nUsing rule-based classification...")
    
    # Rule-based classification with improved thresholds
    classifications = {}
    
    for speaker_id, features in feature_dict.items():
        if 'f0_median' in features:
            f0 = features['f0_median']
            centroid = features.get('spectral_centroid', 0)
            zcr = features.get('zero_crossing_rate', 0)
            
            # Enhanced classification thresholds
            if f0 < 155:
                classifications[speaker_id] = "Man"
            elif f0 > 270 and centroid > 3200:
                # Very high pitch with bright spectral characteristics
                classifications[speaker_id] = "Child"
            elif f0 > 220 and zcr > 0.1:
                # Higher pitch and more zero crossings (noise)
                classifications[speaker_id] = "Child"
            elif f0 > 165:
                # Most likely a woman
                classifications[speaker_id] = "Woman"
            else:
                # Edge case
                classifications[speaker_id] = "Man"
        else:
            # Fallback
            centroid = features.get('spectral_centroid', 0)
            if centroid < 2000:
                classifications[speaker_id] = "Man"
            elif centroid > 3200:
                classifications[speaker_id] = "Child"
            else:
                classifications[speaker_id] = "Woman"
    
    # Handle child count hint
    if child_count_hint and child_count_hint > 0:
        current_child_count = sum(1 for label in classifications.values() if label == "Child")
        
        if current_child_count < child_count_hint:
            # Need to reclassify some speakers as children
            women_speakers = [(spk_id, features.get('f0_median', 0)) 
                             for spk_id, features in feature_dict.items() 
                             if classifications[spk_id] == "Woman" and 'f0_median' in features]
            
            # Sort by pitch (highest first)
            women_speakers.sort(key=lambda x: x[1], reverse=True)
            
            # Reclassify as many as needed
            for i in range(min(len(women_speakers), child_count_hint - current_child_count)):
                classifications[women_speakers[i][0]] = "Child"
                print(f"Reclassified speaker {women_speakers[i][0]} as Child (highest female pitch)")
    
    # Handle force_child
    if force_child and "Child" not in classifications.values():
        women_speakers = [(spk_id, features.get('f0_median', 0)) 
                          for spk_id, features in feature_dict.items() 
                          if classifications[spk_id] == "Woman" and 'f0_median' in features]
        
        if women_speakers:
            women_speakers.sort(key=lambda x: x[1], reverse=True)
            classifications[women_speakers[0][0]] = "Child"
            print(f"Forced classification: Speaker {women_speakers[0][0]} as Child (highest female pitch)")
    
    print("\nRule-based Classification Results:")
    for speaker, label in classifications.items():
        print(f"Speaker {speaker} â†’ {label}")
    
    return classifications

def number_speaker_classifications(classifications):
    """
    Add numeric suffixes to speaker classifications (Man1, Woman1, Child1, etc.)
    """
    if not classifications:
        return {}
    
    # Group speakers by their type
    type_groups = {}
    for speaker_id, speaker_type in classifications.items():
        if speaker_type not in type_groups:
            type_groups[speaker_type] = []
        type_groups[speaker_type].append(speaker_id)
    
    # Apply numbering
    numbered_classifications = {}
    for speaker_type, speaker_ids in type_groups.items():
        # Sort IDs to ensure consistent numbering
        sorted_ids = sorted(speaker_ids)
        for i, speaker_id in enumerate(sorted_ids):
            numbered_classifications[speaker_id] = f"{speaker_type}{i+1}"
    
    print("\nNumbered Speaker Classifications:")
    for speaker, label in numbered_classifications.items():
        print(f"Speaker {speaker} â†’ {label}")
    
    return numbered_classifications

def format_timestamp(seconds):
    """
    Formats seconds as HH:MM:SS
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def align_speakers_to_transcription(diarization, transcription_segments, speaker_classifications, include_timestamps=True):
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
                else:
                    speaker_label = "Unknown"
            else:
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
            if speaker_id is not None and speaker_id in speaker_classifications:
                speaker_label = speaker_classifications[speaker_id]
            else:
                speaker_label = "Unknown"
        
        # Format timestamps as HH:MM:SS
        if include_timestamps:
            start_formatted = format_timestamp(t_start)
            end_formatted = format_timestamp(t_end)
            timestamp_str = f"[{start_formatted} - {end_formatted}] "
        else:
            timestamp_str = ""
        
        # Build the final text line
        line = f"{timestamp_str}[{speaker_label}] {tseg['text']}"
        labeled_segments.append(line)
    
    print(f"Alignment completed in {time.time() - start_time:.2f} seconds")
    return labeled_segments

def visualize_speaker_features(feature_dict, classifications):
    """
    Creates a visualization of speaker features.
    """
    if len(feature_dict) < 2:
        print("Not enough speakers to visualize")
        return
        
    plt.figure(figsize=(10, 6))
    
    # Extract x and y coordinates for the plot
    x_values = []
    y_values = []
    labels = []
    colors = []
    
    color_map = {"Man": "blue", "Woman": "red", "Child": "green", "Unknown": "gray"}
    
    for speaker_id, features in feature_dict.items():
        if 'f0_median' in features and 'spectral_centroid' in features:
            x_values.append(features['f0_median'])
            y_values.append(features['spectral_centroid'])
            
            # Get base classification type (without numbers)
            base_type = classifications.get(speaker_id, "Unknown")
            if any(char.isdigit() for char in base_type):
                base_type = ''.join([c for c in base_type if not c.isdigit()])
                
            labels.append(f"{speaker_id}: {classifications.get(speaker_id, 'Unknown')}")
            colors.append(color_map.get(base_type, "gray"))
    
    plt.scatter(x_values, y_values, c=colors, s=100, alpha=0.7)
    
    # Add labels to each point
    for i, label in enumerate(labels):
        plt.annotate(label, (x_values[i], y_values[i]), fontsize=9)
    
    plt.xlabel("Fundamental Frequency (Hz)")
    plt.ylabel("Spectral Centroid")
    plt.title("Speaker Classification")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add reference lines
    plt.axvline(x=155, color='gray', linestyle='--', alpha=0.5, label="Man threshold")
    plt.axvline(x=220, color='gray', linestyle='--', alpha=0.5, label="Woman/Child threshold")
    
    plt.legend()
    plt.tight_layout()
    plt.savefig("speaker_features.png")
    print("Speaker feature visualization saved to speaker_features.png")

def launch_interactive_labeler(labels_file="speaker_labels.json"):
    """
    Launch the interactive speaker labeling tool.
    """
    print("\nLaunching interactive speaker labeling tool...")
    
    # Import the interactive labeling module
    try:
        # First check if the module is in the current directory
        import importlib.util
        spec = importlib.util.spec_from_file_location("interactive_labeler", "interactive_labeler.py")
        if spec and spec.loader:
            interactive_labeler = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(interactive_labeler)
            
            # Run the interactive labeler
            interactive_labeler.interactive_labeling(SPEAKER_SAMPLES_DIR, labels_file)
            return True
    except Exception as e:
        print(f"Error importing interactive labeler: {e}")
    
    # Fallback: try to run as a subprocess
    try:
        print("Trying to run interactive labeler as a subprocess...")
        cmd = [sys.executable, "interactive_labeler.py", "--samples", SPEAKER_SAMPLES_DIR, "--labels", labels_file]
        subprocess.run(cmd)
        return True
    except Exception as e:
        print(f"Error running interactive labeler: {e}")
        print("Please ensure interactive_labeler.py is in the current directory.")
        return False

def process_with_interactive_labeling(
    audio_path, 
    output_txt="transcript.txt", 
    force_child=False,
    child_count=None,
    include_timestamps=True,
    model_size="small",
    language=None,
    numbered_speakers=True
):
    """
    Process audio with interactive speaker labeling.
    
    Workflow:
    1. Diarize speakers and extract features
    2. Generate initial transcript with best-effort classification
    3. Launch interactive tool for user to correct speaker labels
    4. Train a custom model with the user's labels
    5. Generate final transcript with accurate classification
    """
    total_start_time = time.time()
    
    print(f"Processing audio file: {audio_path}")
    
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
        use_auth_token=os.getenv("HF_API_KEY", "YOUR_HF_API_KEY")
    )
    if torch.cuda.is_available():
        diarization_pipeline.to(torch.device("cuda"))
    
    print(f"Models loaded in {time.time() - model_start:.2f} seconds")
    
    # 2. Run diarization and transcription in parallel
    print("Processing audio...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        diarization_future = executor.submit(diarize_audio, diarization_pipeline, audio_path)
        transcription_future = executor.submit(transcribe_audio, whisper_model, audio_path, language)
        
        diarization = diarization_future.result()
        transcription = transcription_future.result()
    
    # 3. Analyze content for child-related topics
    content_info = detect_child_content(transcription)
    
    # Update child count if needed
    if force_child and (child_count is None or child_count < 1):
        child_count = max(1, content_info['estimated_child_count'])
        print(f"Child detection forced. Expected count: {child_count}")
    elif child_count is None:
        child_count = content_info['estimated_child_count']
    
    # 4. Extract speaker features
    speaker_features = extract_speaker_features(audio_path, diarization)
    
    # 5. Generate initial classification
    initial_classifications = classify_with_rules(
        speaker_features, 
        child_count_hint=child_count,
        force_child=force_child
    )
    
    if numbered_speakers:
        initial_classifications = number_speaker_classifications(initial_classifications)
    
    # 6. Generate initial transcript
    initial_transcript = align_speakers_to_transcription(
        diarization, transcription, initial_classifications, include_timestamps
    )
    
    # Save initial transcript
    initial_transcript_text = "\n".join(initial_transcript)
    initial_output = output_txt.replace(".txt", "_initial.txt")
    with open(initial_output, "w", encoding="utf-8") as f:
        f.write(initial_transcript_text)
    
    print(f"\nInitial transcript saved to {initial_output}")
    print("\nInitial Transcript Preview:")
    print(NEON_GREEN + "\n".join(initial_transcript[:10]) + RESET_COLOR)
    
    # 7. Launch interactive labeling
    print("\n" + "=" * 80)
    print("Initial processing complete. Now launching interactive labeling tool.")
    print("Please listen to each speaker and assign the correct label (Man, Woman, or Child).")
    print("After labeling, a custom model will be trained to improve classification accuracy.")
    print("=" * 80 + "\n")
    
    # Define the labels file path
    labels_file = os.path.join(os.path.dirname(output_txt), "speaker_labels.json")
    
    # Launch the interactive labeling tool
    labeling_success = launch_interactive_labeler(labels_file)
    
    if not labeling_success:
        print("\nInteractive labeling failed. Using initial classification for final transcript.")
        final_classifications = initial_classifications
    else:
        # 8. Load the user's labels
        try:
            with open(labels_file, 'r') as f:
                manual_labels = json.load(f)
                print(f"Loaded manual labels for {len(manual_labels)} speakers")
            
            # 9. Train a custom model with the labels
            model_data = train_classifier(speaker_features, manual_labels)
            
            # 10. Generate final classification
            if model_data:
                final_classifications = classify_with_model(
                    speaker_features, 
                    model_data,
                    child_count_hint=child_count,
                    force_child=force_child
                )
            else:
                # Fallback to direct application of manual labels
                final_classifications = manual_labels.copy()
                print("Using manual labels directly")
            
            if numbered_speakers:
                final_classifications = number_speaker_classifications(final_classifications)
            
        except Exception as e:
            print(f"\nError processing manual labels: {e}")
            print("Using initial classification for final transcript")
            final_classifications = initial_classifications
    
    # 11. Generate final transcript
    final_transcript = align_speakers_to_transcription(
        diarization, transcription, final_classifications, include_timestamps
    )
    
    # Save final transcript
    final_transcript_text = "\n".join(final_transcript)
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(final_transcript_text)
    
    # 12. Visualize speaker features
    try:
        visualize_speaker_features(speaker_features, final_classifications)
    except Exception as e:
        print(f"Error creating visualization: {e}")
    
    total_time = time.time() - total_start_time
    print(f"\nProcessing completed in {total_time:.2f} seconds")
    print(f"Final transcript saved to {output_txt}")
    
    print("\nFinal Transcript Preview:")
    print(NEON_GREEN + "\n".join(final_transcript[:10]) + RESET_COLOR)
    
    return {
        'transcript': final_transcript_text,
        'classifications': final_classifications,
        'features': speaker_features,
        'diarization': diarization
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Speaker classifier with interactive labeling")
    parser.add_argument("audio_file", help="Path to the audio file to process")
    parser.add_argument("-o", "--output", default="transcript.txt", help="Output file for the transcript")
    parser.add_argument("--child", action="store_true", help="Force child classification")
    parser.add_argument("--child-count", type=int, help="Expected number of children in the recording")
    parser.add_argument("--no-timestamps", action="store_true", help="Exclude timestamps")
    parser.add_argument("--model", default="small", choices=["tiny", "base", "small", "medium"], 
                        help="Whisper model size (smaller is faster)")
    parser.add_argument("--language", help="Specify language code (e.g., 'en' for English)")
    parser.add_argument("--no-numbering", action="store_true", help="Disable speaker numbering")
    
    args = parser.parse_args()
    
    process_with_interactive_labeling(
        args.audio_file, 
        args.output, 
        force_child=args.child,
        child_count=args.child_count,
        include_timestamps=not args.no_timestamps,
        model_size=args.model,
        language=args.language,
        numbered_speakers=not args.no_numbering
    )
