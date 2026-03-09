import os
import torch
import numpy as np
import librosa
import soundfile as sf
import time
import concurrent.futures
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import xgboost as xgb
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment

# For specialized voice classification 
import torchaudio
from speechbrain.pretrained import EncoderClassifier

# Global constants
AUDIO_SAMPLE_RATE = 16000
MIN_SEGMENT_DURATION = 1.0  # seconds
FEATURE_CACHE_DIR = "feature_cache"
MODEL_DIR = "models"

# Color codes for printing
NEON_GREEN = "\033[92m"
RESET_COLOR = "\033[0m"

# Ensure required directories exist
os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("speaker_samples", exist_ok=True)

def visualize_speaker_features(feature_dict, classifications):
    """
    Creates a visualization of speaker features.
    """
    if len(feature_dict) < 2:
        print("Not enough speakers to visualize")
        return
        
    plt.figure(figsize=(15, 10))
    
    # Check which features are available for all speakers
    common_features = set()
    for features in feature_dict.values():
        if not common_features:
            common_features = set(features.keys())
        else:
            common_features &= set(features.keys())
    
    # Focus on fundamental frequency and spectral features
    if 'f0_median' in common_features and 'spectral_centroid_mean' in common_features:
        # 2D plot: F0 vs Spectral Centroid
        plt.subplot(2, 2, 1)
        
        x_values = []
        y_values = []
        labels = []
        colors = []
        
        color_map = {"Man": "blue", "Woman": "red", "Child": "green", "Unknown": "gray"}
        
        for speaker_id, features in feature_dict.items():
            x_values.append(features['f0_median'])
            y_values.append(features['spectral_centroid_mean'])
            
            # Get base classification type (without numbers)
            base_type = classifications.get(speaker_id, "Unknown")
            if any(char.isdigit() for char in base_type):
                base_type = ''.join([c for c in base_type if not c.isdigit()])
                
            labels.append(f"{speaker_id}: {classifications.get(speaker_id, 'Unknown')}")
            colors.append(color_map.get(base_type, "gray"))
        
        plt.scatter(x_values, y_values, c=colors, s=150, alpha=0.7)
        
        # Add labels to each point
        for i, label in enumerate(labels):
            plt.annotate(label, (x_values[i], y_values[i]), fontsize=11)
        
        plt.xlabel("Fundamental Frequency (Hz)")
        plt.ylabel("Spectral Centroid")
        plt.title("Speaker Classification: Pitch vs Spectral Centroid")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add reference lines
        plt.axvline(x=155, color='gray', linestyle='--', alpha=0.5, label="Man threshold")
        plt.axvline(x=220, color='gray', linestyle='--', alpha=0.5, label="Woman/Child threshold")
        
        plt.legend()
    
    # Feature distributions
    if 'f0_median' in common_features:
        # Plot fundamental frequency distribution
        plt.subplot(2, 2, 2)
        
        man_f0 = [features['f0_median'] for speaker_id, features in feature_dict.items() 
                 if classifications.get(speaker_id, "").startswith("Man")]
        woman_f0 = [features['f0_median'] for speaker_id, features in feature_dict.items() 
                   if classifications.get(speaker_id, "").startswith("Woman")]
        child_f0 = [features['f0_median'] for speaker_id, features in feature_dict.items() 
                   if classifications.get(speaker_id, "").startswith("Child")]
        
        plt.hist([man_f0, woman_f0, child_f0], bins=10, 
                 label=['Men', 'Women', 'Children'], 
                 color=['blue', 'red', 'green'],
                 alpha=0.7)
        plt.xlabel("Fundamental Frequency (Hz)")
        plt.ylabel("Count")
        plt.title("Distribution of Fundamental Frequencies")
        plt.legend()
    
    # Feature importance (if we have enough speakers)
    if len(feature_dict) >= 3:
        plt.subplot(2, 2, 3)
        
        # Select most important features
        feature_names = ['f0_median', 'spectral_centroid_mean', 'spectral_rolloff_mean', 
                       'zero_crossing_rate', 'harmonic_ratio']
        feature_names = [f for f in feature_names if f in common_features]
        
        if feature_names:
            # Create a feature importance plot
            importances = []
            for feature in feature_names:
                man_vals = [features.get(feature, 0) for speaker_id, features in feature_dict.items() 
                           if classifications.get(speaker_id, "").startswith("Man")]
                woman_vals = [features.get(feature, 0) for speaker_id, features in feature_dict.items() 
                             if classifications.get(speaker_id, "").startswith("Woman")]
                child_vals = [features.get(feature, 0) for speaker_id, features in feature_dict.items() 
                             if classifications.get(speaker_id, "").startswith("Child")]
                
                # Calculate separation power (simplified)
                if man_vals and (woman_vals or child_vals):
                    man_mean = np.mean(man_vals)
                    other_mean = np.mean(woman_vals + child_vals)
                    importance = abs(man_mean - other_mean) / (np.std(man_vals) + np.std(woman_vals + child_vals) + 1e-10)
                    importances.append(importance)
                else:
                    importances.append(0)
            
            # Sort features by importance
            sorted_indices = np.argsort(importances)[::-1]
            sorted_features = [feature_names[i] for i in sorted_indices]
            sorted_importances = [importances[i] for i in sorted_indices]
            
            # Plot
            plt.barh(sorted_features, sorted_importances, color='teal')
            plt.xlabel("Discriminative Power")
            plt.title("Feature Importance for Speaker Classification")
    
    # Raw feature values table
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    table_data = []
    headers = ['Speaker', 'Type', 'F0 (Hz)', 'Spectral\nCentroid', 'Zero\nCrossing']
    
    for speaker_id, features in feature_dict.items():
        row = [
            speaker_id,
            classifications.get(speaker_id, "Unknown"),
            f"{features.get('f0_median', 0):.1f}",
            f"{features.get('spectral_centroid_mean', 0)/1000:.1f}k",
            f"{features.get('zero_crossing_rate', 0):.3f}"
        ]
        table_data.append(row)
    
    # Sort by speaker type
    table_data.sort(key=lambda x: x[1])
    
    # Draw table
    table = plt.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colWidths=[0.15, 0.15, 0.2, 0.2, 0.2]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.title("Speaker Acoustic Features", pad=20)
    
    plt.tight_layout()
    plt.savefig("speaker_features.png", dpi=150)
    print("Speaker feature visualization saved to speaker_features.png")

def process_with_premium_classifier(
    audio_path, 
    output_txt="transcript.txt", 
    force_child=False,
    child_count=None,
    include_timestamps=True,
    model_size="small",
    language=None,
    numbered_speakers=True,
    manual_labels=None,
    train_custom_model=False
):
    """
    Process audio with premium speaker classification.
    
    Parameters:
    - audio_path: Path to audio file
    - output_txt: Output transcript file
    - force_child: Ensure at least one speaker is classified as child
    - child_count: Expected number of children
    - include_timestamps: Add timestamps to transcript
    - model_size: Whisper model size
    - language: Language code for transcription
    - numbered_speakers: Add numbers to speaker labels
    - manual_labels: Optional dict mapping speaker IDs to labels
    - train_custom_model: Force training a new model
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
        use_auth_token=os.getenv("HF_API_KEY", "YOUR_HF_API_KEY")
    )
    if torch.cuda.is_available():
        diarization_pipeline.to(torch.device("cuda"))
    
    print(f"Models loaded in {time.time() - model_start:.2f} seconds")
    
    # 2. Run diarization and transcription in parallel for speed
    print("Processing audio...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Start both tasks
        diarization_future = executor.submit(diarize_audio, diarization_pipeline, audio_path)
        transcription_future = executor.submit(transcribe_audio, whisper_model, audio_path, language)
        
        # Wait for both to complete
        diarization = diarization_future.result()
        transcription = transcription_future.result()
    
    # 3. Analyze content for demographic clues
    content_info = analyze_content_for_demographics(transcription)
    
    # Update child count if needed
    if force_child and (child_count is None or child_count < 1):
        child_count = max(1, content_info['estimated_child_count'])
        print(f"Child detection forced. Using count: {child_count}")
    elif child_count is None:
        child_count = content_info['estimated_child_count']
        if child_count > 0:
            print(f"Detected {child_count} children from content analysis")
    
    # 4. Extract comprehensive features
    speaker_features = extract_comprehensive_features(audio_path, diarization)
    
    # 5. Try to load or train a classifier
    if manual_labels or train_custom_model:
        classifier = train_or_load_classifier(speaker_features, manual_labels, force_retrain=train_custom_model)
        
        if classifier:
            # Use the trained model for classification
            classifications = classify_speakers_with_model(
                speaker_features, 
                classifier,
                child_count_hint=child_count,
                force_child=force_child
            )
        else:
            # Fallback to rule-based classification
            classifications = fallback_classification(
                speaker_features,
                child_count_hint=child_count,
                force_child=force_child
            )
    else:
        # Try to load a pre-trained model
        try:
            classifier = train_or_load_classifier()
            if classifier:
                classifications = classify_speakers_with_model(
                    speaker_features, 
                    classifier,
                    child_count_hint=child_count,
                    force_child=force_child
                )
            else:
                classifications = fallback_classification(
                    speaker_features,
                    child_count_hint=child_count,
                    force_child=force_child
                )
        except Exception as e:
            print(f"Error loading classifier: {e}")
            classifications = fallback_classification(
                speaker_features,
                child_count_hint=child_count,
                force_child=force_child
            )
    
    # 6. Apply numbering if requested
    if numbered_speakers:
        classifications = number_speaker_classifications(classifications)
    
    # 7. Visualize speaker features
    try:
        visualize_speaker_features(speaker_features, classifications)
    except Exception as e:
        print(f"Error creating visualization: {e}")
    
    # 8. Align transcription with speaker labels
    labeled_transcript = align_speakers_to_transcription(
        diarization, transcription, classifications, include_timestamps
    )
    
    # 9. Save the transcript
    transcript_text = "\n".join(labeled_transcript)
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(transcript_text)
    
    total_time = time.time() - total_start_time
    print(f"\nProcessing completed in {total_time:.2f} seconds")
    print(f"Transcript saved to {output_txt}")
    
    print("\nTranscript Preview:")
    print(NEON_GREEN + "\n".join(labeled_transcript[:10]) + RESET_COLOR)
    
    return {
        'transcript': transcript_text,
        'speaker_features': speaker_features,
        'classifications': classifications,
        'diarization': diarization
    }

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

def extract_comprehensive_features(audio_path, diarization):
    """
    Extracts a comprehensive set of features for speaker classification.
    """
    print("Extracting comprehensive speaker features...")
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
    
    # Try to load pre-trained SpeechBrain classifier
    try:
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            savedir="pretrained_models/spkrec-ecapa-voxceleb"
        )
        have_classifier = True
        print("Successfully loaded SpeechBrain classifier")
    except Exception as e:
        print(f"Could not load SpeechBrain classifier: {e}")
        have_classifier = False
    
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
            output_path = f"speaker_samples/speaker_{speaker_id}_segment_{i+1}.wav"
            sf.write(output_path, segment_audio, sr)
            
            # 1. Extract standard acoustic features
            features = {}
            
            # Pitch features
            try:
                f0, voiced_flag, _ = librosa.pyin(
                    segment_audio, 
                    fmin=60, 
                    fmax=600,
                    sr=sr,
                    frame_length=1024,
                    hop_length=256
                )
                
                # Filter out NaN values
                valid_f0 = f0[~np.isnan(f0)]
                
                if len(valid_f0) > 0:
                    features['f0_median'] = np.median(valid_f0)
                    features['f0_mean'] = np.mean(valid_f0)
                    features['f0_std'] = np.std(valid_f0)
                    features['f0_min'] = np.min(valid_f0)
                    features['f0_max'] = np.max(valid_f0)
                    features['f0_range'] = features['f0_max'] - features['f0_min']
            except Exception as e:
                print(f"Error extracting pitch for speaker {speaker_id}: {e}")
            
            # Spectral features
            try:
                # Spectral centroid (center of mass of the spectrum)
                cent = librosa.feature.spectral_centroid(y=segment_audio, sr=sr)[0]
                features['spectral_centroid_mean'] = np.mean(cent)
                features['spectral_centroid_std'] = np.std(cent)
                
                # Spectral bandwidth (variance around the centroid)
                bandwidth = librosa.feature.spectral_bandwidth(y=segment_audio, sr=sr)[0]
                features['spectral_bandwidth_mean'] = np.mean(bandwidth)
                
                # Spectral rolloff (frequency below which X% of energy is contained)
                rolloff = librosa.feature.spectral_rolloff(y=segment_audio, sr=sr)[0]
                features['spectral_rolloff_mean'] = np.mean(rolloff)
                
                # Spectral contrast (valleys and peaks in the spectrum)
                contrast = librosa.feature.spectral_contrast(y=segment_audio, sr=sr)
                features['spectral_contrast_mean'] = np.mean(contrast)
                
                # Spectral flatness (measure of noise vs. harmonic content)
                flatness = librosa.feature.spectral_flatness(y=segment_audio)
                features['spectral_flatness_mean'] = np.mean(flatness)
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
                
                # Zero crossing rate (higher for more noisy signals)
                zcr = librosa.feature.zero_crossing_rate(segment_audio)[0]
                features['zero_crossing_rate'] = np.mean(zcr)
            except Exception as e:
                print(f"Error extracting harmonic features for speaker {speaker_id}: {e}")
            
            # Voice quality features
            try:
                # RMS energy
                rms = librosa.feature.rms(y=segment_audio)[0]
                features['rms_energy_mean'] = np.mean(rms)
                features['rms_energy_std'] = np.std(rms)
                
                # Estimate speech rate from energy peaks
                onset_env = librosa.onset.onset_strength(y=segment_audio, sr=sr)
                tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
                features['speech_tempo'] = tempo
            except Exception as e:
                print(f"Error extracting voice quality features for speaker {speaker_id}: {e}")
            
            # MFCC features (capture vocal tract configuration)
            try:
                mfccs = librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=13)
                for i in range(13):
                    features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
                    features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
                
                # Delta and delta-delta features (capture dynamics)
                mfcc_delta = librosa.feature.delta(mfccs)
                mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
                
                for i in range(13):
                    features[f'mfcc_delta_{i+1}_mean'] = np.mean(mfcc_delta[i])
                    features[f'mfcc_delta2_{i+1}_mean'] = np.mean(mfcc_delta2[i])
            except Exception as e:
                print(f"Error extracting MFCC features for speaker {speaker_id}: {e}")
            
            # 2. Extract neural embeddings if classifier is available
            if have_classifier:
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
                # Only keep a few principal components to avoid too many features
                for i in range(min(10, len(avg_embedding))):
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
    global_range_f0 = global_max_f0 - global_min_f0
    
    # Add relative position features for each speaker
    for speaker_id, features in feature_dict.items():
        if 'f0_median' in features and features['f0_median'] > 0:
            # Normalized position in the F0 range (0 = lowest, 1 = highest)
            if global_range_f0 > 0:
                features['f0_relative_position'] = (features['f0_median'] - global_min_f0) / global_range_f0
            
            # How many standard deviations from the median
            if len(f0_values) > 1:
                std_dev = np.std(f0_values)
                if std_dev > 0:
                    features['f0_z_score'] = (features['f0_median'] - global_median_f0) / std_dev

def analyze_content_for_demographics(transcription):
    """
    Analyze transcript content for clues about speaker demographics.
    """
    # Join all text
    full_text = " ".join([seg["text"].lower() for seg in transcription])
    
    # Child-related indicators
    child_indicators = {
        'core': ["school", "homework", "play", "toys", "mommy", "daddy", "playground", "games"],
        'family': ["mom", "dad", "parent", "brother", "sister"],
        'activities': ["cowboys and indians", "play", "toys", "playground", "games"],
        'emotional': ["cry", "hugs", "scary", "fun", "excited"],
        'developmental': ["when i grow up", "big kid", "being young", "older"]
    }
    
    # Adult female indicators (more common in women's speech)
    female_indicators = ["husband", "makeup", "daughter", "sister", "mom", "mother", "she said"]
    
    # Adult male indicators (more common in men's speech)
    male_indicators = ["wife", "son", "brother", "dad", "father", "he said"]
    
    # Count matches
    matches = {}
    for category, words in child_indicators.items():
        category_matches = [word for word in words if word in full_text]
        if category_matches:
            matches[category] = category_matches
    
    female_matches = [word for word in female_indicators if word in full_text]
    male_matches = [word for word in male_indicators if word in full_text]
    
    # Estimate speaker demographics
    child_likelihood = len([item for sublist in matches.values() for item in sublist])
    female_likelihood = len(female_matches)
    male_likelihood = len(male_matches)
    
    # Estimate child count based on context clues
    estimated_child_count = 0
    if child_likelihood > 0:
        if child_likelihood >= 5:
            estimated_child_count = 2  # Likely multiple children
        else:
            estimated_child_count = 1  # Likely one child
    
    if matches:
        match_list = [item for sublist in matches.values() for item in sublist]
        print(f"Child content detected: {', '.join(match_list)}")
        print(f"Estimated child count from content: at least {estimated_child_count}")
    
    return {
        'child_likelihood': child_likelihood,
        'female_likelihood': female_likelihood,
        'male_likelihood': male_likelihood,
        'estimated_child_count': estimated_child_count
    }

def train_or_load_classifier(feature_dict=None, manual_labels=None, force_retrain=False):
    """
    Train a classifier or load a pre-trained one.
    
    If manual_labels is provided, it should be a dict mapping speaker_ids to labels.
    """
    model_path = os.path.join(MODEL_DIR, "speaker_classifier.joblib")
    
    # If not forcing retraining and a model exists, load it
    if not force_retrain and os.path.exists(model_path) and feature_dict is None:
        print(f"Loading pre-trained classifier from {model_path}")
        return joblib.load(model_path)
    
    # If we don't have features to train on, we can't train a model
    if feature_dict is None:
        print("No features provided for training. Please provide features or a pre-trained model.")
        return None
    
    print("Training custom speaker classifier...")
    
    # If no manual labels are provided, use a rule-based approach to create initial labels
    if manual_labels is None:
        manual_labels = {}
        for speaker_id, features in feature_dict.items():
            if 'f0_median' not in features:
                continue
                
            f0 = features['f0_median']
            if f0 < 150:
                manual_labels[speaker_id] = "Man"
            elif f0 > 250:
                manual_labels[speaker_id] = "Child"
            else:
                manual_labels[speaker_id] = "Woman"
    
    # Prepare data for training
    X = []
    y = []
    speaker_ids = []
    
    for speaker_id, features in feature_dict.items():
        if speaker_id not in manual_labels:
            continue
            
        # Extract a fixed set of features
        feature_vector = []
        for feature_name in ['f0_median', 'f0_std', 'spectral_centroid_mean', 
                           'spectral_rolloff_mean', 'harmonic_ratio', 
                           'zero_crossing_rate', 'speech_tempo']:
            feature_vector.append(features.get(feature_name, 0))
        
        # Add some MFCC features
        for i in range(1, 6):
            feature_vector.append(features.get(f'mfcc_{i}_mean', 0))
        
        # Add some embedding features if available
        for i in range(5):
            feature_vector.append(features.get(f'embedding_{i}', 0))
        
        X.append(feature_vector)
        y.append(manual_labels[speaker_id])
        speaker_ids.append(speaker_id)
    
    if len(X) < 2:
        print("Not enough data for training.")
        return None
    
    # Normalize features
    X = StandardScaler().fit_transform(X)
    
    # Train an XGBoost classifier
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        objective='multi:softprob',
        random_state=42
    )
    
    model.fit(X, y)
    
    # Save the model
    joblib.dump(model, model_path)
    
    print(f"Classifier trained and saved to {model_path}")
    return model

def classify_speakers_with_model(feature_dict, model, child_count_hint=None, force_child=False):
    """
    Classify speakers using a trained model.
    """
    if not feature_dict or model is None:
        return {}
    
    # Prepare features for classification
    X = []
    speaker_ids = []
    
    for speaker_id, features in feature_dict.items():
        # Extract a fixed set of features
        feature_vector = []
        for feature_name in ['f0_median', 'f0_std', 'spectral_centroid_mean', 
                           'spectral_rolloff_mean', 'harmonic_ratio', 
                           'zero_crossing_rate', 'speech_tempo']:
            feature_vector.append(features.get(feature_name, 0))
        
        # Add some MFCC features
        for i in range(1, 6):
            feature_vector.append(features.get(f'mfcc_{i}_mean', 0))
        
        # Add some embedding features if available
        for i in range(5):
            feature_vector.append(features.get(f'embedding_{i}', 0))
        
        X.append(feature_vector)
        speaker_ids.append(speaker_id)
    
    if not X:
        return {}
    
    # Normalize features
    X = StandardScaler().fit_transform(X)
    
    # Get predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Map predictions to speaker IDs
    classifications = {}
    prob_dict = {}
    
    for i, speaker_id in enumerate(speaker_ids):
        classifications[speaker_id] = predictions[i]
        # Store probabilities for each class
        prob_dict[speaker_id] = {
            cls: prob for cls, prob in zip(model.classes_, probabilities[i])
        }
    
    # Handle child count hint
    if child_count_hint and child_count_hint > 0:
        current_child_count = sum(1 for label in classifications.values() if label == "Child")
        
        if current_child_count < child_count_hint:
            # Need more children, find candidates by high Child probability
            candidates = []
            for speaker_id, probs in prob_dict.items():
                if classifications[speaker_id] != "Child" and "Child" in probs:
                    candidates.append((speaker_id, probs["Child"]))
            
            # Sort by probability of being a child (highest first)
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Reclassify as many as needed
            for i in range(min(len(candidates), child_count_hint - current_child_count)):
                classifications[candidates[i][0]] = "Child"
                print(f"Reclassified speaker {candidates[i][0]} as Child (probability: {candidates[i][1]:.2f})")
    
    # Handle force_child
    if force_child and "Child" not in classifications.values():
        # Find the speaker with highest Child probability
        max_prob = 0
        max_speaker = None
        
        for speaker_id, probs in prob_dict.items():
            if "Child" in probs and probs["Child"] > max_prob:
                max_prob = probs["Child"]
                max_speaker = speaker_id
        
        if max_speaker:
            classifications[max_speaker] = "Child"
            print(f"Forced classification: Speaker {max_speaker} as Child (probability: {max_prob:.2f})")
        else:
            # Fallback: use the highest-pitched speaker that's not a man
            non_men = [(spk_id, features.get('f0_median', 0)) 
                      for spk_id, features in feature_dict.items() 
                      if classifications.get(spk_id) != "Man" and 'f0_median' in features]
            
            if non_men:
                # Sort by pitch (highest first)
                non_men.sort(key=lambda x: x[1], reverse=True)
                # Reclassify highest-pitched non-man as child
                classifications[non_men[0][0]] = "Child"
                print(f"Forced classification: Speaker {non_men[0][0]} as Child (highest non-male pitch)")
    
    print("\nModel Classification Results:")
    for speaker, label in classifications.items():
        probs_str = ", ".join([f"{cls}: {prob:.2f}" for cls, prob in prob_dict[speaker].items()])
        print(f"Speaker {speaker} â†’ {label} (probabilities: {probs_str})")
    
    return classifications

def fallback_classification(feature_dict, child_count_hint=None, force_child=False):
    """
    Fallback classification using rules when model is not available.
    """
    if not feature_dict:
        return {}
    
    print("\nUsing rule-based classification (fallback)...")
    
    # Rule-based classification with improved thresholds
    classifications = {}
    
    for speaker_id, features in feature_dict.items():
        if 'f0_median' in features:
            f0 = features['f0_median']
            centroid = features.get('spectral_centroid_mean', 0)
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
            centroid = features.get('spectral_centroid_mean', 0)
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

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Premium speaker classification for audio files")
    parser.add_argument("audio_file", help="Path to the audio file to process")
    parser.add_argument("-o", "--output", default="transcript.txt", help="Output file for the transcript")
    parser.add_argument("--child", action="store_true", help="Force child classification")
    parser.add_argument("--child-count", type=int, help="Expected number of children in the recording")
    parser.add_argument("--no-timestamps", action="store_true", help="Exclude timestamps")
    parser.add_argument("--model", default="small", choices=["tiny", "base", "small", "medium"], 
                        help="Whisper model size (smaller is faster)")
    parser.add_argument("--language", help="Specify language code (e.g., 'en' for English)")
    parser.add_argument("--no-numbering", action="store_true", help="Disable speaker numbering")
    parser.add_argument("--train", action="store_true", help="Train a custom model")
    parser.add_argument("--labels", help="Path to JSON file with manual speaker labels")
    
    args = parser.parse_args()
    
    # Load manual labels if provided
    manual_labels = None
    if args.labels:
        import json
        try:
            with open(args.labels, 'r') as f:
                manual_labels = json.load(f)
            print(f"Loaded manual labels for {len(manual_labels)} speakers")
        except Exception as e:
            print(f"Error loading manual labels: {e}")
    
    process_with_premium_classifier(
        args.audio_file, 
        args.output, 
        force_child=args.child,
        child_count=args.child_count,
        include_timestamps=not args.no_timestamps,
        model_size=args.model,
        language=args.language,
        numbered_speakers=not args.no_numbering,
        manual_labels=manual_labels,
        train_custom_model=args.train
    )
