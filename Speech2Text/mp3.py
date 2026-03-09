import os
import torch
import numpy as np
import librosa
import soundfile as sf
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment

# Color codes for printing
NEON_GREEN = "\033[92m"
RESET_COLOR = "\033[0m"

def diarize_audio(pipeline, audio_path):
    """
    Runs speaker diarization on the audio file and returns
    a pyannote.core.Annotation object containing speaker segments.
    """
    print(f"Running diarization on {audio_path}...")
    return pipeline(audio_path)

def transcribe_audio(whisper_model, audio_path):
    """
    Transcribes the audio using faster-whisper and returns a list of segments.
    Each segment has {start, end, text}.
    """
    print(f"Transcribing {audio_path}...")
    segments, info = whisper_model.transcribe(audio_path, beam_size=7)
    result = []
    for seg in segments:
        result.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip()
        })
    return result

def extract_and_analyze_speaker_features(audio_path, diarization, min_duration=1.0):
    """
    Extracts audio features for each speaker and builds feature vectors
    for more sophisticated classification.
    
    Returns a dictionary mapping speaker IDs to feature dictionaries.
    """
    print("Extracting features for each speaker...")
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
        
        # Take the top 5 longest segments (or fewer if not available)
        valid_segments = [s for s in segments[:10] if s.duration >= min_duration]
        
        if not valid_segments:
            continue
            
        # Extract features for each valid segment
        f0_values = []
        spectral_centroids = []
        spectral_rolloffs = []
        spectral_flatness = []
        mfccs_means = []
        
        # Create a directory for saving speaker samples
        os.makedirs("speaker_samples", exist_ok=True)
        
        for i, segment in enumerate(valid_segments[:5]):  # Limit to 5 samples per speaker
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
                
                flatness = np.mean(librosa.feature.spectral_flatness(y=segment_audio))
                spectral_flatness.append(flatness)
                
                # MFCCs (Mel-Frequency Cepstral Coefficients) - capture vocal tract configuration
                mfccs = librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=13)
                mfccs_means.append(np.mean(mfccs, axis=1))
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
        if spectral_flatness:
            features['spectral_flatness'] = np.mean(spectral_flatness)
        if mfccs_means:
            features['mfccs'] = np.mean(mfccs_means, axis=0)
        
        if features:
            speaker_features[speaker_id] = features
            print(f"Speaker {speaker_id} features extracted successfully")
    
    return speaker_features

def classify_speakers(speaker_features, use_advanced=True, context_hints=None):
    """
    Classifies speakers as "Man", "Woman", or "Child" based on extracted features.
    
    Parameters:
    - speaker_features: Dictionary of features for each speaker
    - use_advanced: Whether to use clustering for classification
    - context_hints: Optional dictionary with hints like {'child_present': True}
    """
    if not speaker_features:
        return {}
        
    # Print all features for debugging
    print("\nSpeaker Features Summary:")
    for speaker_id, features in speaker_features.items():
        feature_str = ", ".join([f"{k}: {v:.2f}" if isinstance(v, (float, int)) and not isinstance(v, np.ndarray) 
                               else f"{k}: [...]" for k, v in features.items()])
        print(f"Speaker {speaker_id}: {feature_str}")
    
    # Try clustering-based approach if we have enough speakers and advanced mode is enabled
    if len(speaker_features) >= 3 and use_advanced:
        try:
            return classify_speakers_clustering(speaker_features)
        except Exception as e:
            print(f"Clustering failed: {e}. Falling back to rule-based classification.")
    
    # Child/adult conversation detection
    is_child_adult_conversation = False
    speaker_count = len(speaker_features)
    
    # Check if this is likely a two-person conversation with distinct vocal ranges
    if speaker_count == 2 and context_hints and context_hints.get('child_present', False):
        is_child_adult_conversation = True
        print("Conversation context: Child-adult conversation detected")
    elif speaker_count == 2:
        # Get the two speakers
        speakers = list(speaker_features.keys())
        if 'f0_median' in speaker_features[speakers[0]] and 'f0_median' in speaker_features[speakers[1]]:
            f0_values = [speaker_features[speakers[0]]['f0_median'], speaker_features[speakers[1]]['f0_median']]
            f0_diff = abs(f0_values[0] - f0_values[1])
            
            # If there's a big difference in pitch, likely adult-child
            if f0_diff > 80 and max(f0_values) > 200:
                is_child_adult_conversation = True
                print(f"Conversation context: Detected likely child-adult conversation (F0 difference: {f0_diff:.2f}Hz)")
    
    # Rule-based classification
    classifications = {}
    
    for speaker_id, features in speaker_features.items():
        if 'f0_median' in features:
            f0 = features['f0_median']
            centroid = features.get('spectral_centroid', 0)
            
            # More aggressive child detection for interviews/conversations
            if is_child_adult_conversation:
                # In a child-adult conversation, the speaker with higher pitch is likely the child
                if f0 < 170:
                    classifications[speaker_id] = "Man"
                elif f0 > 170:  # Lower threshold for child detection in this context
                    classifications[speaker_id] = "Child"
                else:
                    classifications[speaker_id] = "Woman"
            else:
                # Standard classification
                if f0 < 150:
                    classifications[speaker_id] = "Man"
                elif f0 > 230 or (f0 > 200 and centroid > 2800):  # Adjusted thresholds for better child detection
                    classifications[speaker_id] = "Child"
                else:
                    classifications[speaker_id] = "Woman"
        else:
            # Fallback if no F0 available
            centroid = features.get('spectral_centroid', 0)
            if centroid < 2000:
                classifications[speaker_id] = "Man"
            elif centroid > 2800:  # Lower threshold for child detection
                classifications[speaker_id] = "Child"
            else:
                classifications[speaker_id] = "Woman"
    
    return classifications

def classify_speakers_clustering(speaker_features):
    """
    Uses K-means clustering to separate speakers into 3 categories,
    then assigns labels based on cluster centers.
    """
    # Extract features for clustering
    feature_matrix = []
    speaker_ids = []
    
    for speaker_id, features in speaker_features.items():
        # Create a feature vector combining the most important features
        feature_vector = []
        
        # Add fundamental frequency (normalized)
        if 'f0_median' in features:
            feature_vector.append(features['f0_median'] / 400)  # Normalize to roughly 0-1 range
        else:
            feature_vector.append(0.5)  # Fallback value
            
        # Add spectral centroid (normalized)
        if 'spectral_centroid' in features:
            feature_vector.append(features['spectral_centroid'] / 5000)
        else:
            feature_vector.append(0.5)
        
        # Add first few MFCCs if available (already normalized)
        if 'mfccs' in features and len(features['mfccs']) > 0:
            # Use first 3 MFCCs
            for i in range(min(3, len(features['mfccs']))):
                feature_vector.append(features['mfccs'][i] / 20)
        
        feature_matrix.append(feature_vector)
        speaker_ids.append(speaker_id)
    
    # Need at least 3 speakers for clustering
    if len(feature_matrix) < 3:
        raise ValueError("Not enough speakers for clustering")
    
    # Run K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=0).fit(feature_matrix)
    
    # Determine which cluster corresponds to which speaker type
    cluster_centers = kmeans.cluster_centers_
    
    # Typically, the cluster with lowest F0 (first feature) is men
    # The cluster with highest F0 is children
    # The middle one is women
    f0_values = [center[0] for center in cluster_centers]
    sorted_indices = np.argsort(f0_values)
    
    cluster_labels = ["Unknown", "Unknown", "Unknown"]
    cluster_labels[sorted_indices[0]] = "Man"
    cluster_labels[sorted_indices[1]] = "Woman"
    cluster_labels[sorted_indices[2]] = "Child"
    
    # Assign labels to speakers
    classifications = {}
    for i, speaker_id in enumerate(speaker_ids):
        cluster = kmeans.labels_[i]
        classifications[speaker_id] = cluster_labels[cluster]
    
    print(f"Clustering results: Men={sorted_indices[0]}, Women={sorted_indices[1]}, Children={sorted_indices[2]}")
    return classifications

def visualize_speaker_features(speaker_features, classifications):
    """
    Creates a scatter plot of speakers based on their features.
    """
    if len(speaker_features) < 2:
        print("Not enough speakers to visualize")
        return
        
    plt.figure(figsize=(10, 6))
    
    # Extract x and y coordinates for the plot
    x_values = []
    y_values = []
    labels = []
    colors = []
    
    color_map = {"Man": "blue", "Woman": "red", "Child": "green", "Unknown": "gray"}
    
    for speaker_id, features in speaker_features.items():
        if 'f0_median' in features and 'spectral_centroid' in features:
            x_values.append(features['f0_median'])
            y_values.append(features['spectral_centroid'])
            
            speaker_type = classifications.get(speaker_id, "Unknown")
            labels.append(f"{speaker_id}: {speaker_type}")
            colors.append(color_map.get(speaker_type, "gray"))
    
    plt.scatter(x_values, y_values, c=colors)
    
    # Add labels to each point
    for i, label in enumerate(labels):
        plt.annotate(label, (x_values[i], y_values[i]), fontsize=9)
    
    plt.xlabel("Fundamental Frequency (Hz)")
    plt.ylabel("Spectral Centroid")
    plt.title("Speaker Acoustic Features")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add some typical reference regions
    plt.axvline(x=150, color='gray', linestyle='--', alpha=0.5, label="Men/Women boundary")
    plt.axvline(x=260, color='gray', linestyle='--', alpha=0.5, label="Women/Children boundary")
    
    plt.legend()
    plt.tight_layout()
    plt.savefig("speaker_features.png")
    print("Speaker feature visualization saved to speaker_features.png")

def align_speakers_to_transcription(diarization, transcription_segments, speaker_classifications, include_timestamps=True):
    """
    Matches each transcription segment to the speaker from the diarization.
    Uses the pre-computed speaker classifications.
    Returns a list of strings like: "[00:01:23 - 00:01:45] [Man] text".
    
    Parameters:
    - diarization: pyannote diarization annotation
    - transcription_segments: list of transcription segments from whisper
    - speaker_classifications: dictionary mapping speaker IDs to classifications
    - include_timestamps: whether to include timestamps in the output
    """
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
    
    return labeled_segments

def format_timestamp(seconds):
    """
    Formats seconds as HH:MM:SS
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def detect_child_conversation(transcription):
    """
    Analyzes transcription content to identify linguistic markers suggesting a child is present.
    Returns True if child-related content is detected.
    """
    # Child-related keywords and phrases
    child_indicators = [
        "school", "homework", "play", "toys", "mommy", "daddy", 
        "mom", "dad", "teacher", "playground", "games",
        "being young", "when i grow up", "when you grow up",
        "cowboys and indians", "cry", "hugs", "babies", "diaper",
        "bullying", "friend at school", "my friend"
    ]
    
    # Join all text and check for indicators
    full_text = " ".join([seg["text"].lower() for seg in transcription])
    matches = [word for word in child_indicators if word in full_text]
    
    if matches:
        print(f"Child conversation detected based on content: {', '.join(matches)}")
        return True
    return False

def process_audio_file(audio_path, output_txt="transcript.txt", force_child=False, include_timestamps=True):
    """
    Process an entire audio file:
    1. Diarize to find speakers
    2. Transcribe the speech
    3. Extract acoustic features for each speaker
    4. Classify speakers by demographic
    5. Align transcription with speaker classifications
    6. Save the transcript to a file
    
    Parameters:
    - audio_path: Path to the audio file
    - output_txt: Output transcript path
    - force_child: Force classification to include a child (for known child interviews)
    - include_timestamps: Whether to include timestamps in the transcript
    """
    # Initialize models
    print("Initializing models...")
    whisper_model = WhisperModel(
        "medium",
        device="cuda" if torch.cuda.is_available() else "cpu",
        compute_type="float16" if torch.cuda.is_available() else "float32"
    )
    
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.getenv("HF_API_KEY", "YOUR_HF_API_KEY")
    )
    if torch.cuda.is_available():
        diarization_pipeline.to(torch.device("cuda"))
    
    # 1. Diarize the audio
    diarization = diarize_audio(diarization_pipeline, audio_path)
    
    # 2. Transcribe the audio
    transcription = transcribe_audio(whisper_model, audio_path)
    
    # Check if this is likely a conversation involving a child
    child_present = force_child or detect_child_conversation(transcription)
    context_hints = {'child_present': child_present}
    
    if force_child:
        print("User indicated this recording contains a child")
    
    # 3. Extract acoustic features for each speaker
    speaker_features = extract_and_analyze_speaker_features(audio_path, diarization)
    
    # 4. Classify speakers
    speaker_classifications = classify_speakers(speaker_features, context_hints=context_hints)
    print("\nSpeaker Classifications:")
    for speaker, label in speaker_classifications.items():
        print(f"Speaker {speaker} â†’ {label}")
    
    # If force_child and no child was detected, reclassify highest-pitched non-man speaker as Child
    if force_child and "Child" not in speaker_classifications.values():
        print("Ensuring child classification as requested...")
        # Find non-man speaker with highest pitch
        highest_f0 = 0
        highest_speaker = None
        
        for speaker_id, features in speaker_features.items():
            if (speaker_classifications[speaker_id] != "Man" and 
                'f0_median' in features and 
                features['f0_median'] > highest_f0):
                highest_f0 = features['f0_median']
                highest_speaker = speaker_id
        
        if highest_speaker:
            speaker_classifications[highest_speaker] = "Child"
            print(f"Reclassified Speaker {highest_speaker} as Child (highest non-male F0)")
    
    # Visualize speaker features
    try:
        from sklearn.cluster import KMeans
        visualize_speaker_features(speaker_features, speaker_classifications)
    except ImportError:
        print("Skipping visualization - sklearn not installed")
    
    # 5. Align transcription with speaker labels
    labeled_transcript = align_speakers_to_transcription(
        diarization, transcription, speaker_classifications, include_timestamps
    )
    
    # 6. Save the transcript
    transcript_text = "\n".join(labeled_transcript)
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(transcript_text)
    
    print(f"\nTranscript with speaker labels saved to {output_txt}")
    print("\nTranscript Preview:")
    print(NEON_GREEN + "\n".join(labeled_transcript[:10]) + RESET_COLOR)
    
    return transcript_text

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Process audio file with speaker classification")
    parser.add_argument("audio_file", help="Path to the audio file to process")
    parser.add_argument("-o", "--output", default="transcript.txt", help="Output file for the transcript")
    parser.add_argument("--child", action="store_true", help="Force child classification for at least one speaker")
    parser.add_argument("--advanced", action="store_true", help="Use advanced classification methods")
    parser.add_argument("--no-timestamps", action="store_true", help="Exclude timestamps from the transcript")
    
    args = parser.parse_args()
    
    process_audio_file(
        args.audio_file, 
        args.output, 
        force_child=args.child,
        include_timestamps=not args.no_timestamps
    )
