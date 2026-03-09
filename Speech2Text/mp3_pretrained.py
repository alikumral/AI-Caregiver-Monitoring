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
from collections import Counter, defaultdict
import traceback
import platform
import sys

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

# Check if running on Windows
IS_WINDOWS = platform.system() == "Windows"

# SpeechBrain configuration (optional)
HAVE_SPEECHBRAIN = False
try:
    import torchaudio
    from speechbrain.inference import EncoderClassifier
    HAVE_SPEECHBRAIN = True
    print("SpeechBrain imported successfully")
except ImportError:
    print("SpeechBrain not available. Install with: pip install speechbrain")

# Resemblyzer configuration (optional)
HAVE_RESEMBLYZER = False
try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    HAVE_RESEMBLYZER = True
    print("Resemblyzer imported successfully")
except ImportError:
    print("Resemblyzer not available. Install with: pip install resemblyzer")

# WavLM configuration (optional)
HAVE_WAVLM = False
try:
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
    HAVE_WAVLM = True
    print("WavLM imported successfully")
except ImportError:
    print("WavLM not available. Install with: pip install transformers")

# PyAudioAnalysis configuration (optional)
HAVE_PYAUDIOANALYSIS = False
try:
    # Try multiple import paths to handle different installations
    try:
        from pyAudioAnalysis import audioTrainTest as aT
        HAVE_PYAUDIOANALYSIS = True
        print("PyAudioAnalysis imported successfully")
    except ImportError:
        # Try lowercase import (package name might be lowercase)
        from pyaudioanalysis import audioTrainTest as aT
        HAVE_PYAUDIOANALYSIS = True
        print("PyAudioAnalysis imported successfully (lowercase)")
except ImportError:
    print("PyAudioAnalysis not available. Install with: pip install pyAudioAnalysis")

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
            
        # Extract features for each valid segment
        f0_values = []
        spectral_centroids = []
        spectral_rolloffs = []
        spectral_bandwidths = []
        zero_crossing_rates = []
        mfccs_list = []
        jitters = []
        
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
            
            # Save the segment for manual inspection
            output_path = f"speaker_samples/speaker_{speaker_id}_segment_{i+1}.wav"
            sf.write(output_path, segment_audio, sr)
            
            # Fundamental frequency using YIN algorithm
            try:
                f0, voiced_flag, _ = librosa.pyin(segment_audio, 
                                                fmin=60, 
                                                fmax=600,  # Extended range for children
                                                sr=sr)
                valid_f0 = f0[~np.isnan(f0)]
                if len(valid_f0) > 0:
                    f0_values.append(np.median(valid_f0))
                    
                    # Calculate jitter (variation in fundamental frequency)
                    if len(valid_f0) > 1:
                        jitter = np.mean(np.abs(np.diff(valid_f0))) / np.mean(valid_f0)
                        jitters.append(jitter)
            except Exception as e:
                print(f"Error extracting F0 for speaker {speaker_id}, segment {i}: {e}")
            
            # Spectral features
            try:
                # Spectral centroid (center of mass of the spectrum)
                centroid = np.mean(librosa.feature.spectral_centroid(y=segment_audio, sr=sr))
                spectral_centroids.append(centroid)
                
                # Spectral rolloff (frequency below which X% of energy is contained)
                rolloff = np.mean(librosa.feature.spectral_rolloff(y=segment_audio, sr=sr))
                spectral_rolloffs.append(rolloff)
                
                # Spectral bandwidth (variance around the centroid)
                bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=segment_audio, sr=sr))
                spectral_bandwidths.append(bandwidth)
                
                # Zero crossing rate (higher for more noisy signals)
                zcr = np.mean(librosa.feature.zero_crossing_rate(segment_audio))
                zero_crossing_rates.append(zcr)
                
                # MFCC features (important for speaker identification)
                mfccs = librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=13)
                mfccs_list.append(np.mean(mfccs, axis=1))
            except Exception as e:
                print(f"Error extracting spectral features for speaker {speaker_id}, segment {i}: {e}")
        
        # Compute average features for this speaker
        features = {}
        if f0_values:
            features['f0_median'] = np.median(f0_values)
            features['f0_mean'] = np.mean(f0_values)
            features['f0_std'] = np.std(f0_values) if len(f0_values) > 1 else 0
        if spectral_centroids:
            features['spectral_centroid'] = np.mean(spectral_centroids)
        if spectral_rolloffs:
            features['spectral_rolloff'] = np.mean(spectral_rolloffs)
        if spectral_bandwidths:
            features['spectral_bandwidth'] = np.mean(spectral_bandwidths)
        if zero_crossing_rates:
            features['zero_crossing_rate'] = np.mean(zero_crossing_rates)
        if jitters:
            features['jitter'] = np.mean(jitters)
        if mfccs_list:
            # Save average MFCCs
            avg_mfccs = np.mean(mfccs_list, axis=0)
            for i, mfcc in enumerate(avg_mfccs):
                features[f'mfcc_{i+1}'] = mfcc
        
        if features:
            speaker_features[speaker_id] = features
            print(f"Speaker {speaker_id} features extracted successfully")
        
        # Store concatenated audio for this speaker (for use with other models)
        if all_audio_segments:
            speaker_audio[speaker_id] = np.concatenate(all_audio_segments)
    
    print(f"Feature extraction completed in {time.time() - start_time:.2f} seconds")
    return speaker_features, speaker_audio

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
                                for k, v in features.items() if not k.startswith('mfcc_')])
        print(f"Speaker {speaker_id}: {feature_str}")
    
    # Classifications and confidence scores
    classifications = {}
    confidence_scores = {}
    
    for speaker_id, features in speaker_features.items():
        if 'f0_median' in features:
            f0 = features['f0_median']
            centroid = features.get('spectral_centroid', 0)
            bandwidth = features.get('spectral_bandwidth', 0)
            zcr = features.get('zero_crossing_rate', 0)
            jitter = features.get('jitter', 0)
            
            # Calculate scores for each class
            man_score = 0.0
            woman_score = 0.0
            child_score = 0.0
            
            # Man classification logic (improved)
            if f0 < 145:  # Strong indicator
                man_score = 0.95
            elif f0 < 165:  # Likely
                man_score = 0.85 - (f0 - 145) * 0.01
            else:  # Unlikely
                man_score = max(0.05, 0.30 - (f0 - 165) * 0.01)
            
            # Child classification logic (improved)
            if f0 > 280 and (centroid > 3500 or zcr > 0.1):  # Strong indicator
                child_score = 0.95
            elif f0 > 245 or (f0 > 230 and jitter > 0.03):  # Likely
                child_score = 0.70 + (f0 - 245) * 0.01
                if jitter > 0.03:  # Children often have more pitch variation
                    child_score += 0.1
            else:  # Unlikely
                child_score = max(0.05, 0.30 - (245 - f0) * 0.01)
            
            # Woman classification logic (improved)
            if 165 < f0 < 245 and jitter < 0.03:  # Ideal range
                woman_score = 0.90 - abs(205 - f0) * 0.004
                # Adjust for bandwidth (women typically have more controlled voice properties)
                if bandwidth < 2000:
                    woman_score += 0.05
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

def classify_with_speechbrain(speaker_audio_dict):
    """
    Classify speakers using SpeechBrain's speaker embeddings and reference templates.
    """
    if not HAVE_SPEECHBRAIN:
        print("SpeechBrain not available. Skipping this classification method.")
        return {}, {}
    
    print("\nClassifying speakers with SpeechBrain...")
    classifications = {}
    confidence_scores = {}
    
    try:
        # On Windows, use a different strategy to avoid symlink issues
        savedir = "pretrained_models/spkrec-ecapa-voxceleb"
        
        if IS_WINDOWS:
            print("Windows detected: Using alternative SpeechBrain loading method to avoid symlink issues")
            # Set environment variable to avoid symlink
            os.environ['SPEECHBRAIN_SYMLINK'] = 'False'
            
            try:
                # Initialize SpeechBrain model with copy strategy
                classifier = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb", 
                    savedir=savedir,
                    run_opts={"device": "cpu"}
                )
            except OSError as e:
                if "privilege is not held" in str(e):
                    print("Windows permission error: Please run the script as Administrator to use SpeechBrain")
                    print("Skipping SpeechBrain classification due to permission error")
                    return {}, {}
                raise
        else:
            # Non-Windows systems can use default configuration
            classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb", 
                savedir=savedir
            )
        
        # Reference embeddings - these are placeholder values and should be replaced
        # with actual embeddings from known examples of each category
        # Ideally, you would average multiple examples of each category
        reference_embeddings = {}
        
        # First, try to load pre-saved reference embeddings if they exist
        if os.path.exists("reference_embeddings.pt"):
            try:
                reference_embeddings = torch.load("reference_embeddings.pt")
                print("Loaded reference embeddings from file")
            except Exception as e:
                print(f"Error loading reference embeddings: {e}")
        
        # If we don't have reference embeddings, we need to create them
        # This is a placeholder - in a real implementation, you would use
        # curated examples of men, women, and children's voices
        if not reference_embeddings:
            print("No reference embeddings found. Using temporary embeddings.")
            # These are synthetic values for demonstration
            random_state = np.random.RandomState(42)  # For reproducibility
            reference_embeddings = {
                "Man": torch.tensor(random_state.normal(0, 1, 192), dtype=torch.float32),
                "Woman": torch.tensor(random_state.normal(0, 1, 192), dtype=torch.float32),
                "Child": torch.tensor(random_state.normal(0, 1, 192), dtype=torch.float32)
            }
        
        for speaker_id, audio in speaker_audio_dict.items():
            # Convert to tensor
            waveform = torch.FloatTensor(audio).unsqueeze(0)
            
            # Extract embedding
            embedding = classifier.encode_batch(waveform).squeeze()
            
            # Compare with reference embeddings using cosine similarity
            similarities = {}
            for category, ref_emb in reference_embeddings.items():
                # Normalize embeddings
                ref_norm = torch.norm(ref_emb)
                emb_norm = torch.norm(embedding)
                # Calculate cosine similarity
                if ref_norm > 0 and emb_norm > 0:
                    sim = torch.dot(embedding, ref_emb) / (ref_norm * emb_norm)
                    similarities[category] = sim.item()
                else:
                    similarities[category] = 0.0
            
            # Get highest similarity
            if similarities:
                best_match = max(similarities.items(), key=lambda x: x[1])
                classifications[speaker_id] = best_match[0]
                
                # Normalize to confidence scores
                total = sum(similarities.values())
                confidence_scores[speaker_id] = {k: v/total for k, v in similarities.items()}
                
                print(f"SpeechBrain classified speaker {speaker_id} as {best_match[0]} with similarity {best_match[1]:.2f}")
            
    except Exception as e:
        print(f"Error in SpeechBrain classification: {e}")
        traceback.print_exc()
    
    return classifications, confidence_scores

def classify_with_resemblyzer(speaker_audio_dict):
    """
    Classify speakers using Resemblyzer's voice embeddings.
    """
    if not HAVE_RESEMBLYZER:
        print("Resemblyzer not available. Skipping this classification method.")
        return {}, {}
    
    print("\nClassifying speakers with Resemblyzer...")
    classifications = {}
    confidence_scores = {}
    
    try:
        # Initialize Resemblyzer
        encoder = VoiceEncoder()
        
        # Reference embeddings - these are placeholder values
        # In a real implementation, you would use actual reference voices
        reference_embeddings = {}
        
        # Try to load pre-saved reference embeddings
        ref_file = "resemblyzer_references.npz"
        if os.path.exists(ref_file):
            try:
                data = np.load(ref_file)
                reference_embeddings = {
                    "Man": data["man"],
                    "Woman": data["woman"],
                    "Child": data["child"]
                }
                print("Loaded Resemblyzer reference embeddings")
            except Exception as e:
                print(f"Error loading Resemblyzer references: {e}")
        
        # If we don't have references, create synthetic ones
        # This is just for demonstration
        if not reference_embeddings:
            print("No Resemblyzer references found. Using synthetic references.")
            random_state = np.random.RandomState(42)
            reference_embeddings = {
                "Man": random_state.normal(0, 1, 256),  # Resemblyzer embeddings are 256-dim
                "Woman": random_state.normal(0, 1, 256),
                "Child": random_state.normal(0, 1, 256)
            }
        
        for speaker_id, audio in speaker_audio_dict.items():
            # Save temporarily to process with Resemblyzer
            temp_file = f"temp_{speaker_id}.wav"
            sf.write(temp_file, audio, 16000)
            
            # Process with Resemblyzer
            try:
                wav = preprocess_wav(temp_file)
                embedding = encoder.embed_utterance(wav)
                
                # Compare with reference embeddings
                similarities = {}
                for category, ref_embed in reference_embeddings.items():
                    # Normalize for cosine similarity
                    ref_norm = np.linalg.norm(ref_embed)
                    emb_norm = np.linalg.norm(embedding)
                    # Calculate similarity
                    if ref_norm > 0 and emb_norm > 0:
                        sim = np.dot(embedding, ref_embed) / (ref_norm * emb_norm)
                        similarities[category] = sim
                    else:
                        similarities[category] = 0.0
                
                # Get best match
                if similarities:
                    best_match = max(similarities.items(), key=lambda x: x[1])
                    classifications[speaker_id] = best_match[0]
                    
                    # Normalize to confidence scores
                    total = sum(similarities.values())
                    confidence_scores[speaker_id] = {k: v/total for k, v in similarities.items()}
                    
                    print(f"Resemblyzer classified speaker {speaker_id} as {best_match[0]} with similarity {best_match[1]:.2f}")
                
                # Clean up temporary file
                os.remove(temp_file)
                
            except Exception as e:
                print(f"Error processing speaker {speaker_id} with Resemblyzer: {e}")
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                
    except Exception as e:
        print(f"Error in Resemblyzer classification: {e}")
        traceback.print_exc()
    
    return classifications, confidence_scores

def classify_with_wavlm(speaker_audio_dict):
    """
    Classify speakers using WavLM from Hugging Face.
    """
    if not HAVE_WAVLM:
        print("WavLM not available. Skipping this classification method.")
        return {}, {}
    
    print("\nClassifying speakers with WavLM...")
    classifications = {}
    confidence_scores = {}
    
    try:
        # Initialize WavLM
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv")
        
        # Reference embeddings - these are placeholders
        reference_embeddings = {}
        
        # Try to load pre-saved reference embeddings
        ref_file = "wavlm_references.pt"
        if os.path.exists(ref_file):
            try:
                reference_embeddings = torch.load(ref_file)
                print("Loaded WavLM reference embeddings")
            except Exception as e:
                print(f"Error loading WavLM references: {e}")
        
        # If we don't have references, create synthetic ones
        if not reference_embeddings:
            print("No WavLM references found. Using synthetic references.")
            random_state = np.random.RandomState(42)
            embed_dim = model.config.xvector_output_dim  # Usually 512
            reference_embeddings = {
                "Man": torch.tensor(random_state.normal(0, 1, embed_dim), dtype=torch.float32),
                "Woman": torch.tensor(random_state.normal(0, 1, embed_dim), dtype=torch.float32),
                "Child": torch.tensor(random_state.normal(0, 1, embed_dim), dtype=torch.float32)
            }
        
        for speaker_id, audio in speaker_audio_dict.items():
            # Process with WavLM - need to ensure proper format
            try:
                # Resample to 16kHz if needed
                if len(audio) < 16000:
                    print(f"Audio segment for speaker {speaker_id} is too short. Skipping.")
                    continue
                
                # Extract features
                inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
                
                # Get embedding
                with torch.no_grad():
                    embedding = model(**inputs).embeddings.squeeze()
                
                # Compare with reference embeddings
                similarities = {}
                for category, ref_emb in reference_embeddings.items():
                    # Normalize for cosine similarity
                    ref_norm = torch.norm(ref_emb)
                    emb_norm = torch.norm(embedding)
                    # Calculate similarity
                    if ref_norm > 0 and emb_norm > 0:
                        sim = torch.dot(embedding, ref_emb) / (ref_norm * emb_norm)
                        similarities[category] = sim.item()
                    else:
                        similarities[category] = 0.0
                
                # Get best match
                if similarities:
                    best_match = max(similarities.items(), key=lambda x: x[1])
                    classifications[speaker_id] = best_match[0]
                    
                    # Normalize to confidence scores
                    total = sum(similarities.values())
                    confidence_scores[speaker_id] = {k: v/total for k, v in similarities.items()}
                    
                    print(f"WavLM classified speaker {speaker_id} as {best_match[0]} with similarity {best_match[1]:.2f}")
            
            except Exception as e:
                print(f"Error processing speaker {speaker_id} with WavLM: {e}")
                
    except Exception as e:
        print(f"Error in WavLM classification: {e}")
        traceback.print_exc()
    
    return classifications, confidence_scores

def classify_with_pyaudioanalysis(speaker_audio_dict):
    """
    Classify speakers using PyAudioAnalysis.
    """
    if not HAVE_PYAUDIOANALYSIS:
        print("PyAudioAnalysis not available. Skipping this classification method.")
        return {}, {}
    
    print("\nClassifying speakers with PyAudioAnalysis...")
    classifications = {}
    confidence_scores = {}
    
    try:
        # PyAudioAnalysis requires files on disk for analysis
        # Save each speaker segment temporarily
        temp_files = {}
        for speaker_id, audio in speaker_audio_dict.items():
            temp_file = f"temp_paa_{speaker_id}.wav"
            sf.write(temp_file, audio, 16000)
            temp_files[speaker_id] = temp_file
        
        # Define reference models - we would need pre-trained models for each category
        # In a real implementation, you would have these models on disk
        # Here we're just using a simple heuristic based on features
        
        for speaker_id, temp_file in temp_files.items():
            try:
                # Extract features using PyAudioAnalysis
                features, _ = aT.extractFeatures([temp_file], 1.0, 1.0, 0.05, 0.05)
                mean_features = np.mean(features, axis=1)
                
                # Simple decision logic based on selected features
                # This is a very basic approach - in a real implementation, you would use
                # a proper classifier trained on representative data
                
                # Features of interest: 
                # - Energy entropy (index 1) - higher for children
                # - Spectral centroid (index 3) - higher for women and children
                # - Spectral entropy (index 8) - varies by gender
                
                energy_entropy = mean_features[1]
                spectral_centroid = mean_features[3]
                spectral_entropy = mean_features[8]
                
                # Simple heuristic classification
                man_score = 1.0
                woman_score = 1.0
                child_score = 1.0
                
                # Adjust scores based on features
                if spectral_centroid > 2000:  # Higher centroids more common in women/children
                    man_score *= 0.5
                else:
                    woman_score *= 0.7
                    child_score *= 0.7
                
                if energy_entropy > 0.7:  # Higher energy entropy for children
                    child_score *= 1.5
                    man_score *= 0.7
                
                if spectral_entropy > 0.8:  # Different spectral entropy patterns
                    woman_score *= 1.2
                
                # Normalize scores
                total = man_score + woman_score + child_score
                man_score /= total
                woman_score /= total
                child_score /= total
                
                # Assign classification
                scores = {
                    "Man": man_score,
                    "Woman": woman_score,
                    "Child": child_score
                }
                best_class = max(scores.items(), key=lambda x: x[1])
                
                classifications[speaker_id] = best_class[0]
                confidence_scores[speaker_id] = scores
                
                print(f"PyAudioAnalysis classified speaker {speaker_id} as {best_class[0]} with confidence {best_class[1]:.2f}")
                
                # Clean up temporary file
                os.remove(temp_file)
                
            except Exception as e:
                print(f"Error processing speaker {speaker_id} with PyAudioAnalysis: {e}")
                if os.path.exists(temp_file):
                    os.remove(temp_file)
    
    except Exception as e:
        print(f"Error in PyAudioAnalysis classification: {e}")
        traceback.print_exc()
        
        # Clean up any remaining temp files
        for temp_file in temp_files.values():
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    return classifications, confidence_scores

def combine_all_classifications(classification_results):
    """
    Combine classifications from all methods with weighted voting.
    
    classification_results: dict of {method_name: (classifications, confidence_scores, weight)}
    """
    # Method weights (adjust these based on which methods you trust more)
    default_weights = {
        "acoustic": 1.0,
        "huggingface": 2.0,  # LLM has higher weight as requested
        "speechbrain": 1.5,
        "resemblyzer": 1.5,
        "wavlm": 1.8,
        "pyaudioanalysis": 1.2
    }
    
    # Get all speaker IDs across all methods
    all_speaker_ids = set()
    for method, (classifications, _, _) in classification_results.items():
        all_speaker_ids.update(classifications.keys())
    
    # Perform weighted voting for each speaker
    final_classifications = {}
    final_confidence = {}
    
    for speaker_id in all_speaker_ids:
        # Collect votes from all methods
        votes = defaultdict(float)
        total_weight = 0.0
        
        for method, (classifications, confidence, weight) in classification_results.items():
            if speaker_id in classifications:
                label = classifications[speaker_id]
                conf = confidence[speaker_id].get(label, 0.5)
                method_weight = weight if weight is not None else default_weights.get(method, 1.0)
                
                votes[label] += conf * method_weight
                total_weight += method_weight
        
        if votes and total_weight > 0:
            # Get label with highest weighted vote
            best_label = max(votes.items(), key=lambda x: x[1])[0]
            final_classifications[speaker_id] = best_label
            
            # Normalize confidence
            final_confidence[speaker_id] = votes[best_label] / total_weight
    
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

def visualize_classifications(classification_results, final_classifications):
    """
    Create a visualization of classifications from all methods.
    """
    # Get all unique speaker IDs
    all_speakers = set()
    for method, (classifications, _, _) in classification_results.items():
        all_speakers.update(classifications.keys())
    all_speakers = sorted(list(all_speakers))
    
    if not all_speakers:
        print("No speakers to visualize")
        return
    
    # Set up the plot
    plt.figure(figsize=(15, 10))
    
    # Create a grid of subplots
    n_methods = len(classification_results)
    if n_methods <= 3:
        rows, cols = 1, n_methods
    else:
        rows = (n_methods + 1) // 2  # +1 to include final classification
        cols = 2
    
    # Track which subplot we're on
    subplot_idx = 1
    
    # Store all confidence data for summary
    all_confidence_data = {}
    
    # Plot each method's results
    for method, (classifications, confidence_scores, _) in classification_results.items():
        plt.subplot(rows, cols, subplot_idx)
        subplot_idx += 1
        
        # Prepare data for plotting
        speakers = []
        man_scores = []
        woman_scores = []
        child_scores = []
        chosen_classes = []
        
        for speaker in all_speakers:
            if speaker in classifications:
                speakers.append(speaker)
                label = classifications[speaker]
                chosen_classes.append(label)
                
                # Get confidence scores for this speaker
                scores = confidence_scores.get(speaker, {})
                man_scores.append(scores.get("Man", 0))
                woman_scores.append(scores.get("Woman", 0))
                child_scores.append(scores.get("Child", 0))
                
                # Store for summary
                if speaker not in all_confidence_data:
                    all_confidence_data[speaker] = {}
                all_confidence_data[speaker][method] = {
                    "classification": label,
                    "confidence": scores
                }
        
        # Create the stacked bar chart
        bar_width = 0.6
        indices = np.arange(len(speakers))
        
        p1 = plt.bar(indices, man_scores, bar_width, color='blue', alpha=0.7, label='Man')
        p2 = plt.bar(indices, woman_scores, bar_width, bottom=man_scores, color='red', alpha=0.7, label='Woman')
        
        # Calculate bottom for child scores
        bottoms = [m+w for m, w in zip(man_scores, woman_scores)]
        p3 = plt.bar(indices, child_scores, bar_width, bottom=bottoms, color='green', alpha=0.7, label='Child')
        
        # Mark the chosen classification
        for i, cls in enumerate(chosen_classes):
            color = 'blue' if cls == 'Man' else 'red' if cls == 'Woman' else 'green'
            plt.plot(indices[i], 1.05, marker='*', color=color, markersize=10)
        
        # Set up the axes
        plt.title(f"{method.capitalize()} Classification")
        plt.xlabel('Speaker ID')
        plt.ylabel('Confidence Score')
        plt.xticks(indices, speakers, rotation=45)
        plt.ylim(0, 1.2)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
    
    # Final classification summary
    if subplot_idx <= rows * cols:
        plt.subplot(rows, cols, subplot_idx)
        
        # Prepare data
        speakers = []
        final_labels = []
        
        for speaker in all_speakers:
            if speaker in final_classifications:
                speakers.append(speaker)
                final_labels.append(final_classifications[speaker])
        
        # Set up bar colors
        colors = ['blue' if label == 'Man' else 'red' if label == 'Woman' else 'green' for label in final_labels]
        
        # Create horizontal bar chart for final classification
        indices = np.arange(len(speakers))
        plt.barh(indices, [1] * len(speakers), color=colors, alpha=0.7)
        
        # Add text labels
        for i, (speaker, label) in enumerate(zip(speakers, final_labels)):
            plt.text(0.5, i, f"{speaker}: {label}", ha='center', va='center', color='white', fontweight='bold')
        
        plt.title("Final Classification")
        plt.yticks(indices, speakers)
        plt.xticks([])
        plt.tight_layout()
    
    plt.savefig("speaker_classification_comparison.png", dpi=150)
    print("Classification comparison visualization saved to speaker_classification_comparison.png")
    
    return all_confidence_data

def create_detailed_report(output_txt, classification_results, final_classes, final_confidence):
    """
    Create a detailed report of all classification methods and their results.
    """
    report_path = output_txt.replace(".txt", "_detailed_report.txt")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("SPEAKER CLASSIFICATION DETAILED REPORT\n")
        f.write("====================================\n\n")
        
        # Get all unique speaker IDs
        all_speakers = set()
        for method, (classifications, _, _) in classification_results.items():
            all_speakers.update(classifications.keys())
        all_speakers = sorted(list(all_speakers))
        
        # Write a section for each speaker
        for speaker in all_speakers:
            f.write(f"SPEAKER: {speaker}\n")
            f.write("-" * 40 + "\n")
            
            # Show final classification
            final_class = final_classes.get(speaker, "Unknown")
            final_conf = final_confidence.get(speaker, 0.0)
            f.write(f"FINAL CLASSIFICATION: {final_class} (confidence: {final_conf:.2f})\n\n")
            
            # Show each method's classification and confidence
            f.write("Individual Model Results:\n")
            
            for method, (classifications, confidence_scores, weight) in classification_results.items():
                if speaker in classifications:
                    label = classifications[speaker]
                    scores = confidence_scores.get(speaker, {})
                    
                    # Format confidence scores
                    scores_str = ", ".join([f"{k}: {v:.2f}" for k, v in scores.items()])
                    
                    f.write(f"  {method.capitalize()}: {label} ({scores_str})\n")
                else:
                    f.write(f"  {method.capitalize()}: No classification\n")
            
            f.write("\n\n")
        
        # Add a summary section
        f.write("CLASSIFICATION SUMMARY\n")
        f.write("=====================\n\n")
        
        # Table header
        f.write(f"{'Speaker':<10} | {'Final':<10} | {'Acoustic':<10} | {'HuggingFace':<15} | {'SpeechBrain':<15} | {'Resemblyzer':<15}\n")
        f.write("-" * 75 + "\n")
        
        # Table rows
        for speaker in all_speakers:
            row = [f"{speaker:<10}"]
            
            # Final classification
            final_class = final_classes.get(speaker, "Unknown")
            row.append(f"{final_class:<10}")
            
            # Each method's classification
            for method in ["acoustic", "huggingface", "speechbrain", "resemblyzer"]:
                if method in classification_results:
                    classifications, _, _ = classification_results[method]
                    if speaker in classifications:
                        label = classifications[speaker]
                        row.append(f"{label:<15}")
                    else:
                        row.append(f"{'N/A':<15}")
                else:
                    row.append(f"{'N/A':<15}")
            
            f.write(" | ".join(row) + "\n")
        
        # Additional statistics
        f.write("\n\nCLASSIFICATION STATISTICS\n")
        f.write("========================\n\n")
        
        # Count occurrences of each speaker type
        counts = Counter(final_classes.values())
        f.write(f"Total speakers identified: {len(final_classes)}\n")
        for speaker_type in ["Man", "Woman", "Child", "Unknown"]:
            count = counts.get(speaker_type, 0)
            f.write(f"  {speaker_type}: {count}\n")
        
        # Count agreement between methods
        f.write("\nModel Agreement:\n")
        agreement_counts = {
            "all_agree": 0,
            "majority_agree": 0,
            "no_agreement": 0
        }
        
        for speaker in all_speakers:
            # Collect all classifications for this speaker
            votes = []
            for method, (classifications, _, _) in classification_results.items():
                if speaker in classifications:
                    votes.append(classifications[speaker])
            
            # Count votes for each class
            vote_counts = Counter(votes)
            
            # Check agreement
            if len(vote_counts) == 1:
                agreement_counts["all_agree"] += 1
            elif vote_counts.most_common(1)[0][1] > 1:
                agreement_counts["majority_agree"] += 1
            else:
                agreement_counts["no_agreement"] += 1
        
        f.write(f"  All models agree: {agreement_counts['all_agree']}\n")
        f.write(f"  Majority agreement: {agreement_counts['majority_agree']}\n")
        f.write(f"  No agreement: {agreement_counts['no_agreement']}\n")
    
    print(f"Detailed classification report saved to {report_path}")
    return report_path

def process_with_multiple_models(
    audio_path, 
    output_txt="transcript.txt", 
    include_timestamps=True,
    model_size="small",
    language=None,
    expected_child_count=None
):
    """
    Process audio with multiple speaker classification models and combine their results.
    """
    total_start_time = time.time()
    
    # Check if input is MP3 and convert if needed
    if audio_path.lower().endswith('.mp3'):
        print("MP3 file detected, converting to WAV for processing...")
        wav_path = convert_mp3_to_wav(audio_path)
        if wav_path is None:
            print("Error converting MP3 to WAV. Aborting.")
            return None
        audio_path = wav_path
    
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
    
    # On Windows, Hugging Face API may have issues with symlinks
    if IS_WINDOWS:
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_API_KEY
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
    
    # 3. Extract features and perform classifications with multiple methods
    
    # 3a. Acoustic features and audio segments
    acoustic_features, speaker_audio = extract_acoustic_features(audio_path, diarization)
    
    # 3b. Classification results from each method
    classification_results = {}
    
    # Acoustic classification
    acoustic_classes, acoustic_confidence = classify_speakers_acoustic(acoustic_features)
    classification_results["acoustic"] = (acoustic_classes, acoustic_confidence, 1.0)
    
    # LLM-based classification (HuggingFace)
    speaker_texts = extract_text_by_speaker(transcription, diarization)
    llm_classes, llm_confidence = classify_with_huggingface(speaker_texts)
    classification_results["huggingface"] = (llm_classes, llm_confidence, 2.0)  # Higher weight for LLM as requested
    
    # Try SpeechBrain classification if available and not on Windows (due to symlink issues)
    if HAVE_SPEECHBRAIN:
        speechbrain_classes, speechbrain_confidence = classify_with_speechbrain(speaker_audio)
        if speechbrain_classes:
            classification_results["speechbrain"] = (speechbrain_classes, speechbrain_confidence, 1.5)
    
    # Try Resemblyzer classification if available
    if HAVE_RESEMBLYZER:
        resemblyzer_classes, resemblyzer_confidence = classify_with_resemblyzer(speaker_audio)
        if resemblyzer_classes:
            classification_results["resemblyzer"] = (resemblyzer_classes, resemblyzer_confidence, 1.5)
    
    # Try WavLM classification if available
    if HAVE_WAVLM:
        wavlm_classes, wavlm_confidence = classify_with_wavlm(speaker_audio)
        if wavlm_classes:
            classification_results["wavlm"] = (wavlm_classes, wavlm_confidence, 1.8)
    
    # Try PyAudioAnalysis classification if available
    if HAVE_PYAUDIOANALYSIS:
        paa_classes, paa_confidence = classify_with_pyaudioanalysis(speaker_audio)
        if paa_classes:
            classification_results["pyaudioanalysis"] = (paa_classes, paa_confidence, 1.2)
    
    # 3c. Apply special handling for expected child count if provided
    if expected_child_count is not None and expected_child_count > 0:
        print(f"Ensuring {expected_child_count} children are classified...")
        
        # For each classification method, ensure the expected number of children
        for method, (classes, confidence, weight) in classification_results.items():
            current_child_count = sum(1 for label in classes.values() if label == "Child")
            
            if current_child_count < expected_child_count:
                # Need to reclassify some speakers - find candidates
                candidates = []
                
                for speaker_id, label in classes.items():
                    if label != "Child":
                        # Calculate a "child-likeness" score
                        child_score = confidence.get(speaker_id, {}).get("Child", 0)
                        
                        # For acoustic features, also consider pitch
                        if method == "acoustic" and speaker_id in acoustic_features:
                            f0 = acoustic_features[speaker_id].get('f0_median', 0)
                            # Higher pitch means more child-like
                            if f0 > 200:
                                child_score += (f0 - 200) / 200  # Boost score for higher pitch
                        
                        candidates.append((speaker_id, child_score))
                
                # Sort by child-likeness score (highest first)
                candidates.sort(key=lambda x: x[1], reverse=True)
                
                # Reclassify as many as needed
                for i in range(min(len(candidates), expected_child_count - current_child_count)):
                    speaker_id = candidates[i][0]
                    classes[speaker_id] = "Child"
                    
                    # Update confidence scores
                    if speaker_id in confidence:
                        # Increase Child confidence, decrease others
                        old_scores = confidence[speaker_id]
                        old_child = old_scores.get("Child", 0.1)
                        
                        # New score: boost child, reduce others
                        confidence[speaker_id] = {
                            "Child": min(0.9, old_child * 2),
                            "Man": max(0.05, old_scores.get("Man", 0.1) * 0.5),
                            "Woman": max(0.05, old_scores.get("Woman", 0.1) * 0.5)
                        }
                    
                print(f"Method {method}: Reclassified {min(len(candidates), expected_child_count - current_child_count)} speakers as Child")
    
    # 4. Combine classifications from all methods
    final_classes, final_confidence = combine_all_classifications(classification_results)
    
    # 5. Create visualizations and reports
    all_confidence_data = visualize_classifications(classification_results, final_classes)
    report_path = create_detailed_report(output_txt, classification_results, final_classes, final_confidence)
    
    # 6. Align transcription with final speaker labels
    labeled_transcript = align_speakers_to_transcription(
        diarization, transcription, final_classes, final_confidence, include_timestamps
    )
    
    # 7. Save the transcript
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
        'classification_results': classification_results,
        'final_classes': final_classes,
        'final_confidence': final_confidence,
        'report_path': report_path
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced speaker classification with multiple models")
    parser.add_argument("audio_file", help="Path to the audio file to process (MP3 or WAV)")
    parser.add_argument("-o", "--output", default="transcript.txt", help="Output file for the transcript")
    parser.add_argument("--no-timestamps", action="store_true", help="Exclude timestamps")
    parser.add_argument("--model", default="small", choices=["tiny", "base", "small", "medium"], 
                        help="Whisper model size (smaller is faster)")
    parser.add_argument("--language", help="Specify language code (e.g., 'en' for English)")
    parser.add_argument("--api-key", help="Hugging Face API key")
    parser.add_argument("--child-count", type=int, help="Expected number of children in the recording")
    parser.add_argument("--admin", action="store_true", help="Flag to indicate the script is run with administrator privileges")
    
    args = parser.parse_args()
    
    # Set API key if provided
    if args.api_key:
        HF_API_KEY = args.api_key
        HF_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}
    
    # Check if running on Windows without admin privileges
    if IS_WINDOWS and not args.admin:
        print("NOTE: On Windows, some features (like SpeechBrain) may require administrator privileges.")
        print("Consider running this script with admin rights if you encounter permission errors.")
    
    process_with_multiple_models(
        args.audio_file, 
        args.output, 
        include_timestamps=not args.no_timestamps,
        model_size=args.model,
        language=args.language,
        expected_child_count=args.child_count
    )
