#!/usr/bin/env python3

import signal
if not hasattr(signal, 'SIGKILL'):
    # On Windows, SIGKILL doesn't exist
    signal.SIGKILL = 9  # Define it as its Unix value
    
import os
import sys
import time
import json
import argparse
import traceback
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Import the actual implementation functions
# You need to place this script in the same directory as mp3_optimized.py and mp3_new.py
try:
    from mp3_optimized import process_hybrid
except ImportError:
    print("Error: Cannot import process_hybrid from mp3_optimized.py")
    print("Make sure mp3_optimized.py is in the same directory as this script")
    sys.exit(1)

try:
    from mp3_new import process_mp3_with_audeering
except ImportError:
    print("Error: Cannot import process_mp3_with_audeering from mp3_new.py")
    print("Make sure mp3_new.py is in the same directory as this script")
    sys.exit(1)

# Define colors for printing
BLUE = "\033[94m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"

class RealComparison:
    """Framework to compare the actual implementations of mp3_optimized and mp3_new."""
    
    def __init__(self, audio_dir, output_dir="comparison_results"):
        """
        Initialize the comparison framework.
        
        Args:
            audio_dir: Directory containing audio files for testing
            output_dir: Directory to save comparison results
        """
        self.audio_dir = audio_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "acoustic_results"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "deep_learning_results"), exist_ok=True)
        
        # Find audio files
        self.audio_files = self._find_audio_files()
        
        # Results storage
        self.results = {
            "acoustic": {},
            "deep_learning": {},
            "comparison": {}
        }
        
        print(f"{BLUE}{BOLD}Real Speaker Classification Comparison Framework{RESET}")
        print(f"Found {len(self.audio_files)} audio files for testing")
    
    def _find_audio_files(self):
        """Find all audio files in the specified directory."""
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
        audio_files = []
        
        for root, _, files in os.walk(self.audio_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    audio_files.append(os.path.join(root, file))
        
        return audio_files
    
    def run_acoustic_approach(self, audio_path):
        """
        Run the actual acoustic feature-based approach from mp3_optimized.py
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary containing results and metrics
        """
        file_id = os.path.basename(audio_path)
        result = {"file": file_id}
        
        try:
            # Create a unique output file for this run
            output_file = os.path.join(self.output_dir, "acoustic_results", f"{file_id}_transcript.txt")
            
            print(f"\n{BLUE}Running Acoustic Feature-Based Approach on {file_id}...{RESET}")
            
            # Measure processing time
            start_time = time.time()
            
            # Run the actual process_hybrid function from mp3_optimized.py
            transcript = process_hybrid(
                audio_path,
                output_file,
                force_child=False,
                include_timestamps=True,
                model_size="small",
                language=None
            )
            
            total_time = time.time() - start_time
            
            # Read the transcript file
            with open(output_file, 'r', encoding='utf-8') as f:
                transcript_text = f.read()
            
            # Parse the transcript to extract speaker information
            speakers = self._extract_speakers_from_transcript(transcript_text)
            
            result.update({
                "success": True,
                "transcript_file": output_file,
                "speaker_count": len(speakers),
                "speakers": speakers,
                "timing": {
                    "total": total_time
                }
            })
            
            print(f"{GREEN}Acoustic approach completed in {total_time:.2f} seconds{RESET}")
            print(f"Speaker count: {len(speakers)}")
            print(f"Speaker types: {', '.join([f'{spk_type} ({count})' for spk_type, count in speakers.items()])}")
            
        except Exception as e:
            result.update({
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            print(f"{RED}Error in acoustic approach: {str(e)}{RESET}")
            
        return result

    def run_deep_learning_approach(self, audio_path):
        """
        Run the actual deep learning-based approach from mp3_new.py
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary containing results and metrics
        """
        file_id = os.path.basename(audio_path)
        result = {"file": file_id}
        
        try:
            # Create a unique output file for this run
            output_file = os.path.join(self.output_dir, "deep_learning_results", f"{file_id}_transcript.txt")
            
            print(f"\n{BLUE}Running Deep Learning-Based Approach on {file_id}...{RESET}")
            
            # Measure processing time
            start_time = time.time()
            
            # Run the actual process_mp3_with_audeering function from mp3_new.py
            results = process_mp3_with_audeering(
                audio_path,
                output_file,
                include_timestamps=True,
                model_size="small",
                language=None
            )
            
            total_time = time.time() - start_time
            
            # Extract speaker information from the results
            if results is not None and isinstance(results, dict):
                speaker_classifications = results.get('speaker_classifications', {})
                confidence_scores = results.get('confidence_scores', {})
                detailed_results = results.get('detailed_results', {})
                
                # Count speakers by type
                speakers = {}
                for speaker_type in speaker_classifications.values():
                    if speaker_type not in speakers:
                        speakers[speaker_type] = 0
                    speakers[speaker_type] += 1
                
                result.update({
                    "success": True,
                    "transcript_file": output_file,
                    "speaker_count": len(speaker_classifications),
                    "speakers": speakers,
                    "confidence_scores": confidence_scores,
                    "detailed_results": detailed_results,
                    "timing": {
                        "total": total_time
                    }
                })
            else:
                # Read the transcript file and try to extract speaker info
                if os.path.exists(output_file):
                    with open(output_file, 'r', encoding='utf-8') as f:
                        transcript_text = f.read()
                    
                    speakers = self._extract_speakers_from_transcript(transcript_text)
                    
                    result.update({
                        "success": True,
                        "transcript_file": output_file,
                        "speaker_count": len(speakers),
                        "speakers": speakers,
                        "timing": {
                            "total": total_time
                        }
                    })
                else:
                    raise Exception("Failed to generate transcript and detailed results")
            
            print(f"{GREEN}Deep Learning approach completed in {total_time:.2f} seconds{RESET}")
            print(f"Speaker count: {result['speaker_count']}")
            print(f"Speaker types: {', '.join([f'{spk_type} ({count})' for spk_type, count in speakers.items()])}")
            
        except Exception as e:
            result.update({
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            print(f"{RED}Error in deep learning approach: {str(e)}{RESET}")
            
        return result
    
    def _extract_speakers_from_transcript(self, transcript_text):
        """
        Extract speaker information from a transcript file.
        
        Args:
            transcript_text: The transcript text
            
        Returns:
            Dictionary with speaker types as keys and counts as values
        """
        speakers = {}
        
        # Example format: [Man] Hello there
        # or [00:01:23 - 00:01:30] [Woman] Hello there
        for line in transcript_text.split('\n'):
            if ']' in line:
                # Find the last ] before the actual text
                parts = line.split(']')
                for part in parts:
                    if '[' in part:
                        # Extract text inside brackets
                        speaker_part = part.split('[')[-1].strip()
                        
                        # Remove any confidence indicators like (0.95)
                        if '(' in speaker_part:
                            speaker_part = speaker_part.split('(')[0].strip()
                        
                        # Skip timestamps
                        if ':' in speaker_part or 'unknown' in speaker_part.lower():
                            continue
                        
                        if speaker_part not in speakers:
                            speakers[speaker_part] = 0
                        speakers[speaker_part] += 1
        
        return speakers
    
    def run_comparison(self):
        """Run both approaches on all audio files and collect comparison metrics."""
        print(f"\n{YELLOW}Running comparison on {len(self.audio_files)} audio files...{RESET}")
        
        for idx, audio_file in enumerate(self.audio_files):
            print(f"\n{BLUE}Processing file {idx+1}/{len(self.audio_files)}: {os.path.basename(audio_file)}{RESET}")
            
            # Run acoustic feature-based approach (mp3_optimized.py)
            acoustic_result = self.run_acoustic_approach(audio_file)
            
            # Run deep learning-based approach (mp3_new.py)
            deep_learning_result = self.run_deep_learning_approach(audio_file)
            
            # Store results
            file_id = os.path.basename(audio_file)
            self.results["acoustic"][file_id] = acoustic_result
            self.results["deep_learning"][file_id] = deep_learning_result
            
            # Compare results
            if acoustic_result.get("success", False) and deep_learning_result.get("success", False):
                acoustic_speakers = acoustic_result.get("speakers", {})
                deep_learning_speakers = deep_learning_result.get("speakers", {})
                
                # Calculate speaker type agreement
                all_speaker_types = set(acoustic_speakers.keys()) | set(deep_learning_speakers.keys())
                total_diff = 0
                for spk_type in all_speaker_types:
                    acoustic_count = acoustic_speakers.get(spk_type, 0)
                    dl_count = deep_learning_speakers.get(spk_type, 0)
                    total_diff += abs(acoustic_count - dl_count)
                
                # Calculate a similarity score (higher is better)
                max_speakers = max(sum(acoustic_speakers.values()), sum(deep_learning_speakers.values()))
                similarity = 1.0 - (total_diff / (2 * max_speakers) if max_speakers > 0 else 0)
                
                # Calculate time ratio
                time_ratio = deep_learning_result["timing"]["total"] / acoustic_result["timing"]["total"] \
                    if acoustic_result["timing"]["total"] > 0 else float('inf')
                
                self.results["comparison"][file_id] = {
                    "acoustic_speaker_count": sum(acoustic_speakers.values()),
                    "deep_learning_speaker_count": sum(deep_learning_speakers.values()),
                    "acoustic_speakers": acoustic_speakers,
                    "deep_learning_speakers": deep_learning_speakers,
                    "similarity_score": similarity,
                    "time_ratio": time_ratio
                }
                
                # Print comparison summary
                print(f"\n{YELLOW}Comparison Summary:{RESET}")
                print(f"  • Acoustic approach: {sum(acoustic_speakers.values())} speakers identified in {acoustic_result['timing']['total']:.2f}s")
                print(f"  • Deep Learning approach: {sum(deep_learning_speakers.values())} speakers identified in {deep_learning_result['timing']['total']:.2f}s")
                print(f"  • Classification similarity: {similarity:.2f} (1.0 = identical, 0.0 = completely different)")
                print(f"  • Deep Learning is {time_ratio:.1f}x slower than Acoustic approach")
            else:
                print(f"\n{RED}Comparison failed for {file_id}:{RESET}")
                if not acoustic_result.get("success", False):
                    print(f"  • Acoustic approach failed: {acoustic_result.get('error', 'Unknown error')}")
                if not deep_learning_result.get("success", False):
                    print(f"  • Deep Learning approach failed: {deep_learning_result.get('error', 'Unknown error')}")
        
        # Save all results
        self.save_results()
        
        # Generate comparison report
        self.generate_comparison_report()
    
    def save_results(self):
        """Save all results to a JSON file."""
        results_path = os.path.join(self.output_dir, "comparison_results.json")
        
        # Create a serializable version of the results
        serializable_results = {
            "acoustic": {},
            "deep_learning": {},
            "comparison": self.results["comparison"]
        }
        
        # Remove non-serializable items from the results
        for file_id, result in self.results["acoustic"].items():
            serializable_results["acoustic"][file_id] = {
                k: v for k, v in result.items() 
                if not isinstance(v, (type, type(lambda: None)))
            }
        
        for file_id, result in self.results["deep_learning"].items():
            serializable_results["deep_learning"][file_id] = {
                k: v for k, v in result.items() 
                if not isinstance(v, (type, type(lambda: None)))
            }
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n{GREEN}Results saved to {results_path}{RESET}")
    
    def generate_comparison_report(self):
        """Generate a comprehensive comparison report with visualizations."""
        print(f"\n{YELLOW}Generating comparison report...{RESET}")
        
        # 1. Calculate aggregate statistics
        self._calculate_aggregate_stats()
        
        # 2. Generate timing comparison
        self._plot_timing_comparison()
        
        # 3. Generate classification agreement visualization
        self._plot_classification_agreement()
        
        # 4. Generate speaker type distribution
        self._plot_speaker_type_distribution()
        
        # 5. Generate detailed HTML report
        self._generate_html_report()
        
        print(f"{GREEN}Comparison report generated in {self.output_dir}{RESET}")

    def _calculate_aggregate_stats(self):
        """Calculate aggregate statistics from all comparisons."""
        if not self.results["comparison"]:
            print(f"{RED}No comparison data available for statistics{RESET}")
            return
        
        # Extract metrics
        similarity_scores = [data.get("similarity_score", 0) for data in self.results["comparison"].values()]
        time_ratios = [data.get("time_ratio", 0) for data in self.results["comparison"].values() 
                       if data.get("time_ratio", 0) != float('inf')]
        
        # Calculate statistics
        self.aggregate_stats = {
            "similarity": {
                "mean": np.mean(similarity_scores) if similarity_scores else 0,
                "min": min(similarity_scores) if similarity_scores else 0,
                "max": max(similarity_scores) if similarity_scores else 0
            },
            "time_ratio": {
                "mean": np.mean(time_ratios) if time_ratios else 0,
                "min": min(time_ratios) if time_ratios else 0,
                "max": max(time_ratios) if time_ratios else 0
            }
        }
        
        # Print summary statistics
        print(f"\n{BLUE}{BOLD}Aggregate Statistics:{RESET}")
        print(f"  • Average classification similarity: {self.aggregate_stats['similarity']['mean']:.2f}")
        print(f"  • Deep Learning is on average {self.aggregate_stats['time_ratio']['mean']:.1f}x slower")
    
    def _plot_timing_comparison(self):
        """Plot timing comparison between the two approaches."""
        plt.figure(figsize=(10, 6))
        
        # Collect timing data
        files = []
        acoustic_times = []
        deep_learning_times = []
        
        for file_id, comparison in self.results["comparison"].items():
            if self.results["acoustic"][file_id].get("success", False) and \
               self.results["deep_learning"][file_id].get("success", False):
                
                acoustic_time = self.results["acoustic"][file_id]["timing"]["total"]
                deep_learning_time = self.results["deep_learning"][file_id]["timing"]["total"]
                
                files.append(file_id)
                acoustic_times.append(acoustic_time)
                deep_learning_times.append(deep_learning_time)
        
        if not files:
            print(f"{RED}No timing data available for plotting{RESET}")
            return
        
        # Set up bar positions
        x = np.arange(len(files))
        width = 0.35
        
        # Create bars
        ax = plt.subplot(111)
        acoustic_bars = ax.bar(x - width/2, acoustic_times, width, label='Acoustic Feature-Based', color='#4299e1')
        dl_bars = ax.bar(x + width/2, deep_learning_times, width, label='Deep Learning-Based', color='#f56565')
        
        # Add labels and title
        ax.set_ylabel('Processing Time (seconds)')
        ax.set_title('Processing Time Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(files, rotation=45, ha='right')
        ax.legend()
        
        # Add the time ratio as text
        for i, file_id in enumerate(files):
            ratio = self.results["comparison"][file_id]["time_ratio"]
            plt.text(i, max(acoustic_times[i], deep_learning_times[i]) + 0.1, 
                    f"{ratio:.1f}x", 
                    ha='center', va='bottom', color='#4a5568')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'timing_comparison.png'), dpi=150)
        plt.close()
    
    def _plot_classification_agreement(self):
        """Plot classification agreement between the two approaches."""
        similarities = [data.get("similarity_score", 0) for data in self.results["comparison"].values()]
        files = list(self.results["comparison"].keys())
        
        if not files:
            print(f"{RED}No agreement data available for plotting{RESET}")
            return
        
        plt.figure(figsize=(10, 6))
        plt.bar(files, similarities, color='#68d391')
        plt.axhline(y=np.mean(similarities), color='#4a5568', linestyle='--', 
                   label=f'Average: {np.mean(similarities):.2f}')
        plt.ylabel('Similarity Score')
        plt.title('Classification Similarity Between Approaches')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'classification_similarity.png'), dpi=150)
        plt.close()
    
    def _plot_speaker_type_distribution(self):
        """Plot the distribution of speaker types identified by each approach."""
        # Collect speaker type data
        acoustic_speakers = {}
        deep_learning_speakers = {}
        
        for file_id, comparison in self.results["comparison"].items():
            # Aggregate speaker types across all files
            a_speakers = comparison.get("acoustic_speakers", {})
            dl_speakers = comparison.get("deep_learning_speakers", {})
            
            for spk_type, count in a_speakers.items():
                if spk_type not in acoustic_speakers:
                    acoustic_speakers[spk_type] = 0
                acoustic_speakers[spk_type] += count
            
            for spk_type, count in dl_speakers.items():
                if spk_type not in deep_learning_speakers:
                    deep_learning_speakers[spk_type] = 0
                deep_learning_speakers[spk_type] += count
        
        if not acoustic_speakers and not deep_learning_speakers:
            print(f"{RED}No speaker data available for plotting{RESET}")
            return
        
        # Create bar chart for speaker types
        plt.figure(figsize=(10, 6))
        
        # Combine all speaker types
        all_speaker_types = sorted(list(set(acoustic_speakers.keys()) | set(deep_learning_speakers.keys())))
        x = np.arange(len(all_speaker_types))
        width = 0.35
        
        # Get counts for each approach
        acoustic_counts = [acoustic_speakers.get(spk_type, 0) for spk_type in all_speaker_types]
        dl_counts = [deep_learning_speakers.get(spk_type, 0) for spk_type in all_speaker_types]
        
        # Create bars
        ax = plt.subplot(111)
        ax.bar(x - width/2, acoustic_counts, width, label='Acoustic Feature-Based', color='#4299e1')
        ax.bar(x + width/2, dl_counts, width, label='Deep Learning-Based', color='#f56565')
        
        # Add labels and title
        ax.set_ylabel('Number of Speakers')
        ax.set_title('Speaker Type Distribution Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(all_speaker_types)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'speaker_type_distribution.png'), dpi=150)
        plt.close()
    
    def _generate_html_report(self):
        """Generate a comprehensive HTML report with all comparison results."""
        html_path = os.path.join(self.output_dir, "comparison_report.html")
        
        # Extract necessary data
        aggregate_stats = getattr(self, 'aggregate_stats', {
            'similarity': {'mean': 0, 'min': 0, 'max': 0},
            'time_ratio': {'mean': 0, 'min': 0, 'max': 0}
        })
        
        # Create speaker counts table data
        speaker_table_data = []
        for file_id, comparison in self.results["comparison"].items():
            if self.results["acoustic"][file_id].get("success", False) and \
               self.results["deep_learning"][file_id].get("success", False):
                
                a_speakers = comparison.get("acoustic_speakers", {})
                dl_speakers = comparison.get("deep_learning_speakers", {})
                
                acoustic_counts = ", ".join([f"{spk_type}: {count}" for spk_type, count in a_speakers.items()])
                dl_counts = ", ".join([f"{spk_type}: {count}" for spk_type, count in dl_speakers.items()])
                
                speaker_table_data.append({
                    "file": file_id,
                    "acoustic_count": comparison.get("acoustic_speaker_count", 0),
                    "acoustic_types": acoustic_counts,
                    "dl_count": comparison.get("deep_learning_speaker_count", 0),
                    "dl_types": dl_counts,
                    "similarity": comparison.get("similarity_score", 0)
                })
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Speaker Classification Approaches Comparison</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    color: #333;
                }}
                h1, h2, h3 {{
                    color: #2c5282;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 20px;
                }}
                th, td {{
                    padding: 12px 15px;
                    border: 1px solid #ddd;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                    font-weight: bold;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .highlight {{
                    background-color: #fffbea;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                    border-left: 5px solid #f6ad55;
                }}
                .image-container {{
                    margin: 20px 0;
                    text-align: center;
                }}
                .image-container img {{
                    max-width: 100%;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    border-radius: 5px;
                }}
                .approaches {{
                    display: flex;
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .approach {{
                    flex: 1;
                    padding: 15px;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                }}
                .acoustic {{
                    background-color: #ebf8ff;
                    border-left: 4px solid #4299e1;
                }}
                .deep-learning {{
                    background-color: #fff5f5;
                    border-left: 4px solid #f56565;
                }}
                .pros-cons {{
                    display: flex;
                    gap: 20px;
                    margin-bottom: 20px;
                }}
                .pros, .cons {{
                    flex: 1;
                    padding: 15px;
                    border-radius: 5px;
                }}
                .pros {{
                    background-color: #f0fff4;
                    border: 1px solid #9ae6b4;
                }}
                .cons {{
                    background-color: #fff5f5;
                    border: 1px solid #feb2b2;
                }}
                .conclusion {{
                    background-color: #e6fffa;
                    padding: 20px;
                    border-radius: 5px;
                    border-left: 5px solid #38b2ac;
                    margin-top: 30px;
                }}
            </style>
        </head>
        <body>
            <h1>Speaker Classification Approaches Comparison</h1>
            
            <div class="highlight">
                <h3>Summary of Findings</h3>
                <p>This report compares two approaches for speaker classification in audio transcription:</p>
                <ol>
                    <li><strong>Acoustic Feature-Based Approach:</strong> Uses fundamental frequency and spectral features with rule-based classification (mp3_optimized.py)</li>
                    <li><strong>Deep Learning-Based Approach:</strong> Uses neural networks (wav2vec2 model) for feature extraction and classification (mp3_new.py)</li>
                </ol>
                <p>Key findings:</p>
                <ul>
                    <li>Classification similarity between approaches: <strong>{aggregate_stats['similarity']['mean']:.2f}</strong> (1.0 = identical, 0.0 = completely different)</li>
                    <li>Deep Learning approach is <strong>{aggregate_stats['time_ratio']['mean']:.1f}x slower</strong> than Acoustic approach</li>
                </ul>
            </div>
            
            <h2>Approach Comparison</h2>
            
            <div class="approaches">
                <div class="approach acoustic">
                    <h3>Acoustic Feature-Based Approach</h3>
                    <p>Uses handcrafted acoustic features and rule-based classification thresholds.</p>
                    <h4>Key Components:</h4>
                    <ul>
                        <li>Feature extraction: F0 (fundamental frequency), spectral centroid</li>
                        <li>Classification rules based on frequency thresholds</li>
                        <li>Context-based adaptation (e.g., detecting child conversations)</li>
                    </ul>
                </div>
                <div class="approach deep-learning">
                    <h3>Deep Learning-Based Approach</h3>
                    <p>Uses pre-trained neural networks to extract features and classify speakers.</p>
                    <h4>Key Components:</h4>
                    <ul>
                        <li>wav2vec2 model for feature extraction</li>
                        <li>Neural network classification with probabilities</li>
                        <li>Age estimation and confidence scores</li>
                        <li>Detailed visualizations and reporting</li>
                    </ul>
                </div>
            </div>
            
            <h2>Performance Comparison</h2>
            
            <div class="image-container">
                <img src="timing_comparison.png" alt="Processing Time Comparison">
                <p><em>Processing time comparison between the two approaches</em></p>
            </div>
            
            <h2>Classification Comparison</h2>
            
            <div class="image-container">
                <img src="classification_similarity.png" alt="Classification Similarity">
                <p><em>Similarity between classifications from both approaches</em></p>
            </div>
            
            <div class="image-container">
                <img src="speaker_type_distribution.png" alt="Speaker Type Distribution">
                <p><em>Distribution of speaker types identified by each approach</em></p>
            </div>
            
            <h2>Detailed Speaker Analysis</h2>
            
            <table>
                <thead>
                    <tr>
                        <th>File</th>
                        <th>Acoustic Speakers</th>
                        <th>Deep Learning Speakers</th>
                        <th>Similarity</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # Add table rows for each file
        for row in speaker_table_data:
            html_content += f"""
                    <tr>
                        <td>{row['file']}</td>
                        <td>{row['acoustic_count']} ({row['acoustic_types']})</td>
                        <td>{row['dl_count']} ({row['dl_types']})</td>
                        <td>{row['similarity']:.2f}</td>
                    </tr>
            """
        
        # Complete the HTML
        html_content += """
                </tbody>
            </table>
            
            <h2>Strengths and Weaknesses</h2>
            
            <h3>Acoustic Feature-Based Approach</h3>
            <div class="pros-cons">
                <div class="pros">
                    <h4>Strengths</h4>
                    <ul>
                        <li>Faster processing</li>
                        <li>Lower resource usage</li>
                        <li>Simpler implementation</li>
                        <li>No need for model loading</li>
                        <li>Effective for clear adult voices</li>
                        <li>Can use contextual clues from conversation</li>
                    </ul>
                </div>
                <div class="cons">
                    <h4>Weaknesses</h4>
                    <ul>
                        <li>Lower accuracy for edge cases</li>
                        <li>No confidence measures</li>
                        <li>Limited visualization options</li>
                        <li>No age estimation</li>
                        <li>Rule-based thresholds may not work for all voices</li>
                        <li>More prone to misclassification with ambiguous voices</li>
                    </ul>
                </div>
            </div>
            
            <h3>Deep Learning-Based Approach</h3>
            <div class="pros-cons">
                <div class="pros">
                    <h4>Strengths</h4>
                    <ul>
                        <li>Higher classification accuracy</li>
                        <li>Provides confidence scores</li>
                        <li>Age estimation capability</li>
                        <li>Better handling of edge cases</li>
                        <li>More detailed visualizations</li>
                        <li>Probability-based classification</li>
                    </ul>
                </div>
                <div class="cons">
                    <h4>Weaknesses</h4>
                    <ul>
                        <li>Slower processing</li>
                        <li>Higher resource usage</li>
                        <li>Requires model loading</li>
                        <li>More complex implementation</li>
                        <li>Resource intensive</li>
                        <li>No contextual adaptation based on conversation content</li>
                    </ul>
                </div>
            </div>
            
            <div class="conclusion">
                <h2>Conclusion</h2>
                <p>Both approaches have distinct advantages and use cases:</p>
                <ul>
                    <li><strong>Acoustic Feature-Based Approach</strong> is recommended for applications where processing speed is critical, resources are limited, or when working primarily with adult voices in clear recordings.</li>
                    <li><strong>Deep Learning-Based Approach</strong> is recommended for applications requiring high accuracy, detailed speaker information (age, confidence), or when working with challenging audio containing children or ambiguous voices.</li>
                </ul>
                <p>The choice between them should be based on specific application requirements, available computational resources, and the importance of accuracy versus speed.</p>
            </div>
            
            <hr>
            <p><small>Report generated on """ + time.strftime("%Y-%m-%d %H:%M:%S") + """</small></p>
        </body>
        </html>
        """
        
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        print(f"{GREEN}HTML report generated at {html_path}{RESET}")


def main():
    """Main function to run the comparison."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Compare real speaker classification approaches")
    parser.add_argument("--audio_dir", type=str, required=True, 
                        help="Directory containing audio files for testing")
    parser.add_argument("--output_dir", type=str, default="comparison_results", 
                        help="Directory to save comparison results")
    
    args = parser.parse_args()
    
    # Check if audio_dir exists
    if not os.path.isdir(args.audio_dir):
        print(f"{RED}Error: Audio directory '{args.audio_dir}' not found{RESET}")
        sys.exit(1)
    
    # Run the comparison
    comparison = RealComparison(args.audio_dir, args.output_dir)
    comparison.run_comparison()
    
    print(f"\n{GREEN}{BOLD}Comparison completed!{RESET}")
    print(f"View the detailed report at: {os.path.join(args.output_dir, 'comparison_report.html')}")


if __name__ == "__main__":
    main()