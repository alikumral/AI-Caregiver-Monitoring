import os
import json
import argparse
import pygame
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
import soundfile as sf
import threading
from pathlib import Path

class SpeakerLabelerApp:
    """
    Interactive GUI application to label speakers in audio recordings.
    """
    def __init__(self, root, samples_dir, labels_file=None):
        self.root = root
        self.root.title("Speaker Labeler")
        self.root.geometry("900x650")
        self.root.minsize(800, 600)
        
        # Initialize pygame for audio playback
        pygame.mixer.init()
        
        # Store paths
        self.samples_dir = samples_dir
        self.labels_file = labels_file
        
        # Find speaker samples
        self.speakers = self.find_speakers()
        self.current_speaker_idx = 0
        
        # Load existing labels if available
        self.labels = {}
        if labels_file and os.path.exists(labels_file):
            try:
                with open(labels_file, 'r') as f:
                    self.labels = json.load(f)
                print(f"Loaded existing labels for {len(self.labels)} speakers")
            except Exception as e:
                print(f"Error loading labels file: {e}")
        
        # Set up the UI
        self.create_ui()
        
        # Load the first speaker
        if self.speakers:
            self.load_speaker(self.speakers[0])
        
    def find_speakers(self):
        """Find all unique speakers in the samples directory."""
        if not os.path.exists(self.samples_dir):
            messagebox.showerror("Error", f"Samples directory not found: {self.samples_dir}")
            return []
        
        speakers = set()
        for filename in os.listdir(self.samples_dir):
            if filename.startswith("speaker_") and filename.endswith(".wav"):
                # Extract speaker ID (e.g., "SPEAKER_00" from "speaker_SPEAKER_00_segment_1.wav")
                parts = filename.split("_segment_")
                if len(parts) >= 2:
                    # Get the part between "speaker_" and "_segment_"
                    speaker_id = parts[0].replace("speaker_", "")
                    speakers.add(speaker_id)
        
        return sorted(list(speakers))
    
    def create_ui(self):
        """Create the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Speaker navigation
        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(nav_frame, text="Speaker:").pack(side=tk.LEFT)
        
        self.speaker_var = tk.StringVar()
        self.speaker_combo = ttk.Combobox(nav_frame, textvariable=self.speaker_var, values=self.speakers)
        self.speaker_combo.pack(side=tk.LEFT, padx=5)
        self.speaker_combo.bind("<<ComboboxSelected>>", self.on_speaker_selected)
        
        ttk.Button(nav_frame, text="Previous", command=self.prev_speaker).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next", command=self.next_speaker).pack(side=tk.LEFT, padx=5)
        
        # Speaker samples list
        samples_frame = ttk.LabelFrame(main_frame, text="Speaker Samples", padding="10")
        samples_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.samples_listbox = tk.Listbox(samples_frame, height=5)
        self.samples_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.samples_listbox.bind("<<ListboxSelect>>", self.on_sample_selected)
        
        samples_scrollbar = ttk.Scrollbar(samples_frame, orient=tk.VERTICAL, command=self.samples_listbox.yview)
        samples_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.samples_listbox.configure(yscrollcommand=samples_scrollbar.set)
        
        # Playback controls
        playback_frame = ttk.Frame(main_frame)
        playback_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(playback_frame, text="Play Sample", command=self.play_sample).pack(side=tk.LEFT, padx=5)
        ttk.Button(playback_frame, text="Stop", command=self.stop_playback).pack(side=tk.LEFT, padx=5)
        
        # Waveform display
        self.fig, self.ax = plt.subplots(figsize=(8, 3))
        self.canvas_frame = ttk.Frame(main_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Speaker labeling
        label_frame = ttk.LabelFrame(main_frame, text="Speaker Classification", padding="10")
        label_frame.pack(fill=tk.X, pady=10)
        
        self.speaker_type_var = tk.StringVar()
        
        ttk.Radiobutton(label_frame, text="Man", variable=self.speaker_type_var, value="Man").grid(row=0, column=0, padx=20)
        ttk.Radiobutton(label_frame, text="Woman", variable=self.speaker_type_var, value="Woman").grid(row=0, column=1, padx=20)
        ttk.Radiobutton(label_frame, text="Child", variable=self.speaker_type_var, value="Child").grid(row=0, column=2, padx=20)
        ttk.Radiobutton(label_frame, text="Unknown", variable=self.speaker_type_var, value="Unknown").grid(row=0, column=3, padx=20)
        
        # Save button
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Apply Label", command=self.apply_label).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save All Labels", command=self.save_labels).pack(side=tk.RIGHT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, pady=5)
    
    def load_speaker(self, speaker_id):
        """Load a speaker's samples into the UI."""
        self.current_speaker = speaker_id
        self.speaker_var.set(speaker_id)
        
        # Find all samples for this speaker
        self.current_samples = []
        for filename in os.listdir(self.samples_dir):
            if filename.startswith(f"speaker_{speaker_id}_") and filename.endswith(".wav"):
                self.current_samples.append(filename)
        
        # Update the listbox
        self.samples_listbox.delete(0, tk.END)
        for sample in self.current_samples:
            self.samples_listbox.insert(tk.END, sample)
        
        # Select the first sample
        if self.current_samples:
            self.samples_listbox.selection_set(0)
            self.load_sample(self.current_samples[0])
        
        # Set the current label if available
        if speaker_id in self.labels:
            self.speaker_type_var.set(self.labels[speaker_id])
        else:
            self.speaker_type_var.set("")
        
        self.status_var.set(f"Loaded speaker {speaker_id} with {len(self.current_samples)} samples")
    
    def load_sample(self, sample_filename):
        """Load an audio sample and display its waveform."""
        sample_path = os.path.join(self.samples_dir, sample_filename)
        if not os.path.exists(sample_path):
            self.status_var.set(f"Error: Sample file not found: {sample_path}")
            return
        
        # Load audio data
        try:
            audio_data, sample_rate = sf.read(sample_path)
            
            # Plot waveform
            self.ax.clear()
            self.ax.plot(np.arange(len(audio_data))/sample_rate, audio_data)
            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('Amplitude')
            self.ax.set_title(f'Waveform: {sample_filename}')
            self.fig.tight_layout()
            self.canvas.draw()
            
            # Store current sample
            self.current_sample_path = sample_path
            self.status_var.set(f"Loaded sample: {sample_filename}")
            
        except Exception as e:
            self.status_var.set(f"Error loading sample: {e}")
    
    def play_sample(self):
        """Play the currently selected audio sample."""
        if hasattr(self, 'current_sample_path') and os.path.exists(self.current_sample_path):
            # Stop any playing sound first
            self.stop_playback()
            
            # Play the sound
            try:
                pygame.mixer.music.load(self.current_sample_path)
                pygame.mixer.music.play()
                self.status_var.set(f"Playing: {os.path.basename(self.current_sample_path)}")
            except Exception as e:
                self.status_var.set(f"Error playing audio: {e}")
        else:
            self.status_var.set("No sample selected")
    
    def stop_playback(self):
        """Stop any playing audio."""
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
            self.status_var.set("Playback stopped")
    
    def next_speaker(self):
        """Navigate to the next speaker."""
        if not self.speakers:
            return
        
        self.current_speaker_idx = (self.current_speaker_idx + 1) % len(self.speakers)
        self.load_speaker(self.speakers[self.current_speaker_idx])
    
    def prev_speaker(self):
        """Navigate to the previous speaker."""
        if not self.speakers:
            return
        
        self.current_speaker_idx = (self.current_speaker_idx - 1) % len(self.speakers)
        self.load_speaker(self.speakers[self.current_speaker_idx])
    
    def on_speaker_selected(self, event):
        """Handle speaker selection from dropdown."""
        selected = self.speaker_var.get()
        if selected in self.speakers:
            self.current_speaker_idx = self.speakers.index(selected)
            self.load_speaker(selected)
    
    def on_sample_selected(self, event):
        """Handle sample selection from listbox."""
        if not self.current_samples:
            return
        
        selection = self.samples_listbox.curselection()
        if selection:
            index = selection[0]
            self.load_sample(self.current_samples[index])
    
    def apply_label(self):
        """Apply the selected label to the current speaker."""
        if not hasattr(self, 'current_speaker'):
            self.status_var.set("No speaker selected")
            return
        
        label = self.speaker_type_var.get()
        if not label:
            self.status_var.set("Please select a speaker type")
            return
        
        self.labels[self.current_speaker] = label
        self.status_var.set(f"Applied label '{label}' to speaker {self.current_speaker}")
        
        # Automatically move to the next speaker
        self.next_speaker()
    
    def save_labels(self):
        """Save the labels to a JSON file."""
        if not self.labels:
            self.status_var.set("No labels to save")
            return
        
        if not self.labels_file:
            self.labels_file = "speaker_labels.json"
        
        try:
            with open(self.labels_file, 'w') as f:
                json.dump(self.labels, f, indent=2)
            self.status_var.set(f"Saved {len(self.labels)} labels to {self.labels_file}")
            messagebox.showinfo("Success", f"Labels saved to {self.labels_file}")
        except Exception as e:
            self.status_var.set(f"Error saving labels: {e}")
            messagebox.showerror("Error", f"Failed to save labels: {e}")

def interactive_labeling(samples_dir="speaker_samples", labels_file="speaker_labels.json"):
    """Launch the interactive speaker labeling tool."""
    root = tk.Tk()
    app = SpeakerLabelerApp(root, samples_dir, labels_file)
    root.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive speaker labeling tool")
    parser.add_argument("--samples", default="speaker_samples", help="Directory containing speaker samples")
    parser.add_argument("--labels", default="speaker_labels.json", help="Output file for speaker labels")
    
    args = parser.parse_args()
    
    interactive_labeling(args.samples, args.labels)