import pyaudio
import wave
import os
import torch

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

def align_speakers_to_transcription(diarization_annotation, transcription_segments):
    """
    Matches each transcription segment to the speaker from the diarization.
    Returns a list of strings like: "[Speaker X] text".
    
    Naive approach: For each transcription segment, find which speaker 
    covers the largest portion of that segment's time range.
    """
    labeled_segments = []

    for tseg in transcription_segments:
        t_start = tseg["start"]
        t_end = tseg["end"]

        # We'll represent the transcription segment as a pyannote Segment
        trans_segment = Segment(t_start, t_end)

        # Find overlapping diarization segments
        # diarization_annotation is a pyannote.core.Annotation
        overlapping_speakers = diarization_annotation.crop(
            trans_segment, mode="intersection"
        )

        if len(overlapping_speakers) == 0:
            # No speaker found (maybe silence?), just label as "Unknown"
            speaker_label = "Unknown"
        else:
            # overlapping_speakers is another Annotation object
            # Possibly multiple speakers overlap. We pick the one with largest overlap.
            speaker_label = None
            max_overlap = 0.0

            for speech_turn, _, spk_label in overlapping_speakers.itertracks(yield_label=True):
                # speech_turn is a pyannote.core.Segment for the overlap
                overlap_duration = speech_turn.duration
                if overlap_duration > max_overlap:
                    max_overlap = overlap_duration
                    speaker_label = spk_label

            # If speaker_label remains None, fallback to "Unknown"
            if speaker_label is None:
                speaker_label = "Unknown"

        # Build the final text line
        line = f"[{speaker_label}] {tseg['text']}"
        labeled_segments.append(line)

    return labeled_segments

def main():
    """
    Repeatedly:
    1) Records a 10-second audio chunk to 'temp_chunk.wav'
    2) Diarizes that chunk to find speaker segments
    3) Transcribes that chunk with faster-whisper
    4) Aligns each transcription segment with the dominant speaker
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

    try:
        while True:
            # (A) Record a 10-second chunk
            record_chunk(p, stream, wav_file, chunk_length=15)

            # (B) Diarize
            diarization_annotation = diarize_audio(diarization_pipeline, wav_file)

            # (C) Transcribe
            transcription_segments = transcribe_audio(whisper_model, wav_file)

            # (D) Align speakers -> labeled text lines
            labeled_transcript_lines = align_speakers_to_transcription(
                diarization_annotation,
                transcription_segments
            )

            # Join them into a block for printing
            block_text = "\n".join(labeled_transcript_lines)
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
