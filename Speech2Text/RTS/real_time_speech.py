import pyaudio
import wave
import os
from faster_whisper import WhisperModel 

# You may have color constants defined somewhere else, for example:
NEON_GREEN = "\033[32m"
RESET_COLOR = "\033[0m"


def record_chunk(p, stream, file_path, chunk_length=1):
    frames = []
    # The loop count is derived from the sample rate (16000), 
    # the input chunk size (1024), and the duration chunk_length in seconds.
    for _ in range(0, int(16000 / 1024 * chunk_length)):
        data = stream.read(1024)
        frames.append(data)

    # Write the recorded frames to a WAV file
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()

def transcribe_chunk(model, file_path):
    segments, info = model.transcribe(file_path, beam_size=7)
    transcription = ' '.join(segment.text for segment in segments)
    return transcription

def main2():
    # Choose your model settings
    model_size = "medium.en"
    model = WhisperModel(
        model_size,
        device="cuda",           # change to "cpu" if you don't have CUDA
        compute_type="float16"   # or "int8"/"float32" depending on your GPU/CPU
    )

    # Initialize PyAudio
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=1024
    )

    accumulated_transcription = ""  # Initialize an empty string to accumulate transcriptions

    try:
        while True:
            chunk_file = "temp_chunk.wav"

            # Record a short chunk of audio
            record_chunk(p, stream, chunk_file)

            # Transcribe that short chunk
            # (Assumes you have a function named transcribe_chunk; you might adapt or inline it)
            transcription = transcribe_chunk(model, chunk_file)

            # Print it to console in some color
            print(NEON_GREEN + transcription + RESET_COLOR)

            # Remove the temporary wav file
            os.remove(chunk_file)

            # Append the new transcription to the accumulated transcription
            accumulated_transcription += transcription + " "

    except KeyboardInterrupt:
        print("Stopping...")

        # Write the accumulated transcription to the log file
        with open("log.txt", "w") as log_file:
            log_file.write(accumulated_transcription)

    finally:
        # Optional: print the final transcription to the console
        print("LOG: " + accumulated_transcription)

        # Clean up PyAudio
        stream.stop_stream()
        stream.close()
        p.terminate()


# If this file is run directly, start main2()
if __name__ == "__main__":
    main2()