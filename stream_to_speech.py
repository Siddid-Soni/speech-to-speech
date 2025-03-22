import os
import threading
import queue
import re
from typing import Iterator
import sounddevice as sd
import numpy as np
import time

from agno.agent import Agent, RunResponse
from agno.models.google import Gemini
from agno.models.ollama import Ollama
from ollama import Client as OllamaClient
from kokoro import KPipeline
import soundfile as sf

# Initialize Kokoro TTS pipeline
pipeline = KPipeline(lang_code='a')  # 'a' for American English
voice = 'af_heart'  # Choose your preferred voice

# Set up Agno agent with streaming capability
agent = Agent(
    model=Gemini(id="gemini-2.0-flash", api_key=""),
    description="You are a voice-assistant model",
    instructions="include proper punctuations for good pronunciation and do not use any markdown or symbols",
)
# agent = Agent(
#     model=Ollama(id="llama3.1:latest", client=OllamaClient("http://172.19.96.1:11434/")),
#     description="You are a voice-assistant model",
#     instructions="include proper punctuations for good pronunciation and do not use any markdown or symbols",
# )

# Create output directory if it doesn't exist
os.makedirs("audio_chunks", exist_ok=True)

class RealTimeAudioStreamer:
    def __init__(self, tts_pipeline, voice="af_heart", speed=1.0):
        self.pipeline = tts_pipeline
        self.voice = voice
        self.speed = speed
        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.text_processing_thread = None
        self.audio_playback_thread = None
        self.stop_event = threading.Event()
        self.full_text = ""
        self.chunk_index = 0
        self.sentence_buffer = ""
        self.processing_lock = threading.Lock()  # For thread safety
        
    def process_text_to_audio(self):
        """Thread function to convert text chunks to audio"""
        while not self.stop_event.is_set() or not self.text_queue.empty():
            try:
                # Get text from queue with timeout
                text_chunk = self.text_queue.get(timeout=0.1)
                
                # Critical section - minimize lock holding time
                sentences_to_process = []
                with self.processing_lock:
                    # Add to sentence buffer
                    self.sentence_buffer += text_chunk
                    
                    # Check if we have complete sentences to process
                    sentences = self._split_into_sentences(self.sentence_buffer)
                    
                    if len(sentences) > 1:  # We have complete sentences
                        # Keep incomplete sentence in buffer
                        self.sentence_buffer = sentences[-1]
                        # Get sentences to process outside the lock
                        sentences_to_process = [s for s in sentences[:-1] if s.strip()]
                
                # Process sentences without holding the lock
                for sentence in sentences_to_process:
                    try:
                        sentence_audio = []
                        for _, _, audio in self.pipeline(sentence, voice=self.voice, speed=self.speed):
                            sentence_audio.append(audio)
                        
                        if sentence_audio:
                            # Add sentence audio to queue
                            self.audio_queue.put(sentence_audio[0])
                            
                            # Save the audio chunk
                            audio_path = f"audio_chunks/chunk_{self.chunk_index}.wav"
                            sf.write(audio_path, sentence_audio[0], 24000)
                            self.chunk_index += 1
                    except Exception as e:
                        print(f"Error processing sentence: {e}")
                
                self.text_queue.task_done()
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error in audio processing thread: {e}")
    
    def play_audio(self):
        """Thread function for audio playback"""
        while not self.stop_event.is_set() or not self.audio_queue.empty():
            try:
                # Get audio from queue with shorter timeout
                audio = self.audio_queue.get(timeout=0.05)
                
                # Play the audio
                sd.play(audio, samplerate=24000)
                sd.wait()  # Wait for playback to complete
                
                self.audio_queue.task_done()
            except queue.Empty:
                # No audio available, just continue
                pass
            except Exception as e:
                print(f"Error in audio playback thread: {e}")
    
    def _split_into_sentences(self, text):
        """Split text into sentences, keeping the last incomplete sentence in buffer"""
        # Basic sentence splitting on punctuation
        sentence_endings = re.finditer(r'([.!?])\s+', text)
        
        # Find positions of all sentence endings
        end_positions = [match.end() for match in sentence_endings]
        
        if not end_positions:
            # No complete sentence found, return original text as buffer
            return [text]
        
        # Extract complete sentences
        sentences = []
        start_pos = 0
        
        for end_pos in end_positions:
            sentences.append(text[start_pos:end_pos])
            start_pos = end_pos
        
        # Add remaining text (incomplete sentence) as the last item
        sentences.append(text[start_pos:])
        
        return sentences
    
    def start(self):
        """Start the processing and playback threads"""
        self.stop_event.clear()
        
        # Start text processing thread
        self.text_processing_thread = threading.Thread(
            target=self.process_text_to_audio, 
            daemon=True
        )
        self.text_processing_thread.start()
        
        # Start audio playback thread
        self.audio_playback_thread = threading.Thread(
            target=self.play_audio, 
            daemon=True
        )
        self.audio_playback_thread.start()
    
    def stop(self):
        """Stop the processing and playback threads"""
        self.stop_event.set()
        
        # Process any remaining text in the buffer
        if self.sentence_buffer.strip():
            for _, _, audio in self.pipeline(self.sentence_buffer, voice=self.voice, speed=self.speed):
                self.audio_queue.put(audio)
                audio_path = f"audio_chunks/chunk_{self.chunk_index}.wav"
                sf.write(audio_path, audio, 24000)
                self.chunk_index += 1
        
        # Wait for threads to finish
        if self.text_processing_thread:
            self.text_processing_thread.join(timeout=5)
        if self.audio_playback_thread:
            self.audio_playback_thread.join(timeout=5)
        
        # Generate complete audio file
        self._save_full_audio()
    
    def add_text(self, text):
        """Add text to the processing queue"""
        self.text_queue.put(text)
        self.full_text += text
    
    def _save_full_audio(self):
        """Generate and save audio for the full text"""
        print("Generating full audio from complete response...")
        full_audio = []
        for _, _, audio in pipeline(self.full_text, voice=self.voice, speed=self.speed, split_pattern=r'\n+'):
            full_audio.append(audio)
        
        # Concatenate and save full audio
        if full_audio:
            full_audio_array = np.concatenate(full_audio)
            sf.write("audio_chunks/full_response.wav", full_audio_array, 24000)

    def save_full_audio(self):
        """Generate and save audio for the full text without stopping the streamer"""
        print("Generating full audio from complete response...")
        full_audio = []
        for _, _, audio in pipeline(self.full_text, voice=self.voice, speed=self.speed, split_pattern=r'\n+'):
            full_audio.append(audio)
        
        # Concatenate and save full audio
        if full_audio:
            full_audio_array = np.concatenate(full_audio)
            sf.write("audio_chunks/full_response.wav", full_audio_array, 24000)

def process_stream_to_speech_realtime(prompt: str, audio_streamer=None):
    """Process streamed LLM response with real-time audio conversion and playback"""
    # Create a new streamer if one wasn't provided
    if audio_streamer is None:
        audio_streamer = RealTimeAudioStreamer(pipeline, voice=voice)
        audio_streamer.start()
    else:
        # Reset for new prompt
        audio_streamer.full_text = ""
        with audio_streamer.processing_lock:
            audio_streamer.sentence_buffer = ""
        
        # Wait for queues to clear with a reasonable timeout
        timeout = time.time() + 2  # 2-second timeout
        while (not audio_streamer.text_queue.empty() or not audio_streamer.audio_queue.empty()) and time.time() < timeout:
            time.sleep(0.05)
    
    # Get streaming response
    try:
        response_stream: Iterator[RunResponse] = agent.run(prompt, stream=True)
    except Exception as e:
        print(f"Error getting response: {e}")
        return "", audio_streamer
    
    try:
        # Process each chunk as it arrives
        for chunk in response_stream:
            if chunk and chunk.content:
                # Add text to the processing queue
                audio_streamer.add_text(chunk.content)
                print(chunk.content, end="", flush=True)  # Print text as it arrives
        
        print("\n")  # Add newline at the end
        
        # Process final sentence - get it from the buffer
        final_sentence = ""
        with audio_streamer.processing_lock:
            if audio_streamer.sentence_buffer.strip():
                final_sentence = audio_streamer.sentence_buffer
                # Add period if needed
                if not any(final_sentence.rstrip().endswith(p) for p in '.!?'):
                    final_sentence += '.'
                # Clear buffer but keep reference to the sentence
                audio_streamer.sentence_buffer = ""
                # Add to full text
                audio_streamer.full_text += final_sentence
        
        # Process final sentence outside the lock
        if final_sentence:
            try:
                final_audio = []
                for _, _, audio in audio_streamer.pipeline(final_sentence, 
                                                          voice=audio_streamer.voice, 
                                                          speed=audio_streamer.speed):
                    final_audio.append(audio)
                    
                if final_audio:
                    # Add to queue and save
                    audio_streamer.audio_queue.put(final_audio[0])
                    audio_path = f"audio_chunks/chunk_{audio_streamer.chunk_index}.wav"
                    sf.write(audio_path, final_audio[0], 24000)
                    audio_streamer.chunk_index += 1
            except Exception as e:
                print(f"Error processing final sentence: {e}")
        
        # Wait for audio to finish with reasonable timeout
        timeout = time.time() + 3  # 3-second timeout
        while not audio_streamer.audio_queue.empty() and time.time() < timeout:
            time.sleep(0.1)
        
        # Save full audio without waiting for completion
        audio_streamer.save_full_audio()
        
        return audio_streamer.full_text, audio_streamer
    except Exception as e:
        print(f"Error during streaming: {e}")
        return audio_streamer.full_text, audio_streamer

# Example usage
if __name__ == "__main__":
    # First install sounddevice: pip install sounddevice
    streamer = None
    
    try:
        while True:
            prompt = input("\nEnter your question (or press Ctrl+C to exit): ")
            if not prompt.strip():
                continue
                
            full_response, streamer = process_stream_to_speech_realtime(prompt, streamer)
            print("\nResponse complete. Audio saved to audio_chunks/full_response.wav")
            print("The streamer is still running. You can ask another question.")
            
    except KeyboardInterrupt:
        print("\nExiting...")
        # Only stop the streamer when exiting the program
        if streamer:
            streamer.stop() 
