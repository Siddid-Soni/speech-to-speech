import os
import threading
import queue
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from typing import Iterator, Optional, List, Dict
import torch

# Import Whisper for speech recognition
from transformers import pipeline as hf_pipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Import Agno for AI agent
from agno.agent import Agent, RunResponse
from agno.models.google import Gemini
# Alternatively use OpenAI
# from agno.models.openai import OpenAIChat

# Import Kokoro for TTS
from kokoro import KPipeline

# Create directories for audio files
os.makedirs("audio_input", exist_ok=True)
os.makedirs("audio_output", exist_ok=True)

class VoiceActivityDetector:
    def __init__(self):
        print("Loading Silero VAD model...")
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=True
        )
        self.get_speech_timestamps, _, self.read_audio, *_ = utils
        torch.set_num_threads(1)  # Optimize for single thread performance
        
    def detect_speech(self, audio_data: np.ndarray, sample_rate: int = 16000) -> List[Dict]:
        """
        Detect speech segments in audio using Silero VAD
        Returns list of timestamps where speech is detected
        """
        # Convert audio to the format expected by Silero VAD
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Get speech timestamps
        speech_timestamps = self.get_speech_timestamps(
            torch.from_numpy(audio_data),
            self.model,
            sampling_rate=sample_rate,
            return_seconds=True  # Return timestamps in seconds
        )
        
        return speech_timestamps

class SpeechToSpeechPipeline:
    def __init__(
        self,
        whisper_model_id: str = "openai/whisper-medium",
        tts_lang_code: str = "a",  # 'a' for American English
        tts_voice: str = "af_heart",
        agent_model_id: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        transcription_language: str = "english",  # Specify the input language
    ):
        # Add VAD initialization
        self.vad = VoiceActivityDetector()
        
        # Initialize Speech Recognition (Whisper)
        print("Loading Whisper model...")
        self.whisper_processor = WhisperProcessor.from_pretrained(whisper_model_id)
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_id)
        self.transcription_language = transcription_language
        
        # Initialize Speech Synthesis (Kokoro)
        print("Loading Kokoro TTS model...")
        self.tts_pipeline = KPipeline(lang_code=tts_lang_code)  # 'a' for American English
        self.tts_voice = tts_voice
        
        # Initialize AI Agent (Agno)
        print("Setting up Agno agent...")
        self.agent = Agent(
            model=Gemini(id=agent_model_id, api_key=api_key),
            description="You are a helpful voice assistant that provides clear, concise information.",
            instructions="Include proper punctuation for good pronunciation. Keep responses brief and direct."
        )
        
        # Set up recording parameters
        self.sample_rate = 16000
        self.recording_duration = 5  # seconds (can be adjusted)
        
        # Initialize audio streamer
        self.audio_streamer = RealTimeAudioStreamer(self.tts_pipeline, voice=self.tts_voice)
        self.audio_streamer.start()
        
        # Add parameters for continuous listening
        self.silence_threshold = 0.5  # seconds of silence to mark end of speech
        self.min_speech_duration = 0.5  # minimum duration to consider as speech
        self.buffer_duration = 0.5  # audio buffer size in seconds
        self.stop_listening = False
        
    def listen(self, duration=None):
        """Record audio from microphone with voice activity detection"""
        if duration is None:
            duration = self.recording_duration
            
        print(f"Listening for {duration} seconds...")
        
        # Record audio
        recording = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1
        )
        sd.wait()
        
        # Detect speech segments
        speech_timestamps = self.vad.detect_speech(recording.squeeze())
        
        if not speech_timestamps:
            print("No speech detected.")
            return None, None
        
        # Extract speech segments
        speech_segments = []
        for ts in speech_timestamps:
            start_sample = int(ts['start'] * self.sample_rate)
            end_sample = int(ts['end'] * self.sample_rate)
            speech_segments.append(recording[start_sample:end_sample])
        
        # Concatenate all speech segments
        if speech_segments:
            speech_audio = np.concatenate(speech_segments)
        else:
            return None, None
        
        # Save the processed recording
        timestamp = int(time.time())
        input_file = f"audio_input/input_{timestamp}.wav"
        sf.write(input_file, speech_audio, self.sample_rate)
        
        return input_file, speech_audio
    
    def transcribe(self, audio_input):
        """Transcribe speech to text using Whisper"""
        print("Transcribing audio...")
        
        # Configure language and task using the recommended approach
        forced_decoder_ids = self.whisper_processor.get_decoder_prompt_ids(
            language=self.transcription_language, 
            task="transcribe"
        )
        
        # Process audio input - could be a file path or numpy array
        if isinstance(audio_input, str):
            # Load audio from file
            speech_array, _ = sf.read(audio_input)
            speech_array = speech_array.astype(np.float32)
        else:
            # Assume it's already a numpy array
            speech_array = audio_input.astype(np.float32)
        
        # Convert to features
        input_features = self.whisper_processor.feature_extractor(
            speech_array, 
            sampling_rate=self.sample_rate, 
            return_tensors="pt"
        ).input_features
        
        # Generate transcription
        predicted_ids = self.whisper_model.generate(
            input_features, 
            forced_decoder_ids=forced_decoder_ids
        )
        
        # Decode the transcription
        transcription = self.whisper_processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]
        
        print(f"Transcription: {transcription}")
        return transcription
    
    def process_with_agent(self, text_input):
        """Process text with Agno agent"""
        print("Processing with AI agent...")
        
        # Get streaming response
        response_stream = self.agent.run(text_input, stream=True)
        
        # Process the response stream
        full_response, self.audio_streamer = process_stream_to_speech_realtime(
            response_stream, 
            self.audio_streamer
        )
        
        return full_response
    
    def run_pipeline(self, audio_input=None, text_input=None):
        """Run the full pipeline with voice activity detection"""
        try:
            # Step 1: Get audio input (record or use provided)
            if audio_input is None and text_input is None:
                audio_input, audio_data = self.listen()
                if audio_input is None:
                    print("No speech detected. Please try again.")
                    return None
            
            # Step 2: Transcribe audio to text (if audio was provided)
            if audio_input is not None:
                text_input = self.transcribe(audio_input)
            
            # Step 3: Process text with AI agent
            if text_input:
                response = self.process_with_agent(text_input)
                return response
            else:
                print("No valid input detected.")
                return None
                
        except Exception as e:
            print(f"Error in pipeline: {e}")
            return None
    
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'audio_streamer'):
            self.audio_streamer.stop()
    
    def continuous_listen(self):
        """Continuously monitor audio input for speech"""
        print("Listening... (Press Ctrl+C to exit)")
        
        # Calculate buffer size
        buffer_size = int(self.sample_rate * self.buffer_duration)
        audio_buffer = []
        is_speaking = False
        silence_start = None
        
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio input error: {status}")
                return
            
            # Add new audio to buffer
            audio_buffer.append(indata.copy())
            if len(audio_buffer) > int(5 / self.buffer_duration):  # Keep 5 seconds of audio
                audio_buffer.pop(0)
            
            nonlocal is_speaking, silence_start
            
            # Detect speech in the current chunk
            audio_chunk = indata.squeeze()
            speech_detected = self.vad.detect_speech(audio_chunk, self.sample_rate)
            
            if speech_detected and not is_speaking:
                # Speech started
                is_speaking = True
                silence_start = None
                print("\rListening... 🎤", end="", flush=True)
            
            elif not speech_detected and is_speaking:
                # Potential end of speech
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > self.silence_threshold:
                    # Process the accumulated audio
                    self.process_speech_segment(np.concatenate(audio_buffer))
                    audio_buffer.clear()
                    is_speaking = False
                    silence_start = None
                    print("\rListening... 🔍", end="", flush=True)
        
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=audio_callback,
                blocksize=buffer_size
            ):
                while not self.stop_listening:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping voice assistant...")
            self.stop_listening = True
    
    def process_speech_segment(self, audio_data):
        """Process a segment of speech"""
        try:
            # Save the audio segment
            timestamp = int(time.time())
            input_file = f"audio_input/input_{timestamp}.wav"
            sf.write(input_file, audio_data, self.sample_rate)
            
            # Transcribe
            transcription = self.transcribe(audio_data)
            if not transcription or transcription.strip() == "":
                return
            
            # Process with AI and generate response
            response = self.process_with_agent(transcription)
            if response:
                print(f"\nYou: {transcription}")
                print(f"Assistant: {response}\n")
            
        except Exception as e:
            print(f"\nError processing speech: {e}")
    
    def run_continuous_pipeline(self):
        """Run the pipeline in continuous conversation mode"""
        try:
            self.stop_listening = False
            self.continuous_listen()
        except Exception as e:
            print(f"Error in continuous pipeline: {e}")
        finally:
            self.stop_listening = True


# Real-time audio streaming class for TTS output
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
                            audio_path = f"audio_output/chunk_{self.chunk_index}.wav"
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
        import re
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
                audio_path = f"audio_output/chunk_{self.chunk_index}.wav"
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
    
    def save_full_audio(self):
        """Generate and save audio for the full text without stopping the streamer"""
        print("Generating full audio from complete response...")
        full_audio = []
        for _, _, audio in self.pipeline(self.full_text, voice=self.voice, speed=self.speed, split_pattern=r'\n+'):
            full_audio.append(audio)
        
        # Concatenate and save full audio
        if full_audio:
            full_audio_array = np.concatenate(full_audio)
            sf.write("audio_output/full_response.wav", full_audio_array, 24000)


# Helper function to process LLM responses and convert to speech
def process_stream_to_speech_realtime(response_stream, audio_streamer=None):
    """Process streamed LLM response with real-time audio conversion and playback"""
    # Create a new streamer if one wasn't provided
    if audio_streamer is None:
        audio_streamer = RealTimeAudioStreamer(KPipeline(lang_code='a'), voice='af_heart')
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
                    audio_path = f"audio_output/chunk_{audio_streamer.chunk_index}.wav"
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


# Interactive voice assistant loop
def interactive_voice_assistant(api_key=None):
    # Initialize the pipeline
    pipeline = SpeechToSpeechPipeline(
        api_key=api_key,
        whisper_model_id="openai/whisper-medium",
        tts_voice="af_heart",
        transcription_language="english"
    )
    
    print("Voice Assistant initialized! Press Ctrl+C to exit.")
    print("Start speaking whenever you're ready...")
    
    try:
        # Run in continuous mode
        pipeline.run_continuous_pipeline()
    except KeyboardInterrupt:
        print("\nExiting voice assistant...")
    finally:
        if hasattr(pipeline, 'audio_streamer'):
            pipeline.audio_streamer.stop()


# Run the voice assistant if executed directly
if __name__ == "__main__":
    # Provide your API key or set it as an environment variable
    api_key = "" # or "your-api-key-here"
    
    interactive_voice_assistant(api_key) 
