import os
import torch
import numpy as np
import subprocess
from pathlib import Path
import tempfile
import re
import argparse
import sys
import json
import requests
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import string
from environment.config.llm import gpt

class VideoTranscriber:
    def __init__(self, model_path=None):
        """Initialize the transcriber with Whisper model via Hugging Face transformers."""
        # Determine model path
        if model_path is None:
            # Get the current file's directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Navigate up to root directory
            project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
            
            # Set model path to correct location in tools directory
            model_path = os.path.join(project_root, 'tools', 'whisper-large-v3-turbo')
        
        print(f"Loading Whisper model from {model_path}...")
        
        # Set up the device and dtype
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Define punctuation to remove
        self.punctuation = string.punctuation + '，。？！；：""''（）【】《》「」『』、'
        
        # Load the model
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path, 
            torch_dtype=self.torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True
        )
        self.model.to(self.device)
        
        # Load the processor
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        # Create the pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
            chunk_length_s=30,  # Process in 30-second chunks
            return_timestamps=True  # Important for subtitles
        )
        
        print(f"Model loaded on {self.device}.")
        
    def extract_audio(self, video_path, temp_audio_path=None):
        """Extract audio from video file to a temporary WAV file."""
        if temp_audio_path is None:
            # Create a temporary file with .wav extension
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_audio_path = temp_file.name
            temp_file.close()
            
        print(f"Extracting audio to {temp_audio_path}...")
        
        # Use FFmpeg to extract audio
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite output file if it exists
            temp_audio_path
        ]
        
        subprocess.run(cmd, check=True)
        print("Audio extraction completed.")
        
        return temp_audio_path
        
    def remove_punctuation(self, text):
        """Remove all punctuation from text."""
        # Create a translation table to remove punctuation
        translator = str.maketrans('', '', self.punctuation)
        return text.translate(translator)
        
    def transcribe_video(self, video_path):
        """Transcribe the audio from a video file."""
        print(f"Transcribing audio from {video_path}...")
        
        # Extract audio to a temporary file
        temp_audio_path = self.extract_audio(video_path)
        
        try:
            # Transcribe the audio
            result = self.pipe(temp_audio_path)
            
            # Print raw result for debugging
            print("Raw transcription result:", result)
            
            # Format the result to be compatible with our subtitle generation
            formatted_result = self._format_result(result)
            
            # Break long sentences into smaller chunks
            processed_result = self._process_segments_for_shorter_subtitles(formatted_result)
            
            # Remove punctuation from all segments
            for segment in processed_result['segments']:
                segment['text'] = self.remove_punctuation(segment['text'])
            
            print(f"Created {len(processed_result['segments'])} subtitle segments")
            
            # Print the first few segments for debugging
            for i, segment in enumerate(processed_result['segments'][:3]):
                if i < len(processed_result['segments']):
                    print(f"Segment {i}: {segment['start']} -> {segment['end']}: {segment['text']}")
            
            print("Transcription completed.")
            return processed_result
            
        finally:
            # Clean up the temporary audio file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                print(f"Temporary audio file removed: {temp_audio_path}")
    
    def _format_result(self, result):
        """Format the transformers pipeline result to match our expected format."""
        segments = []
        
        # Check if we have chunks with timestamps
        if isinstance(result, dict) and 'chunks' in result:
            for i, chunk in enumerate(result['chunks']):
                segments.append({
                    'id': i,
                    'start': chunk['timestamp'][0],
                    'end': chunk['timestamp'][1],
                    'text': chunk['text'].strip()
                })
        
        # If no segments were created, create a segment from the full text
        if not segments and isinstance(result, dict) and 'text' in result:
            segments.append({
                'id': 0,
                'start': 0,
                'end': 30,  # Arbitrary end time if no timestamps
                'text': result['text'].strip()
            })
        
        return {'segments': segments}
    
    def _process_segments_for_shorter_subtitles(self, result):
        """Process segments to create shorter subtitle lines."""
        new_segments = []
        
        for segment in result['segments']:
            text = segment['text'].strip()
            start = segment['start']
            end = segment['end']
            
            # Skip empty segments
            if not text:
                continue
                
            # First, split by punctuation
            parts = re.split(r'([,，。？！；?!;])', text)
            
            # Rejoin punctuation with the preceding text
            chunks = []
            for i in range(0, len(parts), 2):
                if i < len(parts):
                    chunk = parts[i]
                    if i + 1 < len(parts):
                        chunk += parts[i + 1]
                    chunks.append(chunk)
            
            # If the splitting didn't work or produced only one chunk, 
            # split by length (about 10-15 characters per chunk)
            if len(chunks) <= 1:
                # For Chinese, about 10-15 characters is appropriate for a line
                max_chars = 15
                chunks = []
                
                while text:
                    if len(text) <= max_chars:
                        chunks.append(text)
                        break
                    
                    # Try to find a natural break point (space for English, or just split for Chinese)
                    break_point = max_chars
                    if ' ' in text[:max_chars]:
                        # For English, split at the last space
                        break_point = text[:max_chars].rindex(' ')
                    
                    chunks.append(text[:break_point])
                    text = text[break_point:].strip()
            
            # Calculate time per chunk
            if len(chunks) > 0:
                time_per_chunk = (end - start) / len(chunks)
                
                for i, chunk in enumerate(chunks):
                    chunk_start = start + i * time_per_chunk
                    chunk_end = chunk_start + time_per_chunk
                    
                    new_segments.append({
                        'id': len(new_segments),
                        'start': chunk_start,
                        'end': chunk_end,
                        'text': chunk.strip()
                    })
        
        return {'segments': new_segments}
    
    def save_transcript(self, result, output_path):
        """Save the transcript to a file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for segment in result['segments']:
                start_time = segment['start']
                end_time = segment['end']
                text = segment['text']
                f.write(f"{start_time:.2f} --> {end_time:.2f}\n{text}\n\n")
        print(f"Transcript saved to {output_path}")
        return output_path
    
    def create_srt(self, result, output_path):
        """Create an SRT subtitle file from transcription result."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(result['segments']):
                start_time = self._format_time(segment['start'])
                end_time = self._format_time(segment['end'])
                text = segment['text']
                
                f.write(f"{i+1}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{text}\n\n")
        
        print(f"SRT file saved to {output_path}")
        return output_path
    
    def _format_time(self, seconds):
        """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
        # Handle negative timestamps that might result from applying delay
        seconds = max(0, seconds)
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"
    
    def add_subtitles_to_video(self, video_path, result, output_path):
        """Add subtitles directly to the video using MoviePy."""
        print(f"Adding subtitles to {video_path}...")
        
        # Setting a dummy audio device to avoid ALSA errors
        os.environ["MOVIEPY_EDITOR_AUDIO_ARGS"] = "dummy"
        
        video = VideoFileClip(video_path)
        video_duration = video.duration
        print(f"Video duration: {video_duration} seconds")
        
        # Create TextClips for each segment
        subtitle_clips = []
        
        for segment in result['segments']:
            start_time = segment['start']
            end_time = segment['end']
            text = segment['text'].strip()
            
            # Skip segments that would start before the video begins
            if start_time < 0:
                start_time = 0
                
            # Skip segments that would end after the video ends
            if end_time > video_duration:
                end_time = video_duration
                
            if text:
                print(f"Creating subtitle: '{text}' at {start_time} -> {end_time}")
                
                # Create more visible TextClip - white text with size 18
                txt_clip = (TextClip(text, 
                                    fontsize=18,  # Font size 18 as requested
                                    color='white', 
                                    font='Arial',
                                    stroke_color='black',
                                    stroke_width=0.5,  # Thin stroke for visibility
                                    method='caption',
                                    size=(video.w * 0.9, None))  # Width limit
                            .set_position(('center', 'bottom'))
                            .margin(bottom=20, opacity=0)  # Add bottom margin
                            .set_start(start_time)
                            .set_end(end_time))
                
                subtitle_clips.append(txt_clip)
        
        print(f"Created {len(subtitle_clips)} subtitle clips")
        
        # Add subtitles to the video
        final_video = CompositeVideoClip([video] + subtitle_clips)
        
        # Write the result to a file
        print("Writing final video with subtitles...")
        final_video.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            fps=video.fps  # Maintain original fps
        )
        
        print(f"Video with subtitles saved to {output_path}")
        return output_path
        
    def add_subtitles_with_ffmpeg(self, video_path, srt_path, output_path):
        """Alternative method to add subtitles using FFmpeg."""
        print(f"Adding subtitles to {video_path} using FFmpeg...")
        
        # FFmpeg command with smaller white text positioned at bottom
        cmd = [
            'ffmpeg', 
            '-i', video_path,
            '-vf', f"subtitles={srt_path}:force_style='FontName=Arial,FontSize=24,PrimaryColour=&HFFFFFF&,OutlineColour=&H000000&,BorderStyle=1,Outline=1,Alignment=2,MarginV=50'",
            '-c:v', 'libx264', 
            '-c:a', 'copy',
            '-y',  # Overwrite output file if it exists
            output_path
        ]
        
        subprocess.run(cmd, check=True)
        print(f"Video with subtitles saved to {output_path}")
        return output_path

class LLMClient:
    """Client for making requests to LLM API with retry capabilities."""
    
    def __init__(self):
        """Initialize the client."""
        self.model = "gpt-4o-mini"  # Using GPT-4o Mini as specified
    
    @retry(
        stop=stop_after_attempt(5),  # Retry up to 5 times
        wait=wait_exponential(multiplier=1, min=2, max=60),  # Exponential backoff
        reraise=True
    )
    def refine_subtitles(self, transcribed_text, actual_content):
        """Refine transcribed subtitles based on actual content using LLM."""
        print("Refining transcription...")
        
        # Prepare the prompt for the LLM
        prompt = f"""
        Here is a video with subtitles that need to be corrected.

        Here's the actual content that should be in the video (But all of English numbers need to be converted into Arabic Numerals, eg. iphone five s >>> iphone 5s, Tesla model two >>> Tesla model 2):
        {actual_content}

        And here's what was transcribed:
        {transcribed_text}

        Only correct the transcribed text to match the actual content. Preserve the timestamp format (MM:SS --> MM:SS) at the beginning of each line. Only fix the text after the timestamps

        Do not respond to any irrelevant content

        Do not use ```

        """
        
        try:
            # Use the gpt function
            system_message = "You are a helpful assistant that corrects transcription errors."
            result = gpt(model=self.model, system=system_message, user=prompt)
            
            # Extract the content from the response
            if hasattr(result, 'choices') and len(result.choices) > 0:
                if hasattr(result.choices[0], 'message') and hasattr(result.choices[0].message, 'content'):
                    refined_text = result.choices[0].message.content
                else:
                    refined_text = str(result.choices[0])
            else:
                refined_text = str(result)
            
            print("Transcription successfully refined!")
            return refined_text
        
        except Exception as e:
            print(f"Error refining subtitles after multiple attempts: {str(e)}")
            print("Using original transcription instead.")
            return transcribed_text

def get_project_paths():
    """Get standard project paths based on the file location."""
    # Get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate up to root directory
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
    
    # Define paths
    dataset_dir = os.path.join(project_root, 'dataset')
    video_edit_dir = os.path.join(dataset_dir, 'video_edit')
    video_output_dir = os.path.join(video_edit_dir, 'video_output')
    writing_data_dir = os.path.join(video_edit_dir, 'writing_data')
    scene_output_dir = os.path.join(video_edit_dir, 'scene_output')
    
    # Create directories if they don't exist
    for dir_path in [video_output_dir, writing_data_dir, scene_output_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Set model path
    model_path = os.path.join(project_root, 'tools', 'whisper-large-v3-turbo')
    
    return {
        'project_root': project_root,
        'dataset_dir': dataset_dir,
        'video_edit_dir': video_edit_dir,
        'video_output_dir': video_output_dir,
        'writing_data_dir': writing_data_dir,
        'scene_output_dir': scene_output_dir,
        'model_path': model_path
    }

@retry(
    stop=stop_after_attempt(3),  # Retry up to 3 times
    wait=wait_exponential(multiplier=1, min=1, max=10),  # Exponential backoff
    retry=retry_if_exception_type((json.JSONDecodeError, IOError)),
    reraise=True
)
def extract_actual_content_from_json(json_path):
    """Extract the actual subtitle content from the video_scene.json file with retries."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract content_created field
        if "content_created" in data:
            content = data["content_created"]
            
            # Clean up the content by removing the chunk markers
            content = content.replace("/////\n", "").strip()
            
            # Further process to create clean paragraphs
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            clean_content = "\n\n".join(paragraphs)
            
            return clean_content
        else:
            print("Warning: 'content_created' field not found in JSON.")
            return ""
    
    except Exception as e:
        print(f"Error reading JSON file: {str(e)}")
        # Let tenacity handle retry
        raise

def parse_transcript_to_segments(transcript_text):
    """Parse the transcript text file into segments for adding to video."""
    segments = []
    
    # Split the transcript by double newlines to get segments
    raw_segments = transcript_text.strip().split("\n\n")
    
    for i, segment in enumerate(raw_segments):
        if not segment.strip():
            continue
            
        lines = segment.strip().split("\n")
        if len(lines) < 2:
            continue
            
        # Parse timestamp line like "0.00 --> 5.00"
        timestamp_parts = lines[0].split(" --> ")
        if len(timestamp_parts) != 2:
            continue
            
        try:
            start_time = float(timestamp_parts[0])
            end_time = float(timestamp_parts[1])
            text = lines[1].strip()
            
            segments.append({
                "id": i,
                "start": start_time,
                "end": end_time,
                "text": text
            })
        except ValueError:
            # Skip this segment if time conversion fails
            print(f"Warning: Could not parse timestamp in segment: {lines[0]}")
    
    return {"segments": segments}

def parse_refined_transcript(refined_text):
    """Parse the refined transcript from LLM into a format we can use."""
    segments = []
    
    # Split the refined text by double newlines to get segments
    raw_segments = refined_text.strip().split("\n\n")
    
    for i, segment in enumerate(raw_segments):
        if not segment.strip():
            continue
            
        lines = segment.strip().split("\n")
        if len(lines) < 2:
            continue
            
        # Try to find timestamp pattern like "0.00 --> 5.00" or "00:00:00,000 --> 00:00:05,000"
        timestamp_line = None
        content_lines = []
        
        for line in lines:
            if " --> " in line:
                timestamp_line = line
            else:
                content_lines.append(line)
        
        if not timestamp_line:
            continue
            
        # Parse the timestamp
        timestamp_parts = timestamp_line.split(" --> ")
        if len(timestamp_parts) != 2:
            continue
            
        try:
            # Try to handle both decimal format (0.00) and SRT format (00:00:00,000)
            start_str = timestamp_parts[0].strip()
            end_str = timestamp_parts[1].strip()
            
            # Convert from SRT format if needed
            if ":" in start_str:
                # Parse SRT timestamp format (HH:MM:SS,mmm)
                h, m, s = start_str.replace(',', '.').split(':')
                start_time = float(h) * 3600 + float(m) * 60 + float(s)
                
                h, m, s = end_str.replace(',', '.').split(':')
                end_time = float(h) * 3600 + float(m) * 60 + float(s)
            else:
                # Assume decimal format
                start_time = float(start_str)
                end_time = float(end_str)
                
            text = " ".join(content_lines).strip()
            
            segments.append({
                "id": i,
                "start": start_time,
                "end": end_time,
                "text": text
            })
        except ValueError as e:
            # Skip this segment if time conversion fails
            print(f"Warning: Could not parse timestamp in segment: {timestamp_line} - {e}")
    
    return {"segments": segments}

@retry(
    stop=stop_after_attempt(3),  # Retry up to 3 times
    wait=wait_exponential(multiplier=1, min=1, max=10),  # Exponential backoff
    retry=retry_if_exception_type((IOError,)),
    reraise=False  # Don't reraise, return None instead
)
def checker_agent(transcript_path, scene_json_path):
    """
    Check and refine the transcription using actual content from JSON file.
    Uses tenacity for retries on file operations.
    
    Args:
        transcript_path: Path to the transcribed subtitle file
        scene_json_path: Path to the video_scene.json file
    
    Returns:
        Dictionary with segments for refined subtitles or None if failed
    """
    print(f"Running checker agent to refine subtitles...")
    
    # Check if both files exist
    if not os.path.exists(transcript_path):
        print(f"Error: Transcript file not found: {transcript_path}")
        return None
        
    if not os.path.exists(scene_json_path):
        print(f"Error: Scene JSON file not found: {scene_json_path}")
        return None
    
    try:
        # 1. Read the transcript file
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_text = f.read()
        
        # 2. Extract actual content from JSON file (with retries via decorated function)
        actual_content = extract_actual_content_from_json(scene_json_path)
        
        if not actual_content:
            print("Warning: Could not extract actual content from JSON. Using original transcript.")
            return parse_transcript_to_segments(transcript_text)
        
        # 3. Initialize LLM client
        llm_client = LLMClient()
        
        # 4. Refine the transcript using LLM (with retries built into the client)
        refined_text = llm_client.refine_subtitles(transcript_text, actual_content)
        
        # 5. Parse the refined text into segments
        refined_segments = parse_refined_transcript(refined_text)
        
        # 6. Save the refined transcript for reference
        refined_path = transcript_path.replace('.txt', '_refined.txt')
        with open(refined_path, 'w', encoding='utf-8') as f:
            f.write(refined_text)
        
        print(f"Refined transcript saved to: {refined_path}")
        print(f"Processed {len(refined_segments['segments'])} refined subtitle segments")
        
        return refined_segments
    
    except Exception as e:
        print(f"Error in checker agent: {str(e)}")
        # Return None to indicate failure despite retries
        return None

def clean_up_temporary_files(transcript_path):
    """Clean up temporary text files after processing."""
    files_to_remove = [
        transcript_path,
        transcript_path.replace('.txt', '_refined.txt')
    ]
    
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Removed temporary file: {file_path}")
            except Exception as e:
                print(f"Warning: Could not remove file {file_path}: {e}")

def process_video(video_path=None, method="ffmpeg", clean_up=True):
    """Process a video to add subtitles based on its audio content."""
    # Get standard project paths
    paths = get_project_paths()
    
    # Set default video path if not provided
    if video_path is None:
        video_path = os.path.join(paths['video_output_dir'], 'news_output_video.mp4')
    
    video_path = os.path.abspath(video_path)
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Get video filename without extension
    video_name = Path(video_path).stem
    
    # Define output paths
    transcript_path = os.path.join(paths['writing_data_dir'], f"{video_name}_subtitle.txt")
    srt_path = f"{video_name}.srt"
    output_video_path = os.path.join(paths['video_output_dir'], f"{video_name}_subtitled.mp4")
    scene_json_path = os.path.join(paths['scene_output_dir'], 'video_scene.json')
    
    # Initialize transcriber with the model path
    transcriber = VideoTranscriber(model_path=paths['model_path'])
    
    # Step 1: Transcribe video
    result = transcriber.transcribe_video(video_path)
    
    # Step 2: Save transcript to writing_data directory
    transcriber.save_transcript(result, transcript_path)
    print(f"Transcript saved as text file: {transcript_path}")
    
    # Step 3: Use checker_agent to refine the transcript (with built-in retries)
    if os.path.exists(scene_json_path):
        print(f"Found scene JSON file. Running checker agent to refine subtitles.")
        refined_result = checker_agent(transcript_path, scene_json_path)
        
        # If checker agent was successful, use refined subtitles
        if refined_result and refined_result['segments']:
            print("Using refined subtitles for video")
            result = refined_result
    else:
        print(f"Scene JSON file not found: {scene_json_path}")
        print("Using original transcription for subtitles")
    
    # Step 4: Create SRT file in video_output directory
    transcriber.create_srt(result, srt_path)
    
    # Step 5: Add subtitles to video
    if method == "ffmpeg":
        final_video_path = transcriber.add_subtitles_with_ffmpeg(video_path, srt_path, output_video_path)
    else:
        final_video_path = transcriber.add_subtitles_to_video(video_path, result, output_video_path)
    
    # Step 6: Clean up temporary text files if requested
    if clean_up:
        print("Cleaning up temporary text files...")
        clean_up_temporary_files(transcript_path)
    
    return {
        "transcript_path": transcript_path,
        "srt_path": srt_path,
        "output_video_path": final_video_path
    }

def subtitler_main(video_path=None, output_path=None):
    """Main function to process video subtitles with optional custom output path."""
    try:
        # Set environment variable to avoid ALSA errors
        os.environ["XDG_RUNTIME_DIR"] = "/tmp/runtime-dir"
        os.makedirs(os.environ["XDG_RUNTIME_DIR"], exist_ok=True)
        
        paths = get_project_paths()
        
        # If video path not provided, use the default
        if video_path is None:
            video_path = os.path.join(paths['video_output_dir'], 'news_output_video.mp4')
        
        # Get video filename without extension
        video_name = Path(video_path).stem
        
        # Define paths
        transcript_path = os.path.join(paths['writing_data_dir'], f"{video_name}_subtitle.txt")
        srt_path = f"{video_name}.srt"
        
        # Use custom output path if provided, otherwise use default
        if output_path is None:
            output_video_path = os.path.join(paths['video_output_dir'], f"{video_name}_subtitled.mp4")
        else:
            output_video_path = output_path
            
        scene_json_path = os.path.join(paths['scene_output_dir'], 'video_scene.json')
        
        # Initialize transcriber
        transcriber = VideoTranscriber(model_path=paths['model_path'])
        
        # Transcribe video
        result = transcriber.transcribe_video(video_path)
        
        # Save transcript to writing_data directory
        transcriber.save_transcript(result, transcript_path)
        
        # Use checker_agent if possible
        if os.path.exists(scene_json_path):
            print(f"Found scene JSON file. Running checker agent to refine subtitles.")
            refined_result = checker_agent(transcript_path, scene_json_path)
            
            # If checker agent was successful, use refined subtitles
            if refined_result and refined_result['segments']:
                print("Using refined subtitles for video")
                result = refined_result
        
        # Save SRT to video_output directory
        transcriber.create_srt(result, srt_path)
        
        # Add subtitles to video using ffmpeg (it's faster and more reliable)
        final_video_path = transcriber.add_subtitles_with_ffmpeg(video_path, srt_path, output_video_path)
        
        # Clean up temporary files
        clean_up_temporary_files(transcript_path)
        
        return {
            "transcript_path": transcript_path,
            "srt_path": srt_path,
            "output_video_path": final_video_path
        }
        
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise
