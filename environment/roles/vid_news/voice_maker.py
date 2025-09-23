import sys
import os
import json
import re
import torch
import traceback
import torchaudio
from TTS.api import TTS

class Voice_Maker:
    def __init__(self):
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Navigate to the Vtube root directory (two levels up from agents)
        self.parent_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
        
        # Set up paths
        self.video_edit_dir = os.path.join(self.parent_root, 'dataset', 'video_edit')
        self.voice_data_dir = os.path.join(self.video_edit_dir, 'voice_data')
        self.scene_output_dir = os.path.join(self.video_edit_dir, 'scene_output')
        self.voice_gen_dir = os.path.join(self.video_edit_dir, 'voice_gen')
        
        # Initialize Coqui TTS to None
        self.tts = None
        self.prompt_speech_path = None

    def process_with_timestamps(self, json_file_path):
        """Process JSON file and extract segments with proper support for Chinese content"""
        # Read the JSON file
        with open(json_file_path, 'r', encoding='utf-8') as file:  # Ensure UTF-8 encoding
            json_data = json.load(file)
        
        # Get the raw content - check both possible field names
        raw_content = None
        if "content_created" in json_data:
            raw_content = json_data["content_created"]
            print("Using 'content_created' field from JSON")
        else:
            print("Error: Neither 'content_created' nor 'storyboard' field found in JSON file")
            return []
        
        # Normalize line endings
        raw_content = raw_content.replace('\r\n', '\n')
        
        # Check for the exact delimiter pattern from your example
        if '/////\n' in raw_content:
            segments = raw_content.split('/////\n')
            print("Using exact '/////\n' delimiter pattern")
        else:
            # Fallback to more generic regex pattern
            segments = re.split(r'/+\s*\n', raw_content)
            print("Using regex pattern for delimiter detection")
        
        # Filter out empty segments and strip whitespace
        segments = [seg.strip() for seg in segments if seg.strip()]
        
        # Create a list to store both full content and individual segments
        segment_list = []
        for i, segment in enumerate(segments):
            segment_list.append({
                "segment_id": i+1,
                "content": segment
            })
        
        # Create a new object with the full content and segments
        clean_json = {
            "user_idea": json_data.get("user_idea", ""),
            "segments": segment_list
        }
        
        # Save to a new file with UTF-8 encoding
        output_path = json_file_path.replace(".json", "_clean.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(clean_json, f, indent=2, ensure_ascii=False)  # ensure_ascii=False to preserve Chinese characters
        
        print(f"Original content had {len(segments)} segments separated by delimiters")
        print(f"Clean content saved to {output_path}")
        
        return segment_list

    def split_into_sentences(self, text, max_length=200):  # Increased max_length for Chinese
        """Split text into manageable chunks for TTS processing with Chinese support"""
        # Chinese sentences typically use different punctuation
        for punct in ['。', '！', '？', '；', '. ', '! ', '? ', '; ']:
            text = text.replace(punct, punct + '|')
        
        sentences = text.split('|')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Further split long sentences
        result = []
        for sentence in sentences:
            if len(sentence) <= max_length:
                result.append(sentence)
            else:
                # Split by commas and Chinese commas if sentence is too long
                comma_parts = sentence.replace('，', ',').split(',')
                current_part = ""
                
                for part in comma_parts:
                    if len(current_part) + len(part) <= max_length:
                        if current_part:
                            current_part += "," + part
                        else:
                            current_part = part
                    else:
                        if current_part:
                            result.append(current_part)
                        current_part = part
                
                if current_part:
                    result.append(current_part)
        
        return result

    def generate_audio_for_segments(self, segments, output_dir=None):
        """Generate audio for each segment and track timestamps with Chinese support"""
        if output_dir is None:
            output_dir = self.voice_gen_dir
            
        # Make sure output_dir exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp_data = {
            "sentence_data": {
                "count": len(segments),
                "chunks": []
            }
        }
        
        current_time = 0  # Running timestamp in seconds
        all_segment_waveforms = []
        all_files_to_delete = []  # Track all files for cleanup
        
        # Process each segment
        for segment in segments:
            segment_id = segment["segment_id"]
            segment_text = segment["content"]
            segment_output_file = f"{output_dir}/segment_{segment_id}.wav"
            all_files_to_delete.append(segment_output_file)  # Add to list for cleanup
            
            print(f"\nProcessing Segment {segment_id}:")
            # For Chinese, we display fewer characters in preview
            text_preview = segment_text[:50] + "..." if len(segment_text) > 50 else segment_text
            print(f"Text preview: {text_preview}")
            
            # Skip any segments that still contain the separator pattern
            if "//////" in segment_text:
                print(f"Skipping segment {segment_id} as it appears to be a separator")
                continue
                
            # Split segment into sentences/chunks for processing
            sentences = self.split_into_sentences(segment_text)
            print(f"Split into {len(sentences)} chunks for processing")
            
            # Skip if no valid sentences
            if not sentences:
                print(f"No valid sentences found in segment {segment_id}, skipping")
                continue
            
            # Track segment start time
            segment_start_time = current_time
            segment_waveforms = []
            
            # Process each sentence individually
            for i, sentence in enumerate(sentences):
                try:
                    # For Chinese, we display fewer characters in preview
                    sentence_preview = sentence[:30] + "..." if len(sentence) > 30 else sentence
                    print(f"  Processing chunk {i+1}/{len(sentences)}: '{sentence_preview}'")
                    
                    # Generate audio using Coqui TTS
                    # For Tacotron2 model, we don't need language or speaker_wav parameters
                    if hasattr(self.tts, 'is_multi_lingual') and self.tts.is_multi_lingual:
                        wav = self.tts.tts(
                            text=sentence,
                            speaker_wav=self.prompt_speech_path,
                            language="en",
                            split_sentences=False
                        )
                    else:
                        # For single-language models like Tacotron2
                        wav = self.tts.tts(
                            text=sentence,
                            split_sentences=False
                        )
                    
                    # Convert to tensor if needed
                    if isinstance(wav, list):
                        wav_tensor = torch.tensor(wav).unsqueeze(0)
                    else:
                        wav_tensor = wav.unsqueeze(0) if wav.dim() == 1 else wav
                    
                    segment_waveforms.append(wav_tensor)
                    print(f"    Successfully processed chunk {i+1}")
                    
                except Exception as e:
                    print(f"  Error processing chunk {i+1}: {str(e)}")
                    print("  Continuing to next chunk...")
            
            # Combine all waveforms for this segment
            if segment_waveforms:
                segment_waveform = torch.cat(segment_waveforms, dim=1)
                
                # Save the complete segment audio file
                torchaudio.save(segment_output_file, segment_waveform, 24000)  # XTTS uses 24kHz sample rate
                
                # Calculate segment duration
                segment_duration = len(segment_waveform[0]) / 24000  # XTTS uses 24kHz sample rate
                
                # Update the current time
                current_time += segment_duration
                
                # Add to timestamp data
                timestamp_data["sentence_data"]["chunks"].append({
                    "id": segment_id,
                    "timestamp": round(current_time, 3),
                    "content": segment_text
                })
                
                # Store the waveform for later concatenation
                all_segment_waveforms.append(segment_waveform)
                
                print(f"Successfully processed segment {segment_id} (duration: {segment_duration:.2f}s)")
            else:
                print(f"Warning: No audio generated for segment {segment_id}")
        
        # Cleanup any chunk files that may have been created previously
        chunk_files = [f for f in os.listdir(output_dir) if f.startswith("segment_") and "_chunk_" in f]
        for chunk_file in chunk_files:
            all_files_to_delete.append(os.path.join(output_dir, chunk_file))
        
        # Return timestamp data, all waveforms, and files for cleanup
        return timestamp_data, all_segment_waveforms, 24000, all_files_to_delete

    def combine_audio_files(self, all_segment_waveforms, sample_rate, timestamp_data, 
                            output_file=None, segment_files=None):
        """
        Combine all segment waveforms into one and save as WAV file
        Also save timestamp data to JSON and delete all intermediate files
        """
        if not all_segment_waveforms:
            print("No audio segments to combine")
            return None, None
        
        # Combine all waveforms
        combined_waveform = all_segment_waveforms[0]
        for waveform in all_segment_waveforms[1:]:
            combined_waveform = torch.cat([combined_waveform, waveform], dim=1)
        
        # Export combined audio as WAV
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save the combined waveform
        torchaudio.save(output_file, combined_waveform, sample_rate)
        print(f"Combined audio saved to: {output_file}")
        
        # Save timestamp JSON with UTF-8 encoding to preserve Chinese characters
        timestamp_json_file = output_file.replace(".wav", "_timestamps.json")
        with open(timestamp_json_file, 'w', encoding='utf-8') as f:
            json.dump(timestamp_data, f, indent=2, ensure_ascii=False)  # ensure_ascii=False preserves Chinese characters
        print(f"Timestamp data saved to: {timestamp_json_file}")
        
        # Clean up all files - both segment files and chunk files
        if segment_files:
            print("Cleaning up temporary files...")
            deleted_count = 0
            for file_path in segment_files:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        deleted_count += 1
                except Exception as e:
                    print(f"Warning: Could not remove {file_path}: {str(e)}")
            print(f"Removed {deleted_count} temporary files")
        
        return output_file, timestamp_json_file

    def initialize_model(self):
        """Initialize the Coqui TTS model"""
        print(f"Ensured all necessary directories exist in: {self.video_edit_dir}")
        
        try:
            # Initialize Coqui TTS with VITS model for better performance and robustness
            print("Loading Coqui TTS VITS model...")
            self.tts = TTS("tts_models/en/ljspeech/vits")
            
            # For this model, we don't need a prompt speech file
            self.prompt_speech_path = None
            print("Using VITS model (fast and robust)")
            return True
            
        except Exception as e:
            print(f"Error initializing Coqui TTS model: {e}")
            traceback.print_exc()
            return False

    def generate_voice(self):
        """Main method to generate voice from JSON content"""
        try:
            # Initialize the model
            if not self.initialize_model():
                return False
            
            # Get content from JSON
            json_file_path = os.path.join(self.scene_output_dir, 'video_scene.json')
            if not os.path.exists(json_file_path):
                print(f"Warning: JSON file not found at {json_file_path}")
                print("Please ensure you have the scene JSON file in the scene_output directory.")
                return False
            
            print(f"Processing content from: {json_file_path}")
            segments = self.process_with_timestamps(json_file_path)
            print(f"Found {len(segments)} segments to process")
            
            if not segments:
                print("No valid segments found. Please check the JSON file format.")
                return False
            
            # Generate audio with timestamp tracking
            timestamp_data, all_segment_waveforms, sample_rate, files_to_delete = self.generate_audio_for_segments(
                segments, output_dir=self.voice_gen_dir
            )
            
            if not all_segment_waveforms:
                print("No audio segments were successfully generated.")
                return False
            
            # Combine all segments, save timestamp JSON, and delete all intermediate files
            output_audio_path = os.path.join(self.voice_gen_dir, 'gen_news_audio.wav')
            final_audio_path, timestamp_json_path = self.combine_audio_files(
                all_segment_waveforms, 
                sample_rate, 
                timestamp_data, 
                output_file=output_audio_path,
                segment_files=files_to_delete
            )
            
            print(f"Audio successfully generated and saved to: {output_audio_path}")
            return {
                "audio_file": final_audio_path,
                "timestamp_file": timestamp_json_path,
                "status": "success"
            }
        
        except Exception as e:
            print(f"An error occurred in the voice generation process: {str(e)}")
            traceback.print_exc()
            return {
                "error": str(e),
                "status": "error"
            }

# Add this function to match what's imported in comm_agent.py
def voice_main():
    """Function that's called from CommAgent to generate voice using Coqui TTS"""
    print("\n=== GENERATING VOICE WITH COQUI TTS ===")
    voice_maker = Voice_Maker()
    result = voice_maker.generate_voice()
    
    if isinstance(result, bool):
        # Convert old-style boolean return to new dict format
        if result:
            return {
                "audio_file": os.path.join(voice_maker.voice_gen_dir, 'gen_news_audio.wav'),
                "timestamp_file": os.path.join(voice_maker.voice_gen_dir, 'gen_news_audio_timestamps.json'),
                "status": "success"
            }
        else:
            return {
                "error": "Voice generation failed",
                "status": "error"
            }
    
    return result