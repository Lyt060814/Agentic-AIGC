import os
import torch
import logging
import glob
import tempfile
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from moviepy.editor import VideoFileClip


class transcribe_main:
    def __init__(self, model_id=None, use_timestamps=False, chunk_length_s=30, batch_size=16, config=None):
        """
        Initialize the TranscribeMain class for speech recognition of videos.
        
        Args:
            model_id (str, optional): Specific model ID to use. Defaults to None (uses whisper-large-v3-turbo).
            use_timestamps (bool, optional): Whether to include timestamps in output. Defaults to False.
            chunk_length_s (int, optional): Length of audio chunks in seconds. Defaults to 30.
            batch_size (int, optional): Batch size for processing. Defaults to 16.
            config (dict, optional): Configuration dictionary that may contain custom paths. Defaults to None.
        """
        # Set up logging
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Navigate to the root directory
        self.parent_root = os.path.abspath(os.path.join(current_dir, '../../../'))
        
        # Define paths for video source and transcribe data
        self.video_edit_dir = os.path.join(self.parent_root, 'dataset', 'video_edit')
        
        # Set video_source_dir from config if provided, otherwise use default
        if config and "video_source_dir" in config:
            self.video_source_dir = config["video_source_dir"]
            self.logger.info(f"Using custom video source directory from config: {self.video_source_dir}")
        else:
            self.video_source_dir = os.path.join(self.video_edit_dir, 'video_source')
            
        self.transcribe_data_dir = os.path.join(self.video_edit_dir, 'writing_data')
        
        # Create output directory if it doesn't exist
        os.makedirs(self.transcribe_data_dir, exist_ok=True)
        os.makedirs(self.video_source_dir, exist_ok=True)  # Also ensure video source dir exists
        
        # Set up device for processing
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Save configuration parameters
        self.model_id = model_id
        self.use_timestamps = use_timestamps
        self.chunk_length_s = chunk_length_s
        self.batch_size = batch_size
        
        # Initialize whisper model
        self._initialize_whisper_model()
        
        self.logger.info(f"TranscribeMain initialized. Device: {self.device}")
        self.logger.info(f"Video source directory: {self.video_source_dir}")
        self.logger.info(f"Transcript output directory: {self.transcribe_data_dir}")


    
    def _initialize_whisper_model(self):
        """Initialize the whisper speech recognition model."""
        # Try to find local model in tools directory
        tools_dir = os.path.join(self.parent_root, 'tools')
        default_model = "whisper-large-v3-turbo"
        model_path = os.path.join(tools_dir, default_model)
        
        # Use provided model_id if available, otherwise use default
        model_id = self.model_id or default_model
        
        # Check if the model directory exists locally
        if os.path.exists(model_path) and not self.model_id:
            model_id = model_path
            self.logger.info(f"Using local whisper model from: {model_path}")
            local_files_only = True
        else:
            if self.model_id:
                self.logger.info(f"Using specified model: {model_id}")
            else:
                self.logger.warning(f"Local model not found at {model_path}, loading {model_id} from Hugging Face")
            local_files_only = False
        
        # Load the model
        try:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, 
                torch_dtype=self.torch_dtype,
                use_safetensors=True,
                local_files_only=local_files_only
            )
            self.model.to(self.device)
            
            # Load the processor
            self.processor = AutoProcessor.from_pretrained(
                model_id, 
                local_files_only=local_files_only
            )
            
            # Create the pipeline
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                max_new_tokens=128,
                chunk_length_s=self.chunk_length_s,
                batch_size=self.batch_size,
                return_timestamps=self.use_timestamps,
                torch_dtype=self.torch_dtype,
                device=self.device,
            )
            
            self.logger.info("Whisper model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading whisper model: {str(e)}")
            raise
    
    def _extract_audio(self, video_path):
        """
        Extract audio from video file and save as temporary WAV file.
        
        Args:
            video_path (str): Path to the video file.
            
        Returns:
            str: Path to the temporary audio file.
        """
        self.logger.info(f"Extracting audio from video: {os.path.basename(video_path)}")
        
        try:
            # Create a temporary file for the audio
            temp_audio_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_audio_path = temp_audio_file.name
            temp_audio_file.close()
            
            # Extract audio using moviepy
            with VideoFileClip(video_path) as video:
                if video.audio is not None:
                    video.audio.write_audiofile(temp_audio_path, codec='pcm_s16le', verbose=False, logger=None)
                    self.logger.info(f"Audio extracted successfully to: {temp_audio_path}")
                    return temp_audio_path
                else:
                    self.logger.error(f"No audio track found in video: {video_path}")
                    # Clean up temporary file
                    os.unlink(temp_audio_path)
                    return None
        except Exception as e:
            self.logger.error(f"Error extracting audio from {video_path}: {str(e)}")
            # Clean up temporary file if it exists
            if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            return None
    
    def transcribe_video(self, video_path, force_overwrite=False):
        """
        Transcribe a single video file and save the transcription.
        
        Args:
            video_path (str): Path to the video file.
            force_overwrite (bool, optional): Whether to overwrite existing transcripts. Defaults to False.
            
        Returns:
            str: Path to the saved transcript file, or None if transcription failed.
        """
        # Get the base filename without extension
        base_name = os.path.basename(video_path)
        file_name = os.path.splitext(base_name)[0]
        output_path = os.path.join(self.transcribe_data_dir, f"audio_transcript.txt")
        
        # Skip if transcript already exists and not forcing overwrite
        if os.path.exists(output_path) and not force_overwrite:
            self.logger.info(f"Transcript already exists for {base_name}, skipping.")
            return output_path
        
        self.logger.info(f"Transcribing video: {base_name}")
        
        try:
            # Extract audio from video first
            audio_path = self._extract_audio(video_path)
            if audio_path is None:
                self.logger.error(f"Failed to extract audio from {base_name}")
                return None
            
            # Run speech recognition on the extracted audio
            result = self.pipe(audio_path)
            
            # Clean up temporary audio file
            os.unlink(audio_path)
            
            # Extract the text
            transcript = result.get("text", "")
            
            # Write the transcript to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(transcript)
            
            self.logger.info(f"Transcription saved to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error transcribing {base_name}: {str(e)}")
            return None
    
    def run(self, file_extensions=None, force_overwrite=False):
        """
        Main execution method to transcribe all videos in the source directory.
        
        Args:
            file_extensions (list, optional): List of file extensions to process. Defaults to ['.mp4', '.MP4'].
            force_overwrite (bool, optional): Whether to overwrite existing transcripts. Defaults to False.
            
        Returns:
            dict: Mapping of video paths to transcript paths.
        """
        # Default file extensions if none provided
        if file_extensions is None:
            file_extensions = ['.mp4', '.MP4', '.avi', '.mov', '.mkv']
        
        # Get all video files from the source directory
        video_files = []
        for ext in file_extensions:
            # Remove the dot if included
            ext = ext if ext.startswith('.') else f'.{ext}'
            video_files.extend(glob.glob(os.path.join(self.video_source_dir, f"*{ext}")))
        
        if not video_files:
            self.logger.warning(f"No video files found in {self.video_source_dir}")
            return {}
        
        self.logger.info(f"Found {len(video_files)} videos to process")
        
        # Transcribe all videos
        results = {}
        for video_path in tqdm(video_files, desc="Transcribing videos"):
            transcript_path = self.transcribe_video(video_path, force_overwrite)
            results[video_path] = transcript_path
        
        # Log summary
        successful_transcriptions = len([r for r in results.values() if r is not None])
        self.logger.info(f"Transcription complete: {successful_transcriptions}/{len(video_files)} videos processed successfully")
        
        return results



