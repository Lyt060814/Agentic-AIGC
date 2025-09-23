import os
import logging
import warnings
import multiprocessing
import json
import importlib.util
import sys

class Video_Searcher:
    def __init__(self):
        # Configure logging and warnings
        warnings.filterwarnings("ignore")
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.logger = logging.getLogger(__name__)
        
        # Set up paths
        self._setup_paths()
        
        # Import dependencies after path setup
        self._import_dependencies()
    
    def _setup_paths(self):
        """Set up necessary paths and directories"""
        # Get project root based on file location
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
        
        # Define paths
        self.dataset_dir = os.path.join(project_root, 'dataset')
        self.video_edit_dir = os.path.join(self.dataset_dir, 'video_edit')
        self.scene_output_dir = os.path.join(self.video_edit_dir, 'scene_output')
        self.scene_output_path = os.path.join(self.scene_output_dir, 'video_scene.json')
        self.working_dir = os.path.join(self.video_edit_dir, 'videosource-workdir')
        
        # Add tools directory to path
        tools_dir = os.path.join(project_root, 'tools')
        if tools_dir not in sys.path:
            sys.path.append(tools_dir)
    
    def _import_dependencies(self):
        """Import dependencies that require specific path setup"""
        try:
            from videorag.videoragcontent import VideoRAG, QueryParam
            self.VideoRAG = VideoRAG
            self.QueryParam = QueryParam
        except ImportError as e:
            self.logger.error(f"Failed to import VideoRAG: {e}")
            raise
    
    def process_scene(self):
        """
        Process a scene from JSON and use VideoRAG to search for matching content
        
        Returns:
            The response from VideoRAG query
        """
        try:
            with open(self.scene_output_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
            # Extract the content
            segment_scene = data.get("segment_scene", "")
            
            if not segment_scene:
                self.logger.warning("Empty segment_scene found in the JSON file")
                
            # Use the content as query - this contains all scenes separated by /////
            query = f'''{segment_scene}'''
            
            self.logger.info(f"Using query: {query}")
            
            param = self.QueryParam(mode="videoragcontent")
            # if param.wo_reference = False, VideoRAG will add reference to video clips in the response
            param.wo_reference = True
            
            videoragcontent = self.VideoRAG(
                working_dir=self.working_dir
            )
            
            response = videoragcontent.query(query=query, param=param)
            self.logger.info("VideoRAG query completed successfully")
            
            # Check if response has any visual segments
            # If response is already structured (from fallback), return it directly
            if response and len(response) > 0 and isinstance(response[0], dict):
                # Already structured format, no need to convert
                return response
            elif response and len(response) > 0:
                # Convert segment IDs to structured format
                response = self._convert_segments_to_structured_format(response)
            
            return response
            
        except FileNotFoundError:
            self.logger.error(f"Error: JSON file not found at {self.scene_output_path}")
            raise
        except json.JSONDecodeError:
            self.logger.error("Error: Invalid JSON format in the file.")
            raise
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}")
            raise
    
    def _convert_segments_to_structured_format(self, segment_ids):
        """
        Convert segment IDs to structured format required by video editor
        
        Args:
            segment_ids: List of segment IDs like ['output_0'] or ['20250909_121908_18']
            
        Returns:
            List of structured segment dictionaries
        """
        structured_segments = []
        
        for segment_id in segment_ids:
            # Parse segment ID: Could be 'output_0' or '20250909_121908_18'
            parts = segment_id.split("_")
            if len(parts) >= 2:
                try:
                    # Handle both formats:
                    # 1. '20250909_121908_18' format - parts = ["20250909", "121908", "18"]
                    # 2. 'output_0' format - parts = ["output", "0"]
                    if len(parts) >= 3:
                        # Format: 20250909_121908_18
                        video_name = f"{parts[0]}_{parts[1]}"  # "20250909_121908"
                        segment_num = int(parts[2])  # 18
                    else:
                        # Format: output_0
                        video_name = parts[0]  # "output"
                        segment_num = int(parts[1])  # 0
                    
                    # Assume each segment is 30 seconds (this should match the actual segment duration)
                    start_time = segment_num * 30
                    end_time = (segment_num + 1) * 30
                    
                    structured_segments.append({
                        "video_name": video_name,
                        "segment_id": segment_num,
                        "start_time": start_time,
                        "end_time": end_time,
                        "description": f"Segment {segment_num} from {video_name}"
                    })
                except ValueError:
                    self.logger.warning(f"Could not parse segment number from {segment_id}")
                    continue
            else:
                self.logger.warning(f"Invalid segment ID format: {segment_id}")
                continue
                
        return structured_segments

    def _create_fallback_segments(self):
        """
        Create fallback video segments when VideoRAG doesn't find any matches
        Uses available video segments from the processed video files
        """
        try:
            # Look for available video segment data
            videosource_workdir = os.path.join(self.working_dir, "videosource-workdir")
            
            # Try to find video segment files
            fallback_segments = []
            if os.path.exists(videosource_workdir):
                # Look for video segment JSON files
                for filename in os.listdir(videosource_workdir):
                    if filename.startswith("kv_store_video_segments") and filename.endswith(".json"):
                        segment_file = os.path.join(videosource_workdir, filename)
                        try:
                            with open(segment_file, 'r', encoding='utf-8') as f:
                                segments_data = json.load(f)
                                
                            # Extract video name from filename
                            video_name = filename.replace("kv_store_video_segments_", "").replace(".json", "")
                            
                            # Create fallback segments using available segments
                            segment_count = 0
                            for key, segment_info in segments_data.items():
                                if segment_count >= 10:  # Limit to 10 segments
                                    break
                                    
                                fallback_segment = {
                                    "video_name": video_name,
                                    "segment_id": segment_count,
                                    "start_time": segment_count * 30,  # 30 second intervals
                                    "end_time": (segment_count + 1) * 30,
                                    "description": f"Product demo segment {segment_count + 1}"
                                }
                                fallback_segments.append(fallback_segment)
                                segment_count += 1
                                
                        except Exception as e:
                            self.logger.error(f"Error reading segment file {segment_file}: {e}")
                            continue
            
            # If no segments found, create basic fallback
            if not fallback_segments:
                self.logger.info("Creating basic fallback segments")
                for i in range(5):  # Create 5 basic segments
                    fallback_segment = {
                        "video_name": "20250909_121908",  # Default video name
                        "segment_id": i,
                        "start_time": i * 30,
                        "end_time": (i + 1) * 30,
                        "description": f"Product demo segment {i + 1}"
                    }
                    fallback_segments.append(fallback_segment)
            
            self.logger.info(f"Created {len(fallback_segments)} fallback segments")
            return fallback_segments
            
        except Exception as e:
            self.logger.error(f"Error creating fallback segments: {e}")
            return []
    
    def run(self):
        """
        Main entry point for Video_Searcher
        """
        self.logger.info("Starting Video_Searcher")
        
        # Initialize multiprocessing with spawn method - use get_context instead
        # Modified to handle the "context already set" error
        
        try:
            if multiprocessing.get_start_method(allow_none=True) != 'spawn':
                multiprocessing.set_start_method('spawn')
        except RuntimeError:
            # If context already set, just use the current context
            self.logger.info("Multiprocessing context already set, using current context")
        
        # Get response from VideoRAG
        response = self.process_scene()
        print(response)
        
        # Ensure we save the response to the visual_retrieved_segments.json file
        visual_segments_file = os.path.join(self.scene_output_dir, "visual_retrieved_segments.json")
        try:
            with open(visual_segments_file, 'w', encoding='utf-8') as f:
                json.dump(response, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved {len(response)} visual segments to {visual_segments_file}")
        except Exception as e:
            self.logger.error(f"Error saving visual segments: {e}")
        
        # Run the vid_editer
        #from environment.roles.vid_comm.vid_editer import main as vid_editer_main
        #vid_editer_main()
        
        self.logger.info("Video_Searcher completed")
        return response

def video_search_main():
    """
    Convenience function to create and run a Video_Searcher
    """
    logging.basicConfig(level=logging.INFO)
    searcher = Video_Searcher()
    return searcher.run()

