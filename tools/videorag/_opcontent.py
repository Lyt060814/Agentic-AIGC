import re
import json
import os
import torch
from sentence_transformers import SentenceTransformer

from .base import (
    QueryParam
)



async def _refine_visual_sentence_segmentation_retrieval_query(
    query,
    query_param: QueryParam,
    global_config: dict,
):
    # Split the text by the section symbol '/////'
    # This pattern matches the separator as shown in the examples
    segments = re.split(r'/////', query)
    
    # Remove empty segments and strip whitespace
    segments = [segment.strip() for segment in segments if segment.strip()]
    
    # Join the segments with semicolons
    return segments

def compute_cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors using PyTorch"""
    # Convert to torch tensors if they aren't already
    if not isinstance(vec1, torch.Tensor):
        vec1 = torch.tensor(vec1, dtype=torch.float32)
    if not isinstance(vec2, torch.Tensor):
        vec2 = torch.tensor(vec2, dtype=torch.float32)
    
    # Normalize vectors
    vec1_normalized = vec1 / vec1.norm()
    vec2_normalized = vec2 / vec2.norm()
    
    # Compute cosine similarity
    similarity = torch.dot(vec1_normalized, vec2_normalized).item()
    return similarity

def compute_text_similarity(text1, text2):
    """Compute cosine similarity between two text strings"""
    # Generate embeddings
    # Load sentence embedding model
    current_dir = os.getcwd()
    model_id = os.path.join(current_dir, 'tools/all-MiniLM-L6-v2')
    sentence_model = SentenceTransformer(model_id)
    embedding1 = sentence_model.encode(text1, show_progress_bar=False)
    embedding2 = sentence_model.encode(text2, show_progress_bar=False)
    
    # Compute cosine similarity
    similarity = compute_cosine_similarity(embedding1, embedding2)
    return similarity

def create_fallback_segments_from_kv(kvdata, used_segments):
    """Create fallback segments from available KV data when VideoRAG fails"""
    available_segments = []
    
    for movie_id, segments in kvdata.items():
        for segment_id, segment_data in segments.items():
            segment_key = f"{movie_id}_{segment_id}"
            if segment_key not in used_segments:
                available_segments.append(segment_key)
    
    # Sort to ensure consistent ordering
    available_segments.sort()
    return available_segments

async def videorag_query(
    query,
    video_segment_feature_vdb,
    query_param: QueryParam,
    global_config: dict,
) -> str:
    query = query
    
    # visual retrieval
    scene_sentences = await _refine_visual_sentence_segmentation_retrieval_query(
        query,
        query_param,
        global_config,
    )

    visual_retrieved_segments = []  # List to store segments for each scene
    used_segments = set()  # Set to track already used segments across all scenes

    # Get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate up 2 levels to reach the Vtube root directory
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

    # Define paths
    dataset_dir = os.path.join(project_root, 'dataset')
    video_edit_dir = os.path.join(dataset_dir, 'video_edit')
    scene_output_dir = os.path.join(video_edit_dir, 'scene_output')
    working_dir = os.path.join(video_edit_dir, 'videosource-workdir')

    # Replace the existing file operations with these:
    with open(os.path.join(scene_output_dir, 'textual_segmentations.json'), 'w', encoding ='utf-8') as f:
        json.dump(scene_sentences, f)

    with open(os.path.join(working_dir, 'kv_store_video_segments.json'), 'r', encoding ='utf-8') as file:
        kvdata = json.load(file)

    # Calculate segments to exclude for each movie source
    movie_segments_info = {}
    segment_duration = 30  # Default segment duration of 30 seconds

    # Try to determine segment duration from the first movie's first segment
    if kvdata:
        for movie_name, segments in kvdata.items():
            if segments and "0" in segments:
                # Get the time string (e.g., "0-30" or "0-10")
                time_str = segments["0"].get("time", "")
                if time_str and "-" in time_str:
                    start, end = time_str.split("-")
                    try:
                        # Calculate duration from the difference
                        segment_duration = int(end) - int(start)
                        print(f"Detected segment duration: {segment_duration} seconds")
                        break
                    except ValueError:
                        print(f"Could not parse time range '{time_str}', using default duration of 30 seconds")
    
    # Process each movie source
    for movie_id, segments in kvdata.items():
        # Get number of segments for this movie
        num_segments = len(segments)
        
        # Calculate total runtime for this specific movie
        total_runtime = segment_duration * num_segments
        
        # Calculate skip ranges for this movie 
        start_credits_end = max(1, int((total_runtime * 0.1) // segment_duration))
        end_credits_start = min(num_segments - 1, int((total_runtime * 0.90) // segment_duration))
        
        # Store the valid segment range for this video
        movie_segments_info[movie_id] = {
            "total_segments": num_segments,
            "valid_range": (start_credits_end, end_credits_start)
        }
    
    print("Opening/Ending video segments have been filtered for each source video")
    print("Searching for videos...")

    # Process each scene sentence
    for scene in scene_sentences:
        try:
            # Query video segments based on scene description
            segment_results = await video_segment_feature_vdb.query(scene)
            
            if not segment_results:
                print(f"Warning: No segments found for scene: {scene[:100]}...")
                # Create fallback segments when query returns empty
                fallback_segments = create_fallback_segments_from_kv(kvdata, used_segments)
                if fallback_segments:
                    visual_retrieved_segments.append([fallback_segments[0]])
                    used_segments.add(fallback_segments[0])
                else:
                    visual_retrieved_segments.append([])
                continue
        except Exception as e:
            print(f"Error querying VideoRAG for scene '{scene[:50]}...': {str(e)}")
            # Fallback to using available segments when query fails
            fallback_segments = create_fallback_segments_from_kv(kvdata, used_segments)
            if fallback_segments:
                visual_retrieved_segments.append([fallback_segments[0]])
                used_segments.add(fallback_segments[0])
            else:
                visual_retrieved_segments.append([])
            continue
            
        # Get top-k results (getting more to have options)
        top_results = segment_results[:20]
        
        # Filter out invalid segments and already used segments
        valid_results = []
        
        for result in top_results:
            segment_id = result['__id__']
            
            # Skip if segment has already been used in previous scenes
            if segment_id in used_segments:
                continue
            
            # Parse segment ID to extract movie_id and segment_num
            # Handle both formats: '20250909_121908_0' and 'output_0'
            parts = segment_id.split("_")
            
            # Check if segment ID is valid and within valid range for its source movie
            is_valid_segment = False
            
            if len(parts) >= 2:  # At least 'output_0' format
                try:
                    # Handle both formats:
                    # 1. '20250909_121908_18' format - parts = ["20250909", "121908", "18"]
                    # 2. 'output_0' format - parts = ["output", "0"]
                    if len(parts) >= 3:
                        # Format: 20250909_121908_0
                        movie_id = f"{parts[0]}_{parts[1]}"  # "20250909_121908"
                        segment_num = int(parts[2])  # 0, 1, 2, etc.
                    else:
                        # Format: output_0
                        movie_id = parts[0]  # "output"
                        segment_num = int(parts[1])  # 0, 1, 2, etc.
                    
                    is_valid_format = True
                except ValueError:
                    is_valid_format = False
                    
                # Check if the segment is within the valid range for its movie
                if is_valid_format and movie_id in movie_segments_info:
                    valid_range = movie_segments_info[movie_id]["valid_range"]
                    if valid_range[0] <= segment_num <= valid_range[1]:
                        is_valid_segment = True
                        print(f"Valid segment: {segment_id} (range: {valid_range})")
                    else:
                        print(f"Segment {segment_id} outside valid range {valid_range}")
                else:
                    if not is_valid_format:
                        print(f"Invalid segment format: {segment_id}")
                    else:
                        print(f"Movie {movie_id} not found in movie_segments_info")
                        print(f"Available movies: {list(movie_segments_info.keys())}")
            else:
                print(f"Segment ID has insufficient parts: {segment_id}")
            
            # If segment is valid, add to valid results
            if is_valid_segment:
                valid_results.append(result)
        
        if not valid_results:
            # No valid segments found
            visual_retrieved_segments.append([])
            continue
        
        # Calculate cosine similarity for each valid segment with the scene description
        segment_similarities = []
        
        for result in valid_results:
            segment_id = result['__id__']
            
            # Extract the movie_id and segment_num from segment_id
            # Handle both formats: '20250909_121908_0' and 'output_0'
            parts = segment_id.split("_")
            if len(parts) >= 3:  # Format: 20250909_121908_0
                movie_id = f"{parts[0]}_{parts[1]}"  # "20250909_121908"
                segment_num = parts[2]  # "0", "1", "2", etc.
            elif len(parts) >= 2:  # Format: output_0
                movie_id = parts[0]  # "output"
                segment_num = parts[1]  # "0", "1", "2", etc.
            else:
                continue  # Skip if format doesn't match
            
            # Get the segment content from kv_store_video_segments.json
            if movie_id in kvdata and segment_num in kvdata[movie_id]:
                segment_content = kvdata[movie_id][segment_num].get("content", "")
                
                # Compute similarity between scene description and segment content
                similarity = compute_text_similarity(scene, segment_content)
                segment_similarities.append((segment_id, similarity))
        
        # Sort segments by similarity score in descending order
        segment_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Select the best segment for this scene (1:1 mapping with audio segments)
        if segment_similarities:
            # Take only the top 1 segment to ensure 1:1 mapping with audio segments
            best_segment_id, best_similarity = segment_similarities[0]
            if best_segment_id not in used_segments:  # Avoid duplicates
                visual_retrieved_segments.append([best_segment_id])
                used_segments.add(best_segment_id)
            else:
                # If best segment already used, try next best unused segment
                found_unused = False
                for segment_id, similarity in segment_similarities[1:]:
                    if segment_id not in used_segments:
                        visual_retrieved_segments.append([segment_id])
                        used_segments.add(segment_id)
                        found_unused = True
                        break
                if not found_unused:
                    visual_retrieved_segments.append([])
        else:
            visual_retrieved_segments.append([])
    
    query_for_visual_retrieval = scene_sentences
    





    

    # Flatten the list - take exactly 1 segment per scene for 1:1 mapping with audio
    # Fix: Only include non-empty segments in the flattened list
    visual_retrieved_segments = [
        segment 
        for segments in visual_retrieved_segments 
        if segments  # Only process non-empty lists
        for segment in segments[:1]  # Take exactly 1 segment per scene
    ]




    print(f"Number of scene sentences: {len(scene_sentences)}")
    for i, (scene, segments) in enumerate(zip(scene_sentences, visual_retrieved_segments)):
        print(f"Scene {i+1}: {scene}")
        print(f"Retrieved Visual Segment: {segments}")

    # Combine all segments while maintaining uniqueness
    retrieved_segments = list(set(visual_retrieved_segments))



    # Sort the segments if needed
    retrieved_segments = sorted(
        retrieved_segments,
        key=lambda x: (
            '_'.join(x.split('_')[:-1]),  # video_name
            eval(x.split('_')[-1])  # index
        )
    )
    

    print(query_for_visual_retrieval)
    print(f"Retrieved Visual Segments {visual_retrieved_segments}")
    # Note: visual_retrieved_segments.json will be saved by vid_searcher.py with proper structure
    # Don't save raw segment IDs here to avoid overriding structured format
    

    # If no visual segments were retrieved, create fallback segments
    if not visual_retrieved_segments:
        print("No visual segments retrieved, creating fallback segments...")
        fallback_segments = create_fallback_segments_from_kv(kvdata, set())
        if fallback_segments:
            # Create proper segment format
            visual_retrieved_segments = []
            for i, segment_key in enumerate(fallback_segments[:5]):  # Use first 5 segments
                parts = segment_key.split('_')
                # Handle both formats: '20250909_121908_0' and 'output_0'
                if len(parts) >= 3:  # Format: 20250909_121908_0
                    video_name = f"{parts[0]}_{parts[1]}"  # 20250909_121908
                    segment_id = int(parts[2])  # 0, 1, 2, etc.
                elif len(parts) >= 2:  # Format: output_0
                    video_name = parts[0]  # output
                    segment_id = int(parts[1])  # 0, 1, 2, etc.
                else:
                    continue  # Skip if format doesn't match
                
                visual_retrieved_segments.append({
                    "video_name": video_name,
                    "segment_id": segment_id,
                    "start_time": segment_id * 30,  # 30 seconds per segment
                    "end_time": (segment_id + 1) * 30,
                    "description": f"Segment {segment_id} from {video_name}"
                })
            print(f"Created {len(visual_retrieved_segments)} fallback segments")
            
            # Save the fallback segments
            with open(os.path.join(scene_output_dir, 'visual_retrieved_segments.json'), 'w') as f:
                json.dump(visual_retrieved_segments, f, indent=2)
    
    return visual_retrieved_segments