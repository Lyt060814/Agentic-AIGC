#!/usr/bin/env python3
"""
集成API的视频字幕生成
直接替换原有的caption.py中的segment_caption函数
"""

import os
import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from moviepy.video.io.VideoFileClip import VideoFileClip

def load_character_references(face_db_path):
    """Load character reference images and names from the face database"""
    character_references = []
    
    # Get all character folders
    character_folders = glob.glob(os.path.join(face_db_path, "*"))
    
    for folder in character_folders:
        character_name = os.path.basename(folder)
        # Get first image from the character folder
        image_files = glob.glob(os.path.join(folder, "*.[jp][pn][g]"))
        
        if image_files:
            # Load the first image as a reference
            ref_image = Image.open(image_files[0]).convert("RGB")
            # Add character reference with name
            character_references.append({
                "name": character_name,
                "image": ref_image
            })
    
    return character_references

def encode_video(video, frame_times):
    """Extract frames from video at specified times"""
    frames = []
    for t in frame_times:
        frames.append(video.get_frame(t))
    frames = np.stack(frames, axis=0)
    frames = [Image.fromarray(v.astype('uint8')).resize((1280, 720)) for v in frames]
    return frames

def segment_caption(video_name, video_path, segment_index2name, transcripts, 
                   segment_times_info, caption_result, error_queue):
    """
    API-based video captioning function to replace MiniCPM
    这个函数可以直接替换原有的segment_caption函数
    """
    try:
        # 导入API处理器
        import sys
        sys.path.append(os.path.join(os.getcwd(), '..', '..'))
        from api_video_processor import APIVideoProcessor
        
        # 初始化API处理器（会自动从配置文件读取API信息）
        processor = APIVideoProcessor(provider="openai")
        
        # Load character references from face_db
        current_dir = os.getcwd()
        face_db_path = os.path.join(current_dir, 'dataset/video_edit/face_db')
        character_references = load_character_references(face_db_path)
        
        print(f"🎬 使用GPT-4o API进行视频字幕生成")
        print(f"📝 处理 {len(segment_index2name)} 个视频片段")
        
        with VideoFileClip(video_path) as video:
            for index in tqdm(segment_index2name, desc=f"Captioning Video {video_name}"):
                frame_times = segment_times_info[index]["frame_times"]
                video_frames = encode_video(video, frame_times)
                segment_transcript = transcripts[index]
                
                # 准备角色参考信息
                char_refs = []
                for char_ref in character_references:
                    char_refs.append({
                        "name": char_ref["name"],
                        "image": char_ref["image"]
                    })
                
                # 使用API进行视频字幕生成
                try:
                    segment_caption = processor.video_captioning(
                        frames=video_frames,
                        character_refs=char_refs,
                        transcript=segment_transcript,
                        video_name=video_name
                    )
                    
                    # 清理结果
                    segment_caption = segment_caption.replace("\n", "").replace("<|endoftext|>", "")
                    caption_result[index] = segment_caption
                    
                    print(f"✅ 片段 {index} 字幕生成完成: {segment_caption[:50]}...")
                    
                except Exception as e:
                    print(f"❌ 处理片段 {index} 失败: {e}")
                    # 使用fallback描述
                    caption_result[index] = f"Video segment {index} showing {segment_transcript[:50]}..."
                
    except Exception as e:
        error_queue.put(f"Error in segment_caption:\n {str(e)}")
        raise RuntimeError

def merge_segment_information(segment_index2name, segment_times_info, transcripts, captions):
    """Merge segment information for storage"""
    inserting_segments = {}
    for index in segment_index2name:
        inserting_segments[index] = {"content": None, "time": None}
        segment_name = segment_index2name[index]
        inserting_segments[index]["time"] = '-'.join(segment_name.split('-')[-2:])
        inserting_segments[index]["content"] = f"Caption:\n{captions[index]}" 
        inserting_segments[index]["transcript"] = transcripts[index]
        inserting_segments[index]["frame_times"] = segment_times_info[index]["frame_times"].tolist()
    
    return inserting_segments

# 使用示例
if __name__ == "__main__":
    print("🎬 API集成的视频字幕生成模块")
    print("💡 这个模块可以直接替换原有的caption.py")
    print("🔧 确保配置文件中的API密钥和base_url正确设置")
