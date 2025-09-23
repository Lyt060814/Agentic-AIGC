#!/usr/bin/env python3
"""
API-based video processing to replace MiniCPM
支持多种API提供商的多模态视频处理
"""

import base64
import io
import json
import requests
from PIL import Image
from typing import List, Dict, Any, Optional
import os

class APIVideoProcessor:
    def __init__(self, provider="openai", api_key=None, base_url=None):
        """
        初始化API视频处理器
        
        Args:
            provider: API提供商 ("openai", "claude", "gemini", "azure")
            api_key: API密钥
            base_url: API基础URL
        """
        self.provider = provider
        self.base_url = base_url
        
        # 优先使用传入的api_key，然后尝试从配置文件读取，最后从环境变量读取
        if api_key:
            self.api_key = api_key
        else:
            # 尝试从配置文件读取
            try:
                from config_loader import get_openai_config
                openai_config = get_openai_config()
                if openai_config.get('api_key'):
                    self.api_key = openai_config['api_key']
                    if openai_config.get('base_url'):
                        self.base_url = openai_config['base_url']
                    print(f"✅ 从配置文件读取API密钥: {self.api_key[:10]}...")
                    print(f"✅ 从配置文件读取Base URL: {self.base_url}")
                else:
                    self.api_key = os.getenv(f"{provider.upper()}_API_KEY")
            except Exception as e:
                print(f"⚠️ 读取配置文件失败: {e}")
                self.api_key = os.getenv(f"{provider.upper()}_API_KEY")
        
        if not self.api_key:
            raise ValueError(f"API key not found for {provider}")
    
    def image_to_base64(self, image: Image.Image) -> str:
        """将PIL图像转换为base64编码"""
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    
    def video_captioning(self, frames: List[Image.Image], character_refs: List[Dict], 
                        transcript: str, video_name: str) -> str:
        """
        视频字幕生成 - 替代MiniCPM的caption功能
        
        Args:
            frames: 视频帧列表
            character_refs: 角色参考信息
            transcript: 语音转录文本
            
        Returns:
            生成的视频描述文本
        """
        if self.provider == "openai":
            return self._gpt4v_caption(frames, character_refs, transcript)
        elif self.provider == "claude":
            return self._claude_caption(frames, character_refs, transcript)
        elif self.provider == "gemini":
            return self._gemini_caption(frames, character_refs, transcript)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def video_frame_selection(self, frames: List[Image.Image], description: str, 
                            duration: float) -> int:
        """
        视频帧选择 - 替代MiniCPM的frame selection功能
        
        Args:
            frames: 视频帧列表
            description: 场景描述
            duration: 所需时长
            
        Returns:
            最佳起始帧的索引
        """
        if self.provider == "openai":
            return self._gpt4v_frame_selection(frames, description, duration)
        elif self.provider == "claude":
            return self._claude_frame_selection(frames, description, duration)
        elif self.provider == "gemini":
            return self._gemini_frame_selection(frames, description, duration)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _gpt4v_caption(self, frames: List[Image.Image], character_refs: List[Dict], 
                      transcript: str) -> str:
        """使用GPT-4o进行视频字幕生成"""
        import openai
        
        # 配置OpenAI客户端
        if self.base_url:
            openai.api_base = self.base_url
            print(f"🔗 使用自定义API端点: {self.base_url}")
        
        # 构建角色参考文本
        char_text = ""
        for char in character_refs:
            char_text += f"Character: {char['name']}\n"
        
        # 构建提示词
        prompt = f"""
        You are analyzing a video segment. Below are character references and video frames.
        
        Character References:
        {char_text}
        
        Audio Transcript: {transcript}
        
        Please provide a detailed video scene description including:
        1. Characters' emotions and actions
        2. Motion dynamics and scene flow
        3. Visual elements and atmosphere
        4. Use character names if they appear in the video
        
        Respond in English only. Be descriptive and engaging.
        """
        
        # 准备消息内容
        content = [{"type": "text", "text": prompt}]
        
        # 添加视频帧
        for frame in frames:
            img_b64 = self.image_to_base64(frame)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })
        
        try:
            # 使用新的OpenAI API格式
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            response = client.chat.completions.create(
                model="gpt-4o",  # 使用GPT-4o
                messages=[{"role": "user", "content": content}],
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"GPT-4o API error: {e}")
            return "Video analysis failed"
    
    def _gpt4v_frame_selection(self, frames: List[Image.Image], description: str, 
                              duration: float) -> int:
        """使用GPT-4o进行视频帧选择"""
        import openai
        
        # 配置OpenAI客户端
        if self.base_url:
            openai.api_base = self.base_url
            print(f"🔗 使用自定义API端点: {self.base_url}")
        
        prompt = f"""
        You are analyzing {len(frames)} consecutive video frames to find the best sequence.
        
        Scene Description: "{description}"
        Required Duration: {duration:.1f} seconds
        
        Requirements:
        1. Analyze ALL {len(frames)} frames
        2. Choose the best starting frame (0-{len(frames)-1})
        3. Maximum starting frame: {len(frames) - int(duration)}
        4. Return ONLY a single number - the frame index
        5. Select frames that best match the scene description
        
        Respond with just the frame number (0-{len(frames)-1}):
        """
        
        content = [{"type": "text", "text": prompt}]
        
        # 添加视频帧
        for i, frame in enumerate(frames):
            img_b64 = self.image_to_base64(frame)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })
        
        try:
            # 使用新的OpenAI API格式
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            response = client.chat.completions.create(
                model="gpt-4o",  # 使用GPT-4o
                messages=[{"role": "user", "content": content}],
                max_tokens=10
            )
            
            # 提取数字
            import re
            result = response.choices[0].message.content.strip()
            numbers = re.findall(r'\d+', result)
            if numbers:
                frame_idx = int(numbers[0])
                return max(0, min(frame_idx, len(frames) - int(duration)))
            return 0
        except Exception as e:
            print(f"GPT-4o frame selection error: {e}")
            return 0
    
    def _claude_caption(self, frames: List[Image.Image], character_refs: List[Dict], 
                       transcript: str) -> str:
        """使用Claude进行视频字幕生成"""
        import anthropic
        
        client = anthropic.Anthropic(api_key=self.api_key)
        
        # 构建角色参考文本
        char_text = ""
        for char in character_refs:
            char_text += f"Character: {char['name']}\n"
        
        prompt = f"""
        You are analyzing a video segment. Below are character references and video frames.
        
        Character References:
        {char_text}
        
        Audio Transcript: {transcript}
        
        Please provide a detailed video scene description including:
        1. Characters' emotions and actions
        2. Motion dynamics and scene flow
        3. Visual elements and atmosphere
        4. Use character names if they appear in the video
        
        Respond in English only. Be descriptive and engaging.
        """
        
        # 准备图像数据
        images = []
        for frame in frames:
            img_b64 = self.image_to_base64(frame)
            images.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": img_b64
                }
            })
        
        try:
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=500,
                messages=[{
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}] + images
                }]
            )
            return response.content[0].text.strip()
        except Exception as e:
            print(f"Claude API error: {e}")
            return "Video analysis failed"
    
    def _claude_frame_selection(self, frames: List[Image.Image], description: str, 
                               duration: float) -> int:
        """使用Claude进行视频帧选择"""
        import anthropic
        
        client = anthropic.Anthropic(api_key=self.api_key)
        
        prompt = f"""
        You are analyzing {len(frames)} consecutive video frames to find the best sequence.
        
        Scene Description: "{description}"
        Required Duration: {duration:.1f} seconds
        
        Requirements:
        1. Analyze ALL {len(frames)} frames
        2. Choose the best starting frame (0-{len(frames)-1})
        3. Maximum starting frame: {len(frames) - int(duration)}
        4. Return ONLY a single number - the frame index
        5. Select frames that best match the scene description
        
        Respond with just the frame number (0-{len(frames)-1}):
        """
        
        # 准备图像数据
        images = []
        for frame in frames:
            img_b64 = self.image_to_base64(frame)
            images.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": img_b64
                }
            })
        
        try:
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=10,
                messages=[{
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}] + images
                }]
            )
            
            # 提取数字
            import re
            result = response.content[0].text.strip()
            numbers = re.findall(r'\d+', result)
            if numbers:
                frame_idx = int(numbers[0])
                return max(0, min(frame_idx, len(frames) - int(duration)))
            return 0
        except Exception as e:
            print(f"Claude frame selection error: {e}")
            return 0
    
    def _gemini_caption(self, frames: List[Image.Image], character_refs: List[Dict], 
                       transcript: str) -> str:
        """使用Gemini进行视频字幕生成"""
        import google.generativeai as genai
        
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # 构建角色参考文本
        char_text = ""
        for char in character_refs:
            char_text += f"Character: {char['name']}\n"
        
        prompt = f"""
        You are analyzing a video segment. Below are character references and video frames.
        
        Character References:
        {char_text}
        
        Audio Transcript: {transcript}
        
        Please provide a detailed video scene description including:
        1. Characters' emotions and actions
        2. Motion dynamics and scene flow
        3. Visual elements and atmosphere
        4. Use character names if they appear in the video
        
        Respond in English only. Be descriptive and engaging.
        """
        
        # 准备内容
        content = [prompt]
        for frame in frames:
            content.append(frame)
        
        try:
            response = model.generate_content(content)
            return response.text.strip()
        except Exception as e:
            print(f"Gemini API error: {e}")
            return "Video analysis failed"
    
    def _gemini_frame_selection(self, frames: List[Image.Image], description: str, 
                               duration: float) -> int:
        """使用Gemini进行视频帧选择"""
        import google.generativeai as genai
        
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        prompt = f"""
        You are analyzing {len(frames)} consecutive video frames to find the best sequence.
        
        Scene Description: "{description}"
        Required Duration: {duration:.1f} seconds
        
        Requirements:
        1. Analyze ALL {len(frames)} frames
        2. Choose the best starting frame (0-{len(frames)-1})
        3. Maximum starting frame: {len(frames) - int(duration)}
        4. Return ONLY a single number - the frame index
        5. Select frames that best match the scene description
        
        Respond with just the frame number (0-{len(frames)-1}):
        """
        
        # 准备内容
        content = [prompt]
        for frame in frames:
            content.append(frame)
        
        try:
            response = model.generate_content(content)
            
            # 提取数字
            import re
            result = response.text.strip()
            numbers = re.findall(r'\d+', result)
            if numbers:
                frame_idx = int(numbers[0])
                return max(0, min(frame_idx, len(frames) - int(duration)))
            return 0
        except Exception as e:
            print(f"Gemini frame selection error: {e}")
            return 0


# 使用示例
if __name__ == "__main__":
    # 示例：使用GPT-4V
    processor = APIVideoProcessor(provider="openai", api_key="your-api-key")
    
    # 示例：使用Claude
    # processor = APIVideoProcessor(provider="claude", api_key="your-api-key")
    
    # 示例：使用Gemini
    # processor = APIVideoProcessor(provider="gemini", api_key="your-api-key")
    
    print("API Video Processor initialized successfully!")
