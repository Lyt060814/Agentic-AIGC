# Agentic-AIGC AI2Apps 迁移 API 设计文档

本文档详细定义了将 Agentic-AIGC 项目迁移到 ai2apps 平台时三个核心服务的 API 接口规范。

## 目录
- [1. Whisper转录服务 API](#1-whisper转录服务-api)
- [2. Coqui TTS语音合成服务 API](#2-coqui-tts语音合成服务-api)
- [3. VideoRAG检索服务 API](#3-videorag检索服务-api)
- [4. 默认值配置](#4-默认值配置)

---

## 1. Whisper转录服务 API

### 服务描述
负责将视频文件中的语音内容转录为文本，支持多种视频格式，基于 Whisper 模型实现。

### 1.1 输入 JSON 格式
```json
{
  "config": {
    "video_source_dir": "dataset/user_video/",
    "model_id": "/private/tmp/whisper-large-v3-turbo",
    "use_timestamps": false,
    "chunk_length_s": 30,
    "batch_size": 16,
    "force_overwrite": true,
    "file_extensions": [".mp4", ".MP4", ".avi", ".mov", ".mkv"]
  },
  "output_config": {
    "output_dir": "dataset/video_edit/writing_data/",
    "output_filename": "audio_transcript.txt"
  }
}
```

#### 重要说明
- **服务处理方式**: Whisper服务会**自动扫描**`video_source_dir`目录中的所有符合`file_extensions`的视频文件
- **输出逻辑**: 按照原始源码逻辑，每个视频文件会依次覆盖写入到同一个输出文件，最终保留最后一个视频的转录结果
- **设备和数据类型**: 服务会自动检测CUDA可用性并设置最优配置
- **模型路径**: 默认使用 `/private/tmp/whisper-large-v3-turbo` 本地模型路径

#### 参数说明

**config 部分**:
- `video_source_dir`: 视频文件所在目录
- `model_id`: 使用的 Whisper 模型路径，默认为 `/private/tmp/whisper-large-v3-turbo`
- `use_timestamps`: 是否在输出中包含时间戳
- `chunk_length_s`: 音频分块长度（秒）
- `batch_size`: 批处理大小
- `force_overwrite`: 是否强制覆盖已存在的转录文件
- `file_extensions`: 支持的视频文件扩展名列表

**output_config 部分**:
- `output_dir`: 输出目录路径
- `output_filename`: 输出文件名

### 1.2 输出 JSON 格式
```json
{
  "success": true,
  "results": {
    "/path/to/video1.mp4": "dataset/video_edit/writing_data/audio_transcript.txt",
    "/path/to/video2.mp4": "dataset/video_edit/writing_data/audio_transcript.txt"
  },
  "summary": {
    "total_videos": 2,
    "successful_transcriptions": 2,
    "failed_transcriptions": 0
  },
  "combined_transcript": "Final transcription text (last processed video)...",
  "output_file": "dataset/video_edit/writing_data/audio_transcript.txt",
  "processing_info": {
    "model_used": "whisper-large-v3-turbo",
    "device": "cuda:0",
    "torch_dtype": "float16",
    "total_processing_time": 45.2
  }
}
```

#### 输出说明
- `success`: 处理是否成功
- `results`: 视频文件到转录文件路径的映射
- `summary`: 处理统计信息
- `combined_transcript`: 最终转录文本内容（最后处理的视频）
- `output_file`: 输出文件的完整路径
- `processing_info`: 处理过程信息（模型、设备、处理时间等）

---

## 2. Coqui TTS语音合成服务 API

### 服务描述
将文本内容转换为语音音频，支持多种 TTS 模型，能够处理带分段标记的内容并生成时间戳信息。

### 2.1 输入 JSON 格式
```json
{
  "content_source": {
    "type": "json_file",
    "path": "dataset/video_edit/scene_output/video_scene.json",
    "content_field": "content_created"
  },
  "voice_config": {
    "model_name": "tts_models/en/ljspeech/vits",
    "language": "en",
    "speaker_wav": null,
    "sample_rate": 24000,
    "split_sentences": false
  },
  "output_config": {
    "output_dir": "dataset/video_edit/voice_gen/",
    "output_filename": "gen_news_audio.wav",
    "timestamp_filename": "gen_news_audio_timestamps.json",
    "segment_format": "wav",
    "cleanup_intermediate": true
  },
  "processing_options": {
    "segment_delimiter": "/////",
    "max_sentence_length": 200,
    "pause_between_segments": 0.5
  }
}
```

#### 参数说明
- `content_source`: 内容来源配置
  - `type`: 内容类型（"json_file" 或 "text"）
  - `path`: JSON文件路径
  - `content_field`: JSON中的内容字段名
- `voice_config`: 语音生成配置
  - `model_name`: TTS模型名称
  - `language`: 语言代码
  - `speaker_wav`: 说话人音频文件路径（用于声音克隆）
  - `sample_rate`: 音频采样率
- `output_config`: 输出配置
  - `output_dir`: 输出目录
  - `output_filename`: 最终音频文件名
  - `timestamp_filename`: 时间戳文件名
  - `cleanup_intermediate`: 是否清理中间文件
- `processing_options`: 处理选项
  - `segment_delimiter`: 段落分隔符
  - `max_sentence_length`: 最大句子长度
  - `pause_between_segments`: 段落间暂停时间

### 2.2 输出 JSON 格式
```json
{
  "success": true,
  "audio_output": {
    "final_audio_path": "dataset/video_edit/voice_gen/gen_news_audio.wav",
    "duration_seconds": 120.5,
    "sample_rate": 24000,
    "channels": 1
  },
  "timestamp_data": {
    "timestamp_file": "dataset/video_edit/voice_gen/gen_news_audio_timestamps.json",
    "sentence_data": {
      "total_chunks": 8,
      "chunks": [
        {
          "id": 0,
          "timestamp": 5.234,
          "duration": 4.123,
          "text": "First segment content..."
        }
      ]
    }
  },
  "processing_info": {
    "segments_processed": 8,
    "model_used": "tts_models/en/ljspeech/vits",
    "total_processing_time": 23.1,
    "intermediate_files_deleted": true
  }
}
```

#### 输出说明
- `audio_output`: 音频输出信息
- `timestamp_data`: 时间戳数据和文件信息
- `processing_info`: 处理过程统计信息

---

## 3. VideoRAG检索服务 API

### 服务描述
提供视频预处理和检索功能，支持视频分片、转录、字幕生成、特征提取和相似度搜索。

### 3.1 预处理操作 (Preload)

#### 3.1.1 输入 JSON 格式
```json
{
  "action": "preload",
  "video_config": {
    "video_source_dir": "dataset/user_video/",
    "working_dir": "dataset/video_edit/videosource-workdir/"
  },
  "processing_config": {
    "video_segment_length": 30,
    "rough_num_frames_per_segment": 10,
    "video_output_format": "mp4",
    "audio_output_format": "mp3",
    "video_embedding_batch_num": 2,
    "video_embedding_dim": 1024
  },
  "api_config": {
    "provider": "openai",
    "api_key": "from_config_file",
    "base_url": "from_config_file"
  }
}
```

#### 重要变更说明
- **移除MiniCPM配置**: 现在使用GPT-4o API进行视觉分析，不需要本地MiniCPM模型
- **自动视频发现**: 服务会自动扫描`video_source_dir`中的.mp4文件，不需要指定`video_paths`
- **API配置**: 新增`api_config`用于GPT-4o API调用配置

#### 3.1.2 输出 JSON 格式
```json
{
  "success": true,
  "processed_videos": {
    "video1.mp4": {
      "segments_created": 12,
      "total_duration": 360,
      "transcription_completed": true,
      "caption_completed": true,
      "features_encoded": true
    }
  },
  "index_info": {
    "total_segments": 12,
    "total_features": 12288,
    "storage_location": "dataset/video_edit/videosource-workdir/",
    "index_files": [
      "kv_store_video_segments_video1.json",
      "vector_db_features.db"
    ]
  },
  "processing_info": {
    "total_processing_time": 180.5,
    "models_loaded": ["whisper", "minicpm", "imagebind", "sentence_transformer"]
  }
}
```

### 3.2 搜索操作 (Search)

#### 3.2.1 输入 JSON 格式
```json
{
  "action": "search",
  "query_config": {
    "query_source": {
      "type": "json_file",
      "path": "dataset/video_edit/scene_output/video_scene.json",
      "field": "segment_scene"
    }
  },
  "search_config": {
    "mode": "videoragcontent",
    "wo_reference": true
  },
  "working_config": {
    "working_dir": "dataset/video_edit/videosource-workdir/",
    "scene_output_dir": "dataset/video_edit/scene_output/"
  }
}
```

#### 重要说明
- **固定查询源**: 搜索服务固定从`video_scene.json`的`segment_scene`字段读取查询内容
- **简化配置**: 移除了`top_k`、`similarity_threshold`等参数，使用VideoRAG内部默认值
- **自动输出**: 搜索结果自动保存到`visual_retrieved_segments.json`

#### 3.2.2 输出 JSON 格式
```json
{
  "success": true,
  "search_results": [
    {
      "video_name": "20250909_121908",
      "segment_id": 5,
      "start_time": 150.0,
      "end_time": 180.0,
      "similarity_score": 0.89,
      "description": "Discussion about AI developments in machine learning",
      "transcript": "Today we're talking about the latest breakthroughs...",
      "file_path": "dataset/video_edit/videosource-workdir/segments/20250909_121908_5.mp4"
    }
  ],
  "query_info": {
    "original_query": "Tech news about artificial intelligence...",
    "total_matches": 15,
    "returned_matches": 10,
    "processing_time": 2.3
  },
  "output_files": {
    "visual_segments_file": "dataset/video_edit/scene_output/visual_retrieved_segments.json"
  }
}
```

#### 参数说明
- `action`: 操作类型（"preload" 或 "search"）
- `query_config`: 查询配置
- `search_config`: 搜索参数配置
- `working_config`: 工作目录配置

---

## 4. 默认值配置

### 4.1 Whisper服务默认值
```json
{
  "config": {
    "model_id": "/private/tmp/whisper-large-v3-turbo",
    "use_timestamps": false,
    "chunk_length_s": 30,
    "batch_size": 16,
    "file_extensions": [".mp4", ".MP4", ".avi", ".mov", ".mkv"],
    "force_overwrite": false,
    "video_source_dir": "dataset/user_video/"
  },
  "output_config": {
    "output_dir": "dataset/video_edit/writing_data/",
    "output_filename": "audio_transcript.txt"
  }
}
```

**注意**: `device`和`torch_dtype`由服务自动检测设置。模型默认路径为本地 `/private/tmp/whisper-large-v3-turbo`。

### 4.2 Coqui TTS服务默认值
```json
{
  "model_name": "tts_models/en/ljspeech/vits",
  "language": "en",
  "sample_rate": 24000,
  "segment_delimiter": "/////",
  "pause_between_segments": 0.5,
  "cleanup_intermediate": true,
  "max_sentence_length": 200
}
```

### 4.3 VideoRAG服务默认值
```json
{
  "video_segment_length": 30,
  "rough_num_frames_per_segment": 10,
  "video_output_format": "mp4",
  "audio_output_format": "mp3",
  "video_embedding_batch_num": 2,
  "video_embedding_dim": 1024,
  "mode": "videoragcontent",
  "wo_reference": true,
  "api_provider": "openai"
}
```

---

## 5. 使用示例

### 5.1 完整工作流调用序列

```python
# 1. VideoRAG预处理
preload_input = {
    "action": "preload",
    "video_config": {
        "video_source_dir": "dataset/user_video/"
    }
}
preload_result = await session.pipeChat("/video_rag/api.py", preload_input, False)

# 2. Whisper转录
whisper_input = {
    "config": {
        "video_source_dir": "dataset/user_video/",
        "force_overwrite": True
    }
}
transcript_result = await session.pipeChat("/whisper/api.py", whisper_input, False)

# 3. 内容生成（GPT处理，在news_agent中）
# 这部分在news_agent中处理，生成video_scene.json文件

# 4. Coqui TTS语音合成 (固定输入源)
tts_input = {}  # 空对象，服务会自动读取video_scene.json
voice_result = await session.pipeChat("/coqui_tts/api.py", tts_input, False)

# 5. VideoRAG视频搜索 (固定输入源)
search_input = {
    "action": "search"
}  # 简化输入，服务会自动读取video_scene.json
search_result = await session.pipeChat("/video_rag/api.py", search_input, False)
```

---

## 6. 错误处理

所有 API 都应该在失败时返回以下格式：

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "详细错误描述",
    "details": "额外的错误详细信息"
  },
  "debug_info": {
    "timestamp": "2025-09-24T10:30:00Z",
    "processing_stage": "model_loading",
    "input_received": true
  }
}
```

## 7. 重要实现细节

### 7.1 源码依据说明
本API设计完全基于源码分析，确保参数和行为与实际实现一致：

- **Whisper服务**: 基于`environment/roles/vid_news/audio_writer.py`中的`transcribe_main`类
- **CoquiTTS服务**: 基于`environment/roles/vid_news/voice_maker.py`中的`Voice_Maker`类
- **VideoRAG服务**: 基于`environment/roles/vid_preloader.py`和`environment/roles/vid_news/vid_searcher.py`

### 7.2 GPT-4o集成说明
- **视觉分析**: 使用`api_video_processor.py`中的`APIVideoProcessor`类
- **配置来源**: API密钥和base_url从`environment/config/config.yml`自动读取
- **替代关系**: 完全替代原MiniCPM-V模型的视觉分析功能

### 7.3 文件路径约定
- **输入目录**: `dataset/user_video/` (可配置)
- **工作目录**: `dataset/video_edit/` (固定结构)
- **输出文件**: 遵循原项目的文件命名和路径约定

### 7.4 性能考虑
- **Whisper**: GPU自动检测，批处理优化
- **CoquiTTS**: 模型初始化缓存，VITS模型优化
- **VideoRAG**: 预处理一次，多次搜索复用

---

*文档版本: 2.0 (基于完整源码分析)*
*最后更新: 2025-09-24*
*验证状态: ✅ 已与源码完全对齐*