### Clone and Install

```bash
# 1. Clone the repository
git clone https://github.com/HKUDS/Agentic-AIGC.git

# 2. Create and activate a Conda environment
conda create --name aicreator python=3.10
conda activate aicreator

# 3. Install system dependencies (pynini, ffmpeg)
conda install -y -c conda-forge pynini==2.1.5 ffmpeg

# 4. Install Python dependencies
pip install -r requirements_mac.txt
```

### Download Required Models

```bash
# Download CosyVoice
cd tools/CosyVoice
hf download PillowTa1k/CosyVoice --local-dir pretrained_models

# Download Whisper
cd tools
hf download openai/whisper-large-v3-turbo --local-dir whisper-large-v3-turbo

# Download all-MiniLM-L6-v2
cd tools
hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir all-MiniLM-L6-v2

# Download ImageBind
cd tools
mkdir .checkpoints
cd .checkpoints
wget https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth
```

### Quick Start

1. Edit Agentic-AIGC/environment/config/config.yml to add your LLM API key and base URL.

2. Place your source video/audio file in a directory (e.g., dataset/user_video/).

3. Execute python main.py to start the process.

4. When prompted, input "Summary of News".