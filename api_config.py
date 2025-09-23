#!/usr/bin/env python3
"""
API配置管理
支持多种API提供商的配置和切换
"""

import os
from typing import Dict, Any, Optional

class APIConfig:
    """API配置管理类"""
    
    def __init__(self):
        self.configs = {
            "openai": {
                "name": "OpenAI GPT-4o",
                "api_key_env": "OPENAI_API_KEY",
                "model": "gpt-4o",
                "max_tokens": 500,
                "temperature": 0.1,
                "cost_per_1k_tokens": 0.005,  # GPT-4o比GPT-4V便宜
                "features": ["vision", "text", "multimodal", "faster", "cheaper"]
            },
            "claude": {
                "name": "Anthropic Claude 3.5 Sonnet",
                "api_key_env": "ANTHROPIC_API_KEY", 
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 500,
                "temperature": 0.1,
                "cost_per_1k_tokens": 0.015,
                "features": ["vision", "text", "multimodal"]
            },
            "gemini": {
                "name": "Google Gemini Pro Vision",
                "api_key_env": "GOOGLE_API_KEY",
                "model": "gemini-1.5-pro",
                "max_tokens": 500,
                "temperature": 0.1,
                "cost_per_1k_tokens": 0.005,
                "features": ["vision", "text", "multimodal"]
            }
        }
    
    def get_config(self, provider: str) -> Dict[str, Any]:
        """获取指定提供商的配置"""
        if provider not in self.configs:
            raise ValueError(f"Unsupported provider: {provider}")
        return self.configs[provider]
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """获取API密钥"""
        config = self.get_config(provider)
        return os.getenv(config["api_key_env"])
    
    def set_api_key(self, provider: str, api_key: str):
        """设置API密钥到环境变量"""
        config = self.get_config(provider)
        os.environ[config["api_key_env"]] = api_key
        print(f"API key set for {config['name']}")
    
    def validate_config(self, provider: str) -> bool:
        """验证配置是否完整"""
        try:
            config = self.get_config(provider)
            api_key = self.get_api_key(provider)
            return api_key is not None and len(api_key) > 0
        except:
            return False
    
    def list_providers(self) -> list:
        """列出所有可用的提供商"""
        return list(self.configs.keys())
    
    def get_provider_info(self, provider: str) -> str:
        """获取提供商信息"""
        config = self.get_config(provider)
        return f"""
{config['name']}
- Model: {config['model']}
- Max Tokens: {config['max_tokens']}
- Cost per 1K tokens: ${config['cost_per_1k_tokens']}
- Features: {', '.join(config['features'])}
- API Key: {'✓ Set' if self.validate_config(provider) else '✗ Not set'}
        """
    
    def recommend_provider(self, budget: str = "medium") -> str:
        """根据预算推荐提供商"""
        if budget == "low":
            return "gemini"  # 最便宜
        elif budget == "high":
            return "openai"  # 最稳定
        else:
            return "claude"  # 平衡性价比

# 全局配置实例
api_config = APIConfig()

def setup_api(provider: str, api_key: str = None):
    """快速设置API"""
    if api_key:
        api_config.set_api_key(provider, api_key)
    
    if api_config.validate_config(provider):
        print(f"✅ {provider.upper()} API configured successfully!")
        return True
    else:
        print(f"❌ {provider.upper()} API configuration failed!")
        print(f"Please set the API key: {api_config.get_config(provider)['api_key_env']}")
        return False

def get_available_providers():
    """获取可用的API提供商列表"""
    providers = []
    for provider in api_config.list_providers():
        if api_config.validate_config(provider):
            providers.append(provider)
    return providers

# 使用示例
if __name__ == "__main__":
    print("🔧 API Configuration Manager")
    print("=" * 50)
    
    # 显示所有提供商信息
    for provider in api_config.list_providers():
        print(api_config.get_provider_info(provider))
        print("-" * 30)
    
    # 推荐提供商
    print(f"💡 Recommended for low budget: {api_config.recommend_provider('low')}")
    print(f"💡 Recommended for high budget: {api_config.recommend_provider('high')}")
    
    # 检查当前配置
    print("\n🔍 Current Configuration:")
    for provider in api_config.list_providers():
        status = "✅ Ready" if api_config.validate_config(provider) else "❌ Not configured"
        print(f"  {provider.upper()}: {status}")
