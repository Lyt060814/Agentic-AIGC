#!/usr/bin/env python3
"""
配置文件加载工具
从environment/config/config.yml读取API配置
"""

import os
import yaml
from typing import Dict, Any, Optional

class ConfigLoader:
    """配置文件加载器"""
    
    def __init__(self, config_path: str = None):
        """
        初始化配置加载器
        
        Args:
            config_path: 配置文件路径，默认为environment/config/config.yml
        """
        if config_path is None:
            self.config_path = os.path.join(os.getcwd(), 'environment', 'config', 'config.yml')
        else:
            self.config_path = config_path
        
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            if not os.path.exists(self.config_path):
                print(f"⚠️ 配置文件不存在: {self.config_path}")
                return {}
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                print(f"✅ 成功加载配置文件: {self.config_path}")
                return config or {}
        except Exception as e:
            print(f"❌ 加载配置文件失败: {e}")
            return {}
    
    def get_llm_config(self) -> Dict[str, Any]:
        """获取LLM配置"""
        return self.config.get('llm', {})
    
    def get_api_key(self) -> Optional[str]:
        """获取API密钥"""
        llm_config = self.get_llm_config()
        return llm_config.get('api_key')
    
    def get_base_url(self) -> Optional[str]:
        """获取Base URL"""
        llm_config = self.get_llm_config()
        return llm_config.get('base_url')
    
    def get_openai_config(self) -> Dict[str, Any]:
        """获取OpenAI配置"""
        llm_config = self.get_llm_config()
        return {
            'api_key': llm_config.get('api_key'),
            'base_url': llm_config.get('base_url')
        }
    
    def validate_config(self) -> bool:
        """验证配置是否完整"""
        api_key = self.get_api_key()
        base_url = self.get_base_url()
        
        if not api_key:
            print("❌ 缺少API密钥")
            return False
        
        if not base_url:
            print("❌ 缺少Base URL")
            return False
        
        print(f"✅ 配置验证通过")
        print(f"🔑 API密钥: {api_key[:10]}...")
        print(f"🌐 Base URL: {base_url}")
        return True
    
    def print_config(self):
        """打印配置信息"""
        print("📋 当前配置:")
        print(f"  配置文件路径: {self.config_path}")
        print(f"  API密钥: {self.get_api_key()[:10] if self.get_api_key() else '未设置'}...")
        print(f"  Base URL: {self.get_base_url() or '未设置'}")
        print(f"  配置完整: {'✅' if self.validate_config() else '❌'}")

# 全局配置实例
config_loader = ConfigLoader()

def get_openai_config() -> Dict[str, Any]:
    """获取OpenAI配置"""
    return config_loader.get_openai_config()

def validate_api_config() -> bool:
    """验证API配置"""
    return config_loader.validate_config()

# 使用示例
if __name__ == "__main__":
    print("🔧 配置加载器测试")
    print("=" * 40)
    
    # 打印配置信息
    config_loader.print_config()
    
    # 验证配置
    if validate_api_config():
        print("\n🎉 配置验证成功！可以开始使用GPT-4o API了")
    else:
        print("\n⚠️ 配置验证失败，请检查配置文件")

