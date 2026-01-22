#!/usr/bin/env python3
"""测试 LLM 连接"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from whisperlm.services.llm_service import LLMService
from whisperlm.config import get_settings


async def test_llm():
    """测试 LLM 连接"""
    settings = get_settings()
    service = LLMService(settings)
    
    print(f"LLM 配置:")
    print(f"  启用: {service.is_enabled}")
    print(f"  提供商: {settings.llm.provider}")
    print(f"  模型: {settings.llm.model}")
    print(f"  地址: {settings.llm.base_url}")
    print()
    
    if not service.is_enabled:
        print("❌ LLM 未启用")
        return
    
    print("正在测试连接...")
    connected = await service.check_connection()
    
    if connected:
        print("✅ LLM 连接成功！")
        
        # 测试一个简单的请求
        print("\n测试简单请求...")
        try:
            client = service._get_client()
            response = await client.chat.completions.create(
                model=settings.llm.model,
                messages=[{"role": "user", "content": "请说'你好'"}],
                max_tokens=10,
            )
            result = response.choices[0].message.content
            print(f"✅ 请求成功: {result}")
        except Exception as e:
            print(f"❌ 请求失败: {e}")
    else:
        print("❌ LLM 连接失败")
        print("\n请检查:")
        print("1. vLLM 服务是否运行: vllm serve Qwen/Qwen3-32B --port 8001")
        print("2. 服务地址是否正确:", settings.llm.base_url)
        print("3. 网络连接是否正常")


if __name__ == "__main__":
    asyncio.run(test_llm())

