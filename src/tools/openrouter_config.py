import os
import time
import logging
import requests
from dataclasses import dataclass
import backoff
from typing import Optional, Dict, Any

# 设置日志记录
logger = logging.getLogger('api_calls')
logger.setLevel(logging.DEBUG)

# 移除所有现有的处理器
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# 创建日志目录
log_dir = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), 'logs')
os.makedirs(log_dir, exist_ok=True)

# 设置文件处理器
log_file = os.path.join(log_dir, f'api_calls_{time.strftime("%Y%m%d")}.log')
print(f"Creating log file at: {log_file}")

try:
    file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
    file_handler.setLevel(logging.DEBUG)
    print("Successfully created file handler")
except Exception as e:
    print(f"Error creating file handler: {str(e)}")

# 设置控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# 设置日志格式
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 添加处理器
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 立即测试日志记录
logger.debug("Logger initialization completed")
logger.info("API logging system started")

# 状态图标
SUCCESS_ICON = "✓"
ERROR_ICON = "✗"
WAIT_ICON = "⟳"


@dataclass
class ChatMessage:
    content: str


@dataclass
class ChatChoice:
    message: ChatMessage


@dataclass
class ChatCompletion:
    choices: list[ChatChoice]


# Ollama 配置
OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Ollama API 地址
MODEL = "deepseek-r1:7b"  # 使用的模型

logger.info(f"{SUCCESS_ICON} Ollama 配置成功: {OLLAMA_API_URL}, 模型: {MODEL}")


@backoff.on_exception(
    backoff.expo,
    (Exception),
    max_tries=5,
    max_time=300,
    giveup=lambda e: "API limit" not in str(e)
)
def generate_content_with_retry(prompt: str, config: Optional[Dict[str, Any]] = None):
    """带重试机制的内容生成函数"""
    try:
        logger.info(f"{WAIT_ICON} 正在调用 Ollama API...")
        logger.info(f"请求内容: {prompt[:500]}..." if len(prompt) > 500 else f"请求内容: {prompt}")
        logger.info(f"请求配置: {config}")

        # 准备请求体
        payload = {
            "model": MODEL,
            "prompt": prompt,
            "stream": False  # 设置为 True 以流式传输响应
        }

        # 调用 Ollama API
        response = requests.post(OLLAMA_API_URL, json=payload)

        if response.status_code != 200:
            raise Exception(f"API 返回错误: {response.status_code} - {response.text}")

        # 解析响应
        response_data = response.json()
        generated_text = response_data.get("response", "")

        logger.info(f"{SUCCESS_ICON} API 调用成功")
        logger.info(f"响应内容: {generated_text[:500]}..." if len(generated_text) > 500 else f"响应内容: {generated_text}")
        return generated_text

    except Exception as e:
        if "API limit" in str(e):
            logger.warning(f"{ERROR_ICON} 触发 API 限制，等待重试... 错误: {str(e)}")
            time.sleep(5)
            raise e
        logger.error(f"{ERROR_ICON} API 调用失败: {str(e)}")
        logger.error(f"错误详情: {str(e)}")
        raise e


def get_chat_completion(messages, max_retries=3, initial_retry_delay=1):
    """获取聊天完成结果，包含重试逻辑"""
    try:
        logger.info(f"{WAIT_ICON} 使用模型: {MODEL}")
        logger.debug(f"消息内容: {messages}")

        for attempt in range(max_retries):
            try:
                # 转换消息格式
                prompt = ""
                system_instruction = None

                for message in messages:
                    role = message["role"]
                    content = message["content"]
                    if role == "system":
                        system_instruction = content
                    elif role == "user":
                        prompt += f"User: {content}\n"
                    elif role == "assistant":
                        prompt += f"Assistant: {content}\n"

                # 准备配置
                config = {}
                if system_instruction:
                    config['system_instruction'] = system_instruction

                # 调用 API
                response_text = generate_content_with_retry(
                    prompt=prompt.strip(),
                    config=config
                )

                if not response_text:
                    logger.warning(
                        f"{ERROR_ICON} 尝试 {attempt + 1}/{max_retries}: API 返回空值")
                    if attempt < max_retries - 1:
                        retry_delay = initial_retry_delay * (2 ** attempt)
                        logger.info(f"{WAIT_ICON} 等待 {retry_delay} 秒后重试...")
                        time.sleep(retry_delay)
                        continue
                    return None

                # 返回响应内容
                logger.debug(f"API 原始响应: {response_text}")
                logger.info(f"{SUCCESS_ICON} 成功获取响应")
                return response_text

            except Exception as e:
                logger.error(
                    f"{ERROR_ICON} 尝试 {attempt + 1}/{max_retries} 失败: {str(e)}")
                if attempt < max_retries - 1:
                    retry_delay = initial_retry_delay * (2 ** attempt)
                    logger.info(f"{WAIT_ICON} 等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"{ERROR_ICON} 最终错误: {str(e)}")
                    return None

    except Exception as e:
        logger.error(f"{ERROR_ICON} get_chat_completion 发生错误: {str(e)}")
        return None