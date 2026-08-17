import time
import re
import json
import os
import sys
import socket
import threading
import traceback
import platform
import random
import requests
import httpx
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from openai import OpenAI

# ===== 配置 =====
PROVIDER = "deepseek"   # 默认厂商：deepseek / openai / moonshot / zhipu / qwen / ollama / openrouter（可用环境变量 CHAT_PROVIDER 覆盖）
API_KEY = ""            # 默认 API Key：厂商专用环境变量缺失时使用
BASE_URL = "https://api.deepseek.com"           # 自定义 API 地址（留空用厂商默认；可指向自建 OpenAI 兼容网关）
MODEL = "deepseek-v4-flash"              # 自定义模型名（留空用厂商默认；如需全局覆盖可填，如 "qwen-max"）
CONTEXT_WINDOW = 1_000_000      # 默认上下文窗口（被厂商配置覆盖时以厂商为准）
AUTO_TRIM_THRESHOLD = 900_000   # 裁剪阈值上限（实际取 min(此值, 厂商窗口×90%)）
HISTORY_FILE = "chat_history.json"  # 对话历史保存文件
EXPORT_FILE = "chat_export.md"      # 对话导出文件
MAX_RETRIES = 3                     # 请求失败自动重试次数
RETRY_BACKOFF = 2.0                 # 重试初始等待秒数（指数退避）
SUMMARIZE_ON_TRIM = True            # 裁剪时把早期对话压缩为摘要保留（消耗少量 token）
MAX_MESSAGE_CHARS = 50_000          # 单条消息最大字符数（超出返回明确错误，防止误发超长内容）
MAX_BODY_BYTES = 1_000_000          # 网页版请求体上限（防止误发超大 JSON）
PROMPT_FILE = "prompt_templates.json"  # 提示词模板库（网页版/CLI 共用）
HISTORY_COMPACT_SIZE = 1_000_000    # 存档 JSON 超过该字节数时用紧凑格式写盘（长会话提速、省内存）
HISTORY_LIMIT_WEB = 500             # 网页版一次加载的最大消息条数（弱机自动降到 200）
IMPORT_MAX_MESSAGES = 5000          # 网页版/CLI 导入单次最大消息条数
# 内置提示词预置模板（prompt_templates.json 不存在时自动创建，可直接用 prompt apply 应用）
PROMPT_PRESETS = {
    "翻译助手": "你是一位专业翻译，把用户的输入准确翻译为目标语言，保留原有格式、语气与专业术语。",
    "代码审查": "你是资深软件工程师。请审查用户给出的代码：指出 Bug、性能与安全问题，并给出修正后的完整代码。",
    "内容总结": "你是内容总结助手。把用户的输入压缩为结构化要点：结论、关键信息、待办事项，不要复述原文。",
    "写作润色": "你是中文写作专家。润色用户的文字：更简洁、更有条理、保留原意，并简要说明你的改动。",
}
# 参考单价（USD / 百万 token）：仅用于 usage 命令的成本估算，可按实际价格调整
PRICE_INPUT = 0.27
PRICE_OUTPUT = 1.10
VERSION = "1.9.0"  # 程序版本（--version / version 命令 / 网页版状态栏）

# ===== 网页版（Beta） =====
WEB_ENABLED = True                  # 是否启动内置网页版
WEB_HOST = "127.0.0.1"              # 默认仅本机可访问；如需局域网访问改为 "0.0.0.0"
WEB_PORT = 8080
# ==================

# ===== 大模型厂商（全部走 OpenAI 兼容接口，可按需增删） =====
PROVIDERS = {
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
        "env": "DEEPSEEK_API_KEY",
        "thinking": True,     # 支持思考链（reasoning_content）
        "balance": True,      # 支持余额查询
        "stream_usage": True, # 支持 stream_options.include_usage
        "window": 1_000_000,
    },
    "openai": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "env": "OPENAI_API_KEY",
        "thinking": False,
        "balance": False,
        "stream_usage": True,
        "window": 128_000,
    },
    "moonshot": {
        "name": "Kimi（月之暗面）",
        "base_url": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-8k",
        "env": "MOONSHOT_API_KEY",
        "thinking": False,
        "balance": False,
        "stream_usage": True,
        "window": 32_000,
    },
    "zhipu": {
        "name": "智谱 GLM",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-flash",
        "env": "ZHIPU_API_KEY",
        "thinking": False,
        "balance": False,
        "stream_usage": True,
        "window": 128_000,
    },
    "qwen": {
        "name": "通义千问（阿里云百炼）",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "env": "DASHSCOPE_API_KEY",
        "thinking": False,
        "balance": False,
        "stream_usage": True,
        "window": 128_000,
    },
    "ollama": {
        "name": "Ollama（本地模型）",
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "qwen2.5:7b",
        "env": "",            # 本地服务无需 API Key
        "thinking": False,
        "balance": False,
        "stream_usage": False,
        "window": 8_192,
        "local": True,
    },
    "openrouter": {
        "name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "deepseek/deepseek-chat",
        "env": "OPENROUTER_API_KEY",
        "thinking": False,
        "balance": False,
        "stream_usage": True,
        "window": 128_000,
    },
}

# ===== 厂商状态（CLI 与网页版互通，共用同一份） =====
current_provider = (os.environ.get("CHAT_PROVIDER") or PROVIDER).strip().lower()
if current_provider not in PROVIDERS:
    current_provider = PROVIDER
model_override = MODEL.strip() or None

# 网页版填写的运行期配置（API Key / 自定义地址 / 模型），持久化到 chat_config.json，
# 未在代码里配置 base_url / api_key / 模型时程序不再退出，而是引导到网页版填写
CONFIG_FILE = "chat_config.json"
runtime_settings = {"api_key": {}, "base_url": "", "model": ""}


class ConfigError(Exception):
    """配置缺失（如未填写 API Key）"""


def load_runtime_config():
    """启动时加载网页版保存的运行期配置（chat_config.json）"""
    global runtime_settings, model_override
    try:
        # utf-8-sig：兼容 Windows 编辑器/PowerShell 写入时带 BOM 的文件
        with open(CONFIG_FILE, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if isinstance(data, dict):
            runtime_settings = {
                "api_key": dict(data.get("api_key") or {}),
                "base_url": str(data.get("base_url") or "").strip(),
                "model": str(data.get("model") or "").strip(),
            }
            model_override = runtime_settings["model"] or model_override
            return True
    except Exception:
        pass
    return False


def save_runtime_config():
    """把运行期配置原子写入 chat_config.json"""
    try:
        tmp = CONFIG_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(runtime_settings, f, ensure_ascii=False, indent=2)
        os.replace(tmp, CONFIG_FILE)
        return True
    except Exception:
        return False


def set_runtime_model(text):
    """设置全局模型覆盖（CLI model 命令与网页版共用，持久化）"""
    global model_override
    text = (text or "").strip()
    model_override = text or None
    runtime_settings["model"] = text
    save_runtime_config()


def provider_meta(name=None):
    """当前（或指定）厂商配置"""
    return PROVIDERS[name or current_provider]


def provider_base(name=None):
    """厂商 API 地址（网页版自定义地址 → 配置 BASE_URL → 厂商默认）"""
    return (runtime_settings["base_url"] or BASE_URL.strip() or provider_meta(name)["base_url"])


def current_model():
    """当前生效的模型名（model 覆盖 → 网页版配置 → MODEL 配置 → 厂商默认）"""
    return model_override or runtime_settings["model"] or MODEL.strip() or provider_meta()["model"]


def current_window():
    """当前生效的上下文窗口（以厂商配置为准）"""
    return int(provider_meta().get("window") or CONTEXT_WINDOW)


def resolve_api_key(name=None):
    """解析厂商 API Key：厂商专用环境变量 → 网页版填写 → 配置 API_KEY → 通用 CHAT_API_KEY"""
    meta = provider_meta(name)
    env = meta.get("env") or ""
    env_key = os.environ.get(env, "") if env else ""
    runtime_key = (runtime_settings.get("api_key") or {}).get(name or current_provider, "")
    return (env_key or runtime_key or API_KEY or os.environ.get("CHAT_API_KEY", "")).strip()


def config_error_message():
    """返回当前配置缺失的提示；配置可用返回 None"""
    meta = provider_meta()
    if meta.get("local"):
        return None  # 本地服务无需 Key
    if not resolve_api_key():
        env = meta.get("env") or "CHAT_API_KEY"
        return (f"未配置 API Key：请打开网页版设置面板填写（当前厂商 {meta['name']}，"
                f"也可设置环境变量 {env}）")
    return None


def ensure_utf8_stdio():
    """跨平台：stdin/stdout/stderr 统一 UTF-8（Windows 重定向到文件时避免 GBK 编码报错）"""
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

HELP_TEXT = """
命令：
  exit / quit / q  退出并保存对话
  balance / b      查询账户余额（仅 DeepSeek 支持）
  thinking      切换思考模式（仅支持的厂商生效）
  level <等级>  设置思考等级 low / medium / high / max
  provider      查看厂商列表；provider <名称> 切换厂商
  model         查看当前模型；model <名称> 切换模型
  system        查看系统提示词；system <内容> 设置；system off 清除
  temp          查看温度；temp <0-2> 设置；temp default 恢复默认
  usage         查看累计用量与参考成本；usage reset 清零
  stats         对话统计：角色分布 / 平均回复长度 / 上下文占用
  find <词>     在对话历史中搜索关键词
  session       查看会话列表；session new <名> 新建 / session <名> 切换 / session del <名> 删除 / session ren <新名> 重命名 / session cp <新名> 复制
  new           存档当前对话（自动存为新会话）并开启新对话
  redo          用上一个问题重新生成回复
  undo / u      撤销最后一轮对话（删除最后一条 AI 回复及其问题）
  summarize     把早期对话手动压缩为摘要（长对话释放上下文空间）
  findall <词>  跨全部会话搜索关键词
  prompts       查看提示词模板；prompt save <名> 存当前提示词 / prompt apply <名> 应用 / prompt del <名> 删除
  import <文件> 从 JSON 文件导入消息到当前会话（支持本程序导出的 chat_export.json）
  import merge <文件>  追加导入：把文件消息合并到当前对话末尾
  models        查询当前厂商可用模型列表（网页版模型输入框同步联想）
  ollama        查看本机 Ollama 状态（本地 GPU/CPU 推理，无需 API Key，附推荐模型）
  hw            查看硬件信息与自适应加速配置（CPU/内存/GPU/电源/调优结果）
  bench         运行快速基准测试（token 估算 / JSON / 磁盘写速度）
  latency       测量到当前 API 服务器的网络延迟
  suggest       推荐适合本机硬件的本地 Ollama 模型
  power         查看电源状态（电池供电时自动进入低功耗模式：降低轮询与保存频率）
  health        系统体检：Python 兼容性 / 硬件 / 预配内存 / 磁盘 / 网络 / 配置
  mem           查看本进程内存、系统内存预算与对话占用
  clear / c     清空当前对话（同步落盘）
  tokens / t    查看上下文占用
  save          手动保存存档
  load          重新加载存档
  history       查看历史消息列表
  export        导出对话为 Markdown；export json 导出为 JSON；export all 导出全部会话
  web           在浏览器中打开网页版（未配置 Key 时也用它填写）
  version / v   显示程序版本与环境信息
  help / ?      显示本帮助

厂商：
  deepseek / openai / moonshot / zhipu / qwen / ollama / openrouter
  默认厂商可用环境变量 CHAT_PROVIDER 指定；Key 用各厂商专用环境变量
  （如 OPENAI_API_KEY），或统一填在文件顶部 API_KEY。

未配置 Key / 地址 / 模型？
  程序不会退出：打开网页版（输入 web），在『系统提示词 / 高级设置』面板
  填写 API Key（可选：自定义 Base URL、模型）并点『应用设置』，
  确认后网页版与终端命令行即可同时对话，配置保存在 chat_config.json。

技巧：
  · 行尾输入 \\ 可继续输入下一行（方便粘贴多行代码）
  · 输入 Ctrl+C 中断当前回复；在输入提示处按 Ctrl+C（或 Ctrl+D）保存并退出
  · 网页版与终端 CLI 互通：共用同一份对话、厂商/模型/提示词/温度设置
  · 上下文接近上限时，早期对话会自动压缩为摘要保留（可在配置中关闭）
"""

# ===== 网络层（把网卡用足，按 CPU 核数动态缩放） =====
# HTTP/2：单连接多路复用 + 头部压缩（需 h2 包，已随本脚本启用）
try:
    import h2  # noqa: F401
    _HAS_H2 = True
except ImportError:
    _HAS_H2 = False

# ===== 硬件自适应调优（按 CPU 核数 + 内存大小动态缩放） =====
# 弱机（≤2 核 或 ≤4GB 内存）：缩小连接池与网页并发、降低自动保存频率，保证流畅不卡顿；
# 强机：把网卡与多核用足，更多并发连接 + 更高网页并发线程。
def _get_ram_gb():
    """探测物理内存大小（GB）：Windows 用 ctypes，Linux/macOS 读系统信息；失败返回 None"""
    try:
        if os.name == "nt":
            import ctypes
            class _MSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            m = _MSEX(); m.dwLength = ctypes.sizeof(_MSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return round(m.ullTotalPhys / (1024 ** 3), 1)
        elif sys.platform.startswith("linux"):
            with open("/proc/meminfo", "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return round(int(line.split()[1]) / (1024 ** 2), 1)
        elif sys.platform == "darwin":
            import subprocess
            r = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=3)
            if r.returncode == 0:
                return round(int(r.stdout.strip()) / (1024 ** 3), 1)
    except Exception:
        pass
    return None


def _is_32bit():
    """是否 32 位系统/解释器（32 位 Windows 最多寻址 ~3.2GB 内存，必须按此预配）"""
    try:
        import ctypes
        return ctypes.sizeof(ctypes.c_void_p) == 4
    except Exception:
        try:
            return platform.architecture()[0] == "32bit"
        except Exception:
            return False


def _windows_legacy():
    """是否 Windows 7 / 8 / 8.1（内核 6.x：无 VT 终端、Python 3.9+ 不支持、内存预留 1GB）"""
    if os.name != "nt":
        return False
    try:
        return sys.getwindowsversion().major < 10
    except Exception:
        return False


def _windows_vt_supported():
    """Windows 10 1511+（build 10586）才支持终端 VT 转义；Win7/8 开启会显示乱码转义符"""
    if os.name != "nt":
        return False
    try:
        v = sys.getwindowsversion()
        return v.major > 10 or (v.major == 10 and v.build >= 10586)
    except Exception:
        return False


def get_cpu_ghz():
    """探测 CPU 主频（GHz）：Windows 读注册表，Linux 读 /proc/cpuinfo，macOS 用 sysctl"""
    try:
        if os.name == "nt":
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                mhz = winreg.QueryValueEx(k, "~MHz")[0]
                if mhz:
                    return round(int(mhz) / 1000, 2)
        elif sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.startswith("cpu MHz"):
                        return round(float(line.split(":")[1].strip()) / 1000, 2)
        elif sys.platform == "darwin":
            import subprocess
            r = subprocess.run(["sysctl", "-n", "hw.cpufrequency"], capture_output=True, text=True, timeout=3)
            if r.returncode == 0:
                return round(int(r.stdout.strip()) / 1e9, 2)
    except Exception:
        pass
    return None


def usable_ram_gb():
    """可用内存（GB）= 总内存 - 系统预留 - 32位系统上限。

    预配内存：系统自身要占内存（Win10/11 预留 2GB，Win7/8 预留 1GB，Linux 0.8GB），
    32 位系统最多 3.2GB——这些都不能分给对话上下文，否则弱机直接卡死。
    """
    total = _get_ram_gb()
    if total is None:
        return None
    if _is_32bit():
        total = min(total, 3.2)
    reserve = 2.0 if (os.name == "nt" and not _windows_legacy()) else 1.0
    if sys.platform.startswith("linux"):
        reserve = 0.8
    elif sys.platform == "darwin":
        reserve = 1.2
    return max(0.5, total - reserve)


_cpu_cores = os.cpu_count() or 2
_ram_gb = _get_ram_gb()
_usable_ram = usable_ram_gb()
_slow_hw = _cpu_cores <= 2 or (_ram_gb is not None and _ram_gb <= 4)  # 弱硬件标记
_low_mem = _usable_ram is not None and _usable_ram < 3.0             # 低内存模式（可用 < 3GB）
_pool_max_conn = max(4, min(64, _cpu_cores * 8))   # 连接池大小随 CPU 核数自适应
_pool_max_keep = max(2, min(32, _cpu_cores * 4))
_web_threads = max(4, min(64, _cpu_cores * 4))     # 网页并发请求线程数
_SAVE_EVERY = 3 if _low_mem else (2 if _slow_hw else 1)  # 低内存机保存频率更低
_gz_level = 2 if _slow_hw else 4  # gzip 压缩级别：弱机优先 CPU 速度，强机优先压缩率
if _usable_ram is not None and _usable_ram < 4:
    _pool_max_conn = min(_pool_max_conn, 8)        # 内存紧张时收紧连接池，防内存峰值
if _low_mem:
    _pool_max_conn = min(_pool_max_conn, 6)        # 低内存模式：连接池最小化

# httpcore 默认已对每个 TCP 连接开启 TCP_NODELAY（关闭 Nagle 算法）
_transport = httpx.HTTPTransport(
    http2=_HAS_H2,                   # HTTP/2 协商失败会自动回退 HTTP/1.1
    retries=1,                       # 传输层连接级自动重试（建立连接失败时）
    limits=httpx.Limits(max_connections=_pool_max_conn, max_keepalive_connections=_pool_max_keep, keepalive_expiry=60.0),
)
http_client = httpx.Client(
    transport=_transport,
    timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=15.0),
)

_clients = {}


def get_client(name=None):
    """惰性创建 OpenAI 客户端（按厂商缓存；Key 未配置时抛 ConfigError 提示去网页版填写）。

    本地服务（Ollama）无需 Key：自动用占位 Key 建客户端，避免误报“未配置 API Key”。
    """
    name = name or current_provider
    if name not in _clients:
        key = resolve_api_key(name)
        if not key and not provider_meta(name).get("local"):
            raise ConfigError(config_error_message() or "未配置 API Key")
        _clients[name] = OpenAI(
            api_key=key or "sk-local",   # 本地服务无鉴权，占位即可
            base_url=provider_base(name),
            http_client=http_client,
            max_retries=0,               # 关闭 SDK 静默重试：统一走本脚本的重试策略（可见、可中断、尊重 Retry-After）
        )
    return _clients[name]


session = requests.Session()  # 余额查询也复用连接

messages = []

# token 增量账本：每轮只算新增消息，避免对大历史全量重扫（本地最大 CPU 开销）
_tokens_total = 0
_save_lock = threading.RLock()  # 串行化保存（可重入：_write_store 内部也加锁，防并发写坏存档）
_chat_lock = threading.Lock()  # 串行化对话回合（CLI 与网页版共用同一份对话）


_hw_info = None  # 硬件信息缓存（探测有开销，只算一次）


def detect_hardware():
    """检测本机可用硬件（CPU 架构/型号/核心数、内存、GPU 与显存、推理后端）。

    结果用于自适应调优：连接池大小、网页并发线程数、自动保存频率、
    本地 Ollama 模型推荐（按显存/内存估算，无需手动选）。
    """
    info = {
        "cpu_cores": os.cpu_count() or 1,
        "cpu_ghz": get_cpu_ghz(),
        "arch": platform.machine() or "unknown",
        "cpu": (platform.processor() or "").strip() or "unknown",
        "os": platform.system() or "unknown",
        "ram_gb": _get_ram_gb(),
        "gpus": [],
        "gpu_backend": None,
        "vram_gb": None,
        "slow_hw": _slow_hw,
    }
    # NVIDIA / AMD-ROCm / 其他 CUDA 兼容 GPU：优先 nvidia-smi（不依赖 torch，带显存）
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        for line in r.stdout.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if parts and parts[0]:
                info["gpus"].append(parts[0])
                if len(parts) > 1:
                    try:
                        info["vram_gb"] = max(info["vram_gb"] or 0, int(parts[1].split()[0]) / 1024)
                    except (ValueError, IndexError):
                        pass
        if info["gpus"]:
            info["gpu_backend"] = "CUDA"
    except Exception:
        pass
    # torch 兜底（CUDA / Apple Silicon MPS 统一内存）
    try:
        import importlib.util
        if importlib.util.find_spec("torch") is not None:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    if len(info["gpus"]) <= i:
                        info["gpus"].append(torch.cuda.get_device_name(i))
                    try:
                        vram = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                        info["vram_gb"] = max(info["vram_gb"] or 0, vram)
                    except Exception:
                        pass
                info.setdefault("gpu_backend", "CUDA")
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                # Apple Silicon（Mac）统一内存 GPU 加速
                info.setdefault("gpus", ["Apple Silicon (MPS)"])
                info.setdefault("gpu_backend", "MPS")
                info["vram_gb"] = info["vram_gb"] or info["ram_gb"]
    except Exception:
        pass
    return info


def suggest_ollama_model(vram_gb=None):
    """按显存（无独显时按内存）推荐本机 Ollama 最适合的模型：显存越大模型越大"""
    if vram_gb is None:
        vram_gb = get_hardware_info().get("vram_gb")
    if vram_gb is None:
        vram_gb = _usable_ram or _ram_gb or 8  # 无独显时按可用内存（已预留系统占用）估算
    if vram_gb >= 32:
        return "qwen2.5:32b"
    if vram_gb >= 16:
        return "qwen2.5:14b"
    if vram_gb >= 8:
        return "qwen2.5:7b"
    if vram_gb >= 4:
        return "qwen2.5:3b"
    return "qwen2.5:0.5b"


def get_hardware_info():
    """读取硬件信息（缓存，避免网页轮询时反复跑探测命令）"""
    global _hw_info
    if _hw_info is None:
        _hw_info = detect_hardware()
    return _hw_info


_ollama_cache = (None, 0.0)  # (探测结果, 时间戳)
_battery_cache = (None, 0.0)  # ((电池供电?, 电量%), 时间戳)


def get_power_status():
    """探测电源状态：返回 (是否电池供电, 电量百分比 or None)。

    Windows 用 GetSystemPowerStatus（ctypes 免依赖），Linux 读 /sys/class/power_supply，
    macOS 用 pmset。结果缓存 30 秒（网页轮询时不再反复探测）。
    """
    global _battery_cache
    now = time.time()
    if now - _battery_cache[1] < 30:
        return _battery_cache[0]
    result = (False, None)
    try:
        if os.name == "nt":
            import ctypes
            class _PS(ctypes.Structure):
                _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                            ("BatteryLifePercent", ctypes.c_ubyte), ("Reserved", ctypes.c_ubyte)]
            s = _PS()
            if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
                percent = s.BatteryLifePercent if 0 <= s.BatteryLifePercent <= 100 else None
                result = (s.ACLineStatus == 0, percent)
        elif sys.platform.startswith("linux"):
            import glob
            for bat in glob.glob("/sys/class/power_supply/BAT*"):
                try:
                    with open(bat + "/status", encoding="utf-8") as f:
                        if f.read().strip() == "Discharging":
                            cap = None
                            try:
                                with open(bat + "/capacity", encoding="utf-8") as f2:
                                    cap = int(f2.read().strip())
                            except Exception:
                                pass
                            result = (True, cap)
                            break
                except Exception:
                    pass
        elif sys.platform == "darwin":
            import subprocess
            r = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=3)
            if r.returncode == 0 and "discharging" in r.stdout.lower():
                result = (True, None)
    except Exception:
        pass
    _battery_cache = (result, now)
    return result


def power_save_active():
    """低功耗模式：电池供电时启用（降低轮询频率与保存频率，省电保流畅）"""
    return get_power_status()[0]


def detect_ollama(timeout=0.5):
    """探测本机 Ollama 服务（本地 GPU/CPU 推理）：返回 (是否可用, 模型列表)。

    结果缓存 30 秒，避免网页 10 秒轮询时反复探测。
    """
    global _ollama_cache
    now = time.time()
    if _ollama_cache[1] and now - _ollama_cache[1] < 30:
        return _ollama_cache[0]
    result = (False, [])
    try:
        resp = session.get("http://127.0.0.1:11434/api/tags", timeout=timeout)
        if resp.status_code == 200:
            models = [str(m.get("name", "")) for m in resp.json().get("models", [])]
            result = (True, [m for m in models if m])
    except Exception:
        pass
    _ollama_cache = (result, now)
    return result


def recount_tokens():
    """全量重算 token 账本（load / clear 后调用）"""
    global _tokens_total
    _tokens_total = sum(estimate_tokens(m.get("content", "") or "") for m in messages)
    return _tokens_total


def add_tokens(text):
    """增量累加 token 账本（每轮只算新增消息）"""
    global _tokens_total
    _tokens_total += estimate_tokens(text or "")
    return _tokens_total


def current_tokens():
    return _tokens_total


thinking_enabled = False
thinking_level = "medium"
system_prompt = ""    # 自定义系统提示词（system 命令 / 网页版可设置，随请求发送）
temperature = None    # 采样温度（None = 厂商默认；0-2 可调）
usage_total = {"prompt": 0, "completion": 0, "turns": 0}  # 累计用量统计


_token_cache = {}
_TOKEN_CACHE_MAX = 4096  # 估算缓存上限（只缓存短文本，防内存膨胀）


def estimate_tokens(text):
    """粗略估算 token 数：中文按字，英文按词（带 LRU 式缓存，长会话/自动裁剪提速）"""
    if not text:
        return 0
    if len(text) <= 2000:
        cached = _token_cache.get(text)
        if cached is not None:
            return cached
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    english_words = len(re.findall(r'[A-Za-z0-9]+', text))
    n = chinese_chars + int(english_words * 1.3)
    if len(text) <= 2000:
        if len(_token_cache) >= _TOKEN_CACHE_MAX:
            for k in list(_token_cache.keys())[: _TOKEN_CACHE_MAX // 2]:  # 清掉一半旧条目
                del _token_cache[k]
        _token_cache[text] = n
    return n


# ===== Markdown 输出美化（表格修复 + 加粗渲染） =====

def split_table_row(line):
    """按 | 拆分表格行（忽略 \\| 转义），返回单元格列表；不是表格行返回 None"""
    line = line.strip()
    if not line.startswith("|") or not line.endswith("|"):
        return None
    inner = line[1:-1]
    cells = []
    current = ""
    i = 0
    while i < len(inner):
        ch = inner[i]
        if ch == "\\" and i + 1 < len(inner) and inner[i + 1] == "|":
            current += "|"
            i += 2
            continue
        if ch == "|":
            cells.append(current.strip())
            current = ""
            i += 1
            continue
        current += ch
        i += 1
    cells.append(current.strip())
    return cells


def is_separator_row(cells):
    """是否为表头分隔行（如 |---|---| 或 |:---:|---|）"""
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", c) for c in cells)


def repair_markdown_table(lines):
    """修复一个 markdown 表格块：统一每行列数（少的补空单元格），重建分隔行。

    解决模型输出“上一行五格、下一行两格”导致的表格错位问题。
    """
    rows = [split_table_row(l) for l in lines]
    rows = [r for r in rows if r is not None]
    if not rows:
        return lines
    ncols = max(len(r) for r in rows)  # 以最宽行为准，保证所有行列数一致
    out = []
    for line in lines:
        cells = split_table_row(line)
        if cells is None:
            out.append(line)
            continue
        if is_separator_row(cells):
            out.append("|" + "|".join("---" for _ in range(ncols)) + "|")
        elif len(cells) == ncols:
            out.append(line)  # 已对齐的行保持原样
        else:
            padded = cells + [""] * (ncols - len(cells))
            out.append("| " + " | ".join(padded) + " |")
    return out


def is_table_row(line):
    """判断一行是否为表格行（缩进的代码块内容不算）"""
    return bool(line) and not line.startswith("    ") and line.lstrip().startswith("|")


def bold_table_header(lines):
    """把表格的表头行（分隔行上方第一行）自动加粗；
    表头单元格里已有的 ** 标记一并处理（整行已加粗，无需保留）"""
    if not USE_ANSI or len(lines) < 2:
        return lines
    first_cells = split_table_row(lines[0])
    second_cells = split_table_row(lines[1])
    if first_cells is None or second_cells is None or not is_separator_row(second_cells):
        return lines
    bolded = ["\033[1m" + c.replace("**", "") + "\033[0m" for c in first_cells]
    out = list(lines)
    out[0] = "| " + " | ".join(bolded) + " |"
    return out


class TableAwarePrinter:
    """流式打印器：表格行先缓存，等表格结束后统一修复列数再输出，
    保证用户看到的 markdown 表格每行列数一致、易于阅读。"""

    def __init__(self):
        self.line_buf = ""    # 未完整收到的行
        self.table_buf = []   # 已完整收到的表格行
        self.in_code = False  # 是否在 ``` 代码块内

    def feed(self, text):
        self.line_buf += text
        while "\n" in self.line_buf:
            line, self.line_buf = self.line_buf.split("\n", 1)
            self._flush_line(line.rstrip("\r"))

    def finish(self):
        if self.line_buf:
            self._flush_line(self.line_buf)
            self.line_buf = ""
        if self.table_buf:
            self._print_table()

    def _flush_line(self, line):
        if self.in_code:
            if line.strip().startswith("```"):
                self.in_code = False
            print(line, flush=True)
            return
        if line.strip().startswith("```"):
            if self.table_buf:
                self._print_table()
            self.in_code = True
            print(line, flush=True)
            return
        if is_table_row(line):
            self.table_buf.append(line)
        else:
            if self.table_buf:
                self._print_table()
            print(render_inline_markdown(line), flush=True)

    def _print_table(self):
        lines = bold_table_header(repair_markdown_table(self.table_buf))
        for l in lines:
            print(render_inline_markdown(l), flush=True)
        self.table_buf = []


# ANSI 转义只在直接输出到终端时启用（重定向到文件时不加转义符，保持纯文本；
# Win7/Win8 终端不解析 VT 转义，也自动关闭，避免输出乱码转义符）
USE_ANSI = sys.stdout.isatty() and (os.name != "nt" or _windows_vt_supported())


def enable_ansi_on_windows():
    """Windows 10+ 下启用终端 ANSI 转义支持，否则粗体不生效（Win7/8 自动跳过）"""
    if os.name != "nt" or not _windows_vt_supported():
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        pass


def render_inline_markdown(text):
    """行内 Markdown 渲染：**粗体**、`内联代码`（青色），其余原样保留（忽略转义）"""
    if not USE_ANSI or not text:
        return text
    text = re.sub(
        r"(?<!\\)\*\*(.+?)\*\*(?!\*)",
        lambda m: "\033[1m" + m.group(1) + "\033[0m",
        text,
    )
    return re.sub(
        r"(?<!\\)`([^`\n]+)`",
        lambda m: "\033[36m" + m.group(1) + "\033[0m",
        text,
    )


def check_balance():
    """查询账户余额（仅支持余额查询的厂商，如 DeepSeek）"""
    meta = provider_meta()
    if not meta.get("balance"):
        print(f"\n[提示] {meta['name']} 不支持余额查询（仅 DeepSeek 支持）")
        print("-" * 40)
        return
    err = config_error_message()
    if err:
        print(f"\n[配置缺失] {err}（网页版设置面板填写后立即可用）")
        print("-" * 40)
        return
    url = provider_base().rstrip("/") + "/user/balance"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {resolve_api_key()}"
    }

    print("\n[查询余额]")
    try:
        resp = session.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            infos = data.get("balance_infos") or []
            if not infos:
                print(f"  响应: {data}")
            else:
                for b in infos:
                    currency = b.get("currency", "?")
                    total = float(b.get("total_balance") or 0)
                    granted = float(b.get("granted_balance") or 0)
                    topped = float(b.get("topped_up_balance") or 0)
                    print(f"  {currency}: 余额 {total:.2f}（赠送 {granted:.2f} + 充值 {topped:.2f}）")
        elif resp.status_code == 401:
            print("  错误 401: API Key 无效")
        elif resp.status_code == 403:
            print("  错误 403: 无权限访问")
        elif resp.status_code == 429:
            print("  错误 429: 请求过于频繁")
        elif resp.status_code >= 500:
            print(f"  错误 {resp.status_code}: 服务器内部错误")
        else:
            print(f"  错误 {resp.status_code}: {resp.text[:200]}")
    except requests.exceptions.Timeout:
        print("  错误: 请求超时")
    except requests.exceptions.ConnectionError:
        print("  错误: 无法连接服务器")
    except Exception as e:
        print(f"  未知错误: {e}")
    print("-" * 40)


def is_retryable(exc):
    """判断异常是否值得重试（网络波动 / 限流 / 服务器临时错误）"""
    if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return True
    msg = str(exc).lower()
    return any(k in msg for k in (
        "429", "500", "502", "503", "504",
        "timeout", "timed out", "connection", "rate limit",
        "overloaded", "server error", "temporarily", "try again",
    ))


def get_retry_after(exc):
    """从异常响应中读取 Retry-After 头（秒）；读不到返回 None"""
    resp = getattr(exc, "response", None)
    if resp is not None and getattr(resp, "headers", None) is not None:
        h = resp.headers.get("retry-after")
        if h:
            try:
                return max(0.5, float(h))
            except (TypeError, ValueError):
                pass
    return None


def warm_up_connection():
    """后台预连接当前厂商 API 服务器：省去首轮请求的 TCP/TLS 握手延迟"""
    def _do():
        try:
            http_client.get(provider_base(), timeout=10)
        except Exception:
            pass  # 预热失败无妨，正式请求会正常建立连接

    threading.Thread(target=_do, daemon=True).start()


current_session = "default"  # 当前会话名（多会话管理，存档文件中按会话分存）


_store_cache = (0, None)  # (文件 mtime_ns, 内存中的存档) —— 避免每次轮询都重读大 JSON
_last_bak_ts = 0.0  # 上次创建 .bak 备份的时间（节流：每分钟最多一次）


def _read_store():
    """读取会话存储（新 schema）；旧版单会话格式自动迁移为 default 会话。

    带 mtime 缓存：网页版 10 秒轮询状态时不再反复读盘（长会话文件可达数 MB）。
    """
    global _store_cache
    try:
        try:
            mtime = os.stat(HISTORY_FILE).st_mtime_ns
        except OSError:
            return {"current": "default", "sessions": {}}
        if _store_cache[0] == mtime and _store_cache[1] is not None:
            return _store_cache[1]
        # utf-8-sig：兼容 Windows 编辑器写入的带 BOM 存档
        with open(HISTORY_FILE, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("sessions"), dict):
            store = data
        elif isinstance(data, dict) and "messages" in data:
            store = {"current": "default", "sessions": {"default": data}}  # 旧格式迁移
        else:
            store = {"current": "default", "sessions": {}}
        _store_cache = (mtime, store)
        return store
    except Exception:
        return {"current": "default", "sessions": {}}


def _write_store(store):
    """原子写入会话存储（内部加 _save_lock 串行化，防止与后台异步保存并发写坏）。

    大存档自动用紧凑格式写盘（省时间省内存），并保留上一份 .bak 备份。
    """
    global _store_cache, _last_bak_ts
    with _save_lock:  # RLock：调用方已持锁时重入安全
        tmp = HISTORY_FILE + ".tmp"
        pretty = json.dumps(store, ensure_ascii=False, indent=2)
        if len(pretty) > HISTORY_COMPACT_SIZE:
            pretty = json.dumps(store, ensure_ascii=False, separators=(",", ":"))  # 紧凑写盘
        if len(pretty) < 2_000_000 and os.path.exists(HISTORY_FILE) and time.time() - _last_bak_ts > 60:
            try:
                import shutil
                shutil.copyfile(HISTORY_FILE, HISTORY_FILE + ".bak")  # 上一份好存档（每分钟最多一次）
            except Exception:
                pass
            finally:
                _last_bak_ts = time.time()
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(pretty)
        # Windows 上文件句柄短暂占用（如杀毒/后台线程）时 os.replace 偶发拒绝，
        # 小退避重试几次再放弃（其他平台无影响）
        for _attempt in range(3):
            try:
                os.replace(tmp, HISTORY_FILE)
                break
            except OSError:
                if _attempt == 2:
                    raise
                time.sleep(0.05 * (_attempt + 1))
        try:
            _store_cache = (os.stat(HISTORY_FILE).st_mtime_ns, store)
        except OSError:
            pass


def persist_current_session():
    """把当前（可能为空的）会话写入存档：clear/new 后保证重启不复活旧消息"""
    try:
        with _save_lock:
            store = _read_store()
            store.setdefault("current", current_session)
            store.setdefault("sessions", {})[current_session] = _session_snapshot()
            _write_store(store)
    except Exception:
        pass


def _session_snapshot():
    """当前全局状态 → 一条会话记录"""
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "provider": current_provider,
        "model": current_model(),
        "system_prompt": system_prompt,
        "temperature": temperature,
        "thinking": thinking_enabled,
        "thinking_level": thinking_level,
        "usage": dict(usage_total),
        "messages": messages,
    }


def _apply_session_data(data):
    """一条会话记录 → 全局状态（消息/账本/提示词/温度/思考/用量/厂商/模型）"""
    global messages, _tokens_total, usage_total, system_prompt, temperature
    global current_provider, model_override, thinking_enabled, thinking_level
    messages = list(data.get("messages") or [])
    _tokens_total = recount_tokens()
    system_prompt = str(data.get("system_prompt") or "")
    temperature = data.get("temperature")
    if "thinking" in data:
        thinking_enabled = bool(data.get("thinking"))
    if data.get("thinking_level") in ("low", "medium", "high", "max"):
        thinking_level = data["thinking_level"]
    usage_total = dict(data.get("usage") or {"prompt": 0, "completion": 0, "turns": 0})
    saved_provider = str(data.get("provider") or "")
    if saved_provider in PROVIDERS:
        current_provider = saved_provider
    model_override = str(data.get("model") or "") or None


def _new_msg(role, content):
    """构造带时间戳的消息（网页版/导出显示真实发送时间）"""
    return {"role": role, "content": content,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S")}


def _valid_session_name(name):
    """会话名合法性：非空、≤32 字符、不含路径/特殊字符"""
    return bool(name) and len(name) <= 32 and not any(ch in name for ch in '/\\:?*"<>|')


def list_sessions():
    """所有会话名列表"""
    return sorted(_read_store().get("sessions", {}).keys())


def save_history(quiet=False):
    """保存当前会话到存档（加锁串行化 + 原子写入）"""
    with _save_lock:
        _save_history_locked(quiet)


def _save_history_locked(quiet):
    global messages
    try:
        store = _read_store()
        store.setdefault("current", current_session)
        store.setdefault("sessions", {})[current_session] = _session_snapshot()
        _write_store(store)
        if not quiet:
            print(f"[已保存] 会话「{current_session}」{len(messages)} 条消息，"
                  f"约 {current_tokens():,} tokens → {HISTORY_FILE}")
    except Exception as e:
        if not quiet:
            print(f"[保存失败] {e}")


_turn_count = 0  # 自动保存轮次计数（弱机长会话时降频落盘）


def save_history_async(force=False):
    """后台线程保存历史：与用户输入并行，不阻塞主流程；已有保存进行中则跳过本轮。

    弱机 + 长会话时自动降频（每 _SAVE_EVERY 轮落盘一次，省 CPU 保证流畅）；
    关键路径（客户端断开/停止生成/退出/切换）传 force=True 必存。

    注意：_save_lock 是线程绑定的 RLock，获取与释放必须在同一线程完成，
    因此锁在工作线程内部获取（绝不能在主线程获取、交给工作线程释放）。
    """
    global _turn_count
    _turn_count += 1
    if not force and (_slow_hw or _low_mem or power_save_active()) and current_tokens() > 100_000:
        if _turn_count % _SAVE_EVERY != 0:
            return

    def _do():
        if not _save_lock.acquire(blocking=False):
            return  # 已有保存进行中则跳过本轮
        try:
            _save_history_locked(True)
        finally:
            _save_lock.release()

    threading.Thread(target=_do, daemon=True).start()


def load_history():
    """从存档加载当前会话（含提示词/温度/用量/厂商设置）"""
    global current_session
    if not os.path.exists(HISTORY_FILE):
        print(f"[未找到存档] {HISTORY_FILE} 不存在")
        return

    try:
        store = _read_store()
        name = store.get("current") or "default"
        data = (store.get("sessions") or {}).get(name)
        if data is None or not data.get("messages"):
            print("[存档为空]")
            return

        with _chat_lock:
            current_session = name
            _apply_session_data(data)
        timestamp = data.get("timestamp", "未知")
        print(f"[已加载] 会话「{name}」· {timestamp}"
              f"（厂商: {data.get('provider') or '?'}，模型: {data.get('model') or '?'}）")
        print(f"         消息条数: {len(messages)}，约 {current_tokens():,} tokens")
    except Exception as e:
        print(f"[加载失败] {e}")


def create_session(name):
    """新建会话并切换（调用方需持有 _chat_lock）"""
    global current_session, messages, _tokens_total, usage_total
    name = name.strip()
    if not _valid_session_name(name):
        print(f"[错误] 非法会话名: {name!r}（≤32 字符，不含 / \\ : ? * \" < > |）")
        return False
    if messages:
        save_history(quiet=True)
    store = _read_store()
    snap = _session_snapshot()
    snap["messages"] = []
    snap["usage"] = {"prompt": 0, "completion": 0, "turns": 0}
    store["sessions"][name] = snap
    store["current"] = name
    _write_store(store)
    current_session = name
    messages = []
    _tokens_total = 0
    usage_total = {"prompt": 0, "completion": 0, "turns": 0}
    print(f"[已切换到新会话] 「{name}」")
    return True


def switch_session(name):
    """切换到已有会话（调用方需持有 _chat_lock）"""
    global current_session
    name = name.strip()
    if name == current_session:
        return True
    store = _read_store()
    if name not in store.get("sessions", {}):
        print(f"[错误] 会话不存在: {name}")
        return False
    if messages:
        save_history(quiet=True)
    store = _read_store()  # 保存后重新读取：确保拿到当前会话刚写入的最新数据
    sessions = store.get("sessions", {})
    if name not in sessions:
        print(f"[错误] 会话不存在: {name}")
        return False
    store["current"] = name
    _write_store(store)
    current_session = name
    _apply_session_data(sessions[name])
    print(f"[已切换到会话] 「{name}」（{len(messages)} 条消息）")
    return True


def delete_session(name):
    """删除会话（调用方需持有 _chat_lock；当前会话不可删）"""
    name = name.strip()
    if name == current_session:
        print("[错误] 不能删除当前会话（先切换到其他会话）")
        return False
    store = _read_store()
    if name not in store.get("sessions", {}):
        print(f"[错误] 会话不存在: {name}")
        return False
    del store["sessions"][name]
    if store.get("current") == name:
        # 存档里的 current 指针不能再指向被删的会话（否则重启后会误报“存档为空”）
        store["current"] = current_session
    _write_store(store)
    print(f"[已删除会话] 「{name}」")
    return True


def copy_session(new_name):
    """复制当前会话为新会话（调用方需持有 _chat_lock）"""
    global current_session
    new_name = new_name.strip()
    if not _valid_session_name(new_name):
        print(f"[错误] 非法会话名: {new_name!r}（≤32 字符，不含 / \\ : ? * \x22 < > |）")
        return False
    if new_name == current_session:
        print("[错误] 新名称不能与当前会话相同")
        return False
    store = _read_store()
    if new_name in store.get("sessions", {}):
        print(f"[错误] 会话已存在: {new_name}")
        return False
    snap = _session_snapshot()
    snap["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    store.setdefault("sessions", {})[new_name] = snap
    _write_store(store)
    print(f"[已复制] 「{current_session}」→「{new_name}」（{len(messages)} 条消息）")
    return True


def rename_session(name):
    """重命名当前会话（调用方需持有 _chat_lock）"""
    global current_session
    name = name.strip()
    if not _valid_session_name(name):
        print(f"[错误] 非法会话名: {name!r}（≤32 字符，不含 / \\ : ? * \" < > |）")
        return False
    if name == current_session:
        return True
    store = _read_store()
    sessions = store.get("sessions", {})
    if current_session not in sessions:
        return False
    sessions[name] = sessions.pop(current_session)
    store["current"] = name
    _write_store(store)
    old = current_session
    current_session = name
    print(f"[已重命名] 「{old}」→「{name}」")
    return True


# ===== 提示词模板库（CLI 与网页版共用，存于 prompt_templates.json） =====

_templates_cache = (0, None)  # (文件 mtime_ns, 模板库) —— 状态轮询不再反复读盘


def load_prompt_templates():
    """读取提示词模板库（带 mtime 缓存）"""
    global _templates_cache
    try:
        mtime = os.stat(PROMPT_FILE).st_mtime_ns
    except OSError:
        # 首次运行：文件不存在时自动创建内置预置模板（翻译/代码审查/总结/润色）
        save_prompt_templates(PROMPT_PRESETS)
        try:
            mtime = os.stat(PROMPT_FILE).st_mtime_ns
        except OSError:
            mtime = 0
        _templates_cache = (mtime, dict(PROMPT_PRESETS))
        return dict(PROMPT_PRESETS)
    if _templates_cache[0] == mtime and _templates_cache[1] is not None:
        return _templates_cache[1]
    data = {}
    try:
        with open(PROMPT_FILE, "r", encoding="utf-8-sig") as f:
            d = json.load(f)
        if isinstance(d, dict):
            data = d
    except Exception:
        pass
    _templates_cache = (mtime, data)
    return data


def save_prompt_templates(templates):
    """原子写入模板库（同时更新缓存）"""
    global _templates_cache
    try:
        tmp = PROMPT_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(templates, f, ensure_ascii=False, indent=2)
        os.replace(tmp, PROMPT_FILE)
        try:
            _templates_cache = (os.stat(PROMPT_FILE).st_mtime_ns, dict(templates))
        except OSError:
            pass
        return True
    except Exception:
        return False


def prompt_save(name, text):
    """把提示词存为模板"""
    name = name.strip()
    if not _valid_session_name(name) or not text.strip():
        return False
    templates = load_prompt_templates()
    templates[name] = text
    return save_prompt_templates(templates)


def prompt_apply(name):
    """把模板应用到系统提示词"""
    global system_prompt
    templates = load_prompt_templates()
    if name not in templates:
        return False
    system_prompt = templates[name]
    return True


def prompt_delete(name):
    """删除模板"""
    templates = load_prompt_templates()
    if name not in templates:
        return False
    del templates[name]
    return save_prompt_templates(templates)


def read_multiline_input():
    """读取用户输入；行尾以 \\ 结尾时继续读取下一行（方便粘贴多行代码）。

    兼容管道输入（echo | python chat2.py 时 input 会带回车符，一并去除）。
    """
    first = input("\n你: ").rstrip("\r\n")
    if not first.rstrip().endswith("\\"):
        return first
    parts = [first.rstrip()[:-1]]
    while True:
        more = input("…  ").rstrip("\r\n")
        if more.rstrip().endswith("\\"):
            parts.append(more.rstrip()[:-1])
            continue
        parts.append(more)
        break
    return "\n".join(parts)


def build_export_markdown():
    """把当前对话拼成 Markdown 文本（CLI 导出与网页下载共用）"""
    role_names = {"user": "你", "assistant": "AI", "system": "系统"}
    lines = [f"# 对话记录（{provider_meta()['name']} · {current_model()}）",
             "", f"- 导出时间: {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    for m in messages:
        role = role_names.get(m.get("role"), str(m.get("role")))
        ts = m.get("ts")
        head = role + (f"（{ts}）" if ts else "")
        lines += [f"## {head}", "", str(m.get("content") or ""), ""]
    return "\n".join(lines)


def export_all_sessions():
    """导出全部会话为一个 Markdown 文件（含时间戳）"""
    store = _read_store()
    sessions = store.get("sessions") or {}
    if not sessions:
        print("[导出失败] 没有会话")
        return
    lines = ["# 全部会话导出", "", f"- 导出时间: {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    for name, sdata in sessions.items():
        msgs = (sdata or {}).get("messages") or []
        lines += [f"## 会话「{name}」（{len(msgs)} 条）", ""]
        for m in msgs:
            role = {"user": "你", "assistant": "AI", "system": "系统"}.get(m.get("role"), str(m.get("role")))
            ts = m.get("ts")
            head = role + (f"（{ts}）" if ts else "")
            lines += [f"### {head}", "", str(m.get("content") or ""), ""]
    fname = EXPORT_FILE.replace(".md", f"_{time.strftime('%Y-%m-%d')}_all.md")
    try:
        with open(fname, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[已导出] 全部 {len(sessions)} 个会话 → {fname}")
    except Exception as e:
        print(f"[导出失败] {e}")


def export_history(fmt="md"):
    """把当前对话导出为 Markdown（默认）或 JSON 文件（文件名带日期）"""
    if not messages:
        print("[导出失败] 对话为空")
        return
    try:
        date = time.strftime("%Y-%m-%d")
        if fmt == "json":
            fname = EXPORT_FILE.replace(".md", f"_{date}.json")
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(messages, f, ensure_ascii=False, indent=2)
            print(f"[已导出] {len(messages)} 条消息 → {fname}（JSON，可再次导入/处理）")
        else:
            fname = EXPORT_FILE.replace(".md", f"_{date}.md")
            with open(fname, "w", encoding="utf-8") as f:
                f.write(build_export_markdown())
            print(f"[已导出] {len(messages)} 条消息 → {fname}")
    except Exception as e:
        print(f"[导出失败] {e}")


def summarize_conversation(msgs):
    """把将被裁剪的早期对话压缩为中文摘要；失败返回 None（调用方退回直接删除）"""
    try:
        parts = []
        for m in msgs:
            role = "用户" if m.get("role") == "user" else "AI"
            parts.append(f"{role}: {str(m.get('content') or '')[:1500]}")
        text = "\n".join(parts)
        if len(text) > 6000:
            text = text[-6000:]
        resp = get_client().chat.completions.create(
            model=current_model(),
            messages=[
                {"role": "system", "content": "你是对话压缩助手。请用简洁的中文把下面的对话压缩成要点摘要（3-6 句话），保留关键事实、结论与未完成事项，不要复述原文。"},
                {"role": "user", "content": text},
            ],
            stream=False,
            max_tokens=512,
        )
        summary = (resp.choices[0].message.content or "").strip()
        return summary or None
    except Exception:
        return None


def _is_summary_msg(m):
    """判断是否为受保护的摘要消息（自动裁剪/手工删除时不被删除）"""
    c = str(m.get("content") or "")
    return m.get("role") == "system" and (c.startswith("【早前对话摘要】") or c.startswith("【对话摘要】"))


def effective_trim_threshold():
    """实际裁剪阈值 = min(配置上限, 厂商窗口×90%, 按可用内存收紧的上限)。

    可用内存 = 总内存 - 系统预留（Win10/11 预留 2GB，Win7/8 预留 1GB；32 位封顶 3.2GB）。
    每 GB 可用内存约分配 96k tokens：Win7 32 位 + 2GB 内存的机器自动压到 ~100k tokens，
    防止长会话把内存占满导致卡死——这是预配内存的硬件自适应。
    """
    ram_cap = max(32_000, int((_usable_ram or 8) * 96_000))
    return min(AUTO_TRIM_THRESHOLD, int(current_window() * 0.9), ram_cap)


def web_history_limit():
    """网页版一次加载的消息条数：低内存 100 / 弱机 200 / 正常 500"""
    if _low_mem:
        return 100
    return 200 if _slow_hw else HISTORY_LIMIT_WEB


def auto_trim():
    """上下文快满时，自动裁剪最早的对话。

    基于 token 增量账本，账本超限才扫描历史；
    用前缀和 + 二分找出需删除的条数后一次切片完成（O(n)）。
    若开启 SUMMARIZE_ON_TRIM，被删的早期对话会先交给 API 压缩成
    一条 system 摘要保留在上下文里（失败自动退回直接删除）。
    """
    global messages, _tokens_total
    threshold = effective_trim_threshold()  # 按内存收紧的自适应裁剪阈值
    if _tokens_total <= threshold or len(messages) <= 2:
        return
    sizes = [estimate_tokens(m.get("content", "") or "") for m in messages]
    prefix = [0]
    for s in sizes:
        prefix.append(prefix[-1] + s)
    # 首条是受保护的摘要时，从第 2 条开始裁
    start = 1 if _is_summary_msg(messages[0]) else 0
    lo, hi = start, start + len(sizes) - 2  # 最多删到只剩 2 条（摘要保护前缀不参与删除）
    if lo > hi:
        return
    while lo < hi:
        mid = (lo + hi) // 2
        if prefix[-1] - prefix[mid] <= threshold:
            hi = mid
        else:
            lo = mid + 1
    drop = lo
    if drop <= start:
        return
    dropped = messages[start:drop]
    kept_prefix = messages[:start]
    removed_tokens = prefix[drop] - prefix[start]
    remaining = prefix[-1] - removed_tokens

    summary_msg = None
    if SUMMARIZE_ON_TRIM and len(dropped) >= 2:
        print(f"[自动裁剪] 正在把最早 {len(dropped)} 条消息压缩为摘要…", flush=True)
        summary = summarize_conversation(dropped)
        if summary:
            summary_msg = {"role": "system", "content": "【早前对话摘要】" + summary}

    if summary_msg:
        messages = [summary_msg] + messages[drop:]
        _tokens_total = remaining + estimate_tokens(summary_msg["content"])
        print(f"[自动裁剪] 已将最早 {len(dropped)} 条消息压缩为摘要，释放上下文空间")
    else:
        messages = kept_prefix + messages[drop:]
        _tokens_total = remaining
        print(f"[自动裁剪] 已移除最早 {len(dropped)} 条消息，释放上下文空间")


def stream_reply(thinking=False, level="medium"):
    """流式生成回复（CLI 与网页版共用，跟随当前厂商/模型设置）。

    基于当前 messages（最后一条为刚加入的用户消息），产出事件字典：
      {"type": "retry", ...}            重试提示
      {"type": "reasoning", "text"}     思考链增量（仅支持的厂商）
      {"type": "content", "text"}       回复内容增量
      {"type": "done", "content", "stats"}  结束（含完整回复与统计）
    请求彻底失败时抛出异常（调用方负责回滚用户消息与账本）。
    """
    meta = provider_meta()
    msgs = messages
    if system_prompt:
        # 自定义系统提示词放在最前（不影响裁剪/摘要等既有结构）
        msgs = [{"role": "system", "content": system_prompt}] + list(messages)
    kwargs = {
        "model": current_model(),
        "messages": msgs,
        "stream": True,
    }
    if temperature is not None:
        kwargs["temperature"] = float(temperature)
    if meta.get("stream_usage", True):
        kwargs["stream_options"] = {"include_usage": True}
    if thinking and meta.get("thinking"):
        kwargs["extra_body"] = {
            "thinking": {"type": "enabled"},
            "thinking_level": level,
        }

    start = time.time()
    first_token_time = None
    ttft = None
    final_usage = None
    assistant_content = ""

    # 自动重试：网络波动 / 限流 / 服务器临时错误时指数退避（优先尊重 Retry-After）
    stream = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            stream = get_client().chat.completions.create(**kwargs)
            break
        except Exception as e:
            if attempt >= MAX_RETRIES or not is_retryable(e):
                raise
            wait = get_retry_after(e) or (RETRY_BACKOFF * (2 ** (attempt - 1)))
            wait *= random.uniform(0.8, 1.2)  # 抖动：避免多个客户端同时重试（惊群）
            yield {"type": "retry", "attempt": attempt, "max": MAX_RETRIES,
                   "message": str(e), "wait": wait}
            time.sleep(wait)

    for chunk in stream:
        if getattr(chunk, "usage", None):
            final_usage = chunk.usage
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        reasoning = getattr(delta, "reasoning_content", None) or ""
        if reasoning:
            yield {"type": "reasoning", "text": reasoning}

        content = delta.content or ""
        if content:
            if first_token_time is None:
                first_token_time = time.time()
                ttft = first_token_time - start
            assistant_content += content
            yield {"type": "content", "text": content}

    stats = {
        "ttft_ms": (ttft * 1000) if ttft is not None else None,
        "total_sec": time.time() - start,
        "prompt_tokens": final_usage.prompt_tokens if final_usage else None,
        "completion_tokens": final_usage.completion_tokens if final_usage else None,
    }
    # 累计用量（CLI 与网页版共用同一份统计）
    if final_usage:
        global usage_total
        usage_total["prompt"] += final_usage.prompt_tokens
        usage_total["completion"] += final_usage.completion_tokens
        usage_total["turns"] += 1
    yield {"type": "done", "content": assistant_content, "stats": stats}


def search_messages(keyword, limit=20):
    """在对话历史中搜索关键词，返回 [(序号, 角色, 摘要片段), ...]"""
    kw = keyword.lower()
    role_names = {"user": "你", "assistant": "AI", "system": "系统"}
    out = []
    for i, m in enumerate(messages):
        content = str(m.get("content") or "")
        if kw not in content.lower():
            continue
        role = role_names.get(m.get("role"), str(m.get("role")))
        idx = content.lower().find(kw)
        start = max(0, idx - 40)
        end = min(len(content), idx + len(keyword) + 40)
        snippet = ("…" if start > 0 else "") + content[start:end] + ("…" if end < len(content) else "")
        out.append((i + 1, role, snippet))
        if len(out) >= limit:
            break
    return out


def usage_summary():
    """累计用量统计 + 参考成本估算（按配置单价）"""
    s = dict(usage_total)
    s["cost"] = (usage_total["prompt"] / 1_000_000 * PRICE_INPUT) + \
                (usage_total["completion"] / 1_000_000 * PRICE_OUTPUT)
    return s


def get_disk_free_mb():
    """当前工作目录所在磁盘的剩余空间（MB）；失败返回 None"""
    try:
        if os.name == "nt":
            import ctypes
            free = ctypes.c_ulonglong(0)
            if ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                    ctypes.c_wchar_p(os.getcwd() + "\\"), None, None, ctypes.byref(free)):
                return free.value / (1024 * 1024)
        else:
            st = os.statvfs(os.getcwd())
            return st.f_bavail * st.f_frsize / (1024 * 1024)
    except Exception:
        pass
    return None


def get_process_memory_mb():
    """当前进程占用的物理内存（MB）；失败返回 None"""
    try:
        if os.name == "nt":
            import ctypes
            class _PMC(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ]
            pmc = _PMC()
            pmc.cb = ctypes.sizeof(_PMC)
            if ctypes.windll.psapi.GetProcessMemoryInfo(
                    ctypes.c_void_p(ctypes.windll.kernel32.GetCurrentProcess()),
                    ctypes.byref(pmc), pmc.cb):
                return pmc.WorkingSetSize / (1024 * 1024)
        elif sys.platform.startswith("linux"):
            with open("/proc/self/status", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024
    except Exception:
        pass
    return None


def get_process_cpu():
    """当前进程 CPU 占用率（%）：Windows GetProcessTimes / Linux /proc/self/stat。

    返回 (占用率, 采样间隔秒)；首次调用返回 (None, 0)。用于弱机负载监测：
    持续高占用时网页自动降频轮询与保存。
    """
    global _cpu_sample
    now = time.time()
    try:
        if os.name == "nt":
            import ctypes
            class _FT(ctypes.Structure):
                _fields_ = [("dwLowDateTime", ctypes.c_ulong), ("dwHighDateTime", ctypes.c_ulong)]
            k32 = ctypes.windll.kernel32
            h = k32.GetCurrentProcess()
            # 用命名临时对象传参（byref 不能引用即将回收的临时结构体）
            ct, et = _FT(), _FT()
            kt, ut = _FT(), _FT()
            if not k32.GetProcessTimes(ctypes.c_void_p(h), ctypes.byref(ct), ctypes.byref(et),
                                        ctypes.byref(kt), ctypes.byref(ut)):
                return None, 0
            total = (kt.dwHighDateTime << 32 | kt.dwLowDateTime) + (ut.dwHighDateTime << 32 | ut.dwLowDateTime)
            total /= 1e7  # 100ns -> 秒
        elif sys.platform.startswith("linux"):
            with open("/proc/self/stat", encoding="utf-8") as f:
                parts = f.read().rsplit(")", 1)[1].split()
            total = (int(parts[11]) + int(parts[12])) / 100.0  # utime+stime，单位 jiffies（通常 100/s）
        else:
            return None, 0
    except Exception:
        return None, 0
    prev, prev_ts = _cpu_sample
    _cpu_sample = (total, now)
    if prev is None or now - prev_ts < 1.0:
        return None, 0
    pct = (total - prev) / (now - prev_ts) * 100.0
    return round(min(100.0, max(0.0, pct)), 1), round(now - prev_ts, 1)


def health_check():
    """health 命令：系统体检（Python 兼容性 / 硬件 / 预配内存 / 磁盘 / 网络 / 配置）"""
    print("\n[系统体检] 运行环境与配置检查")
    rows = []
    pyver = sys.version_info
    if pyver < (3, 8):
        rows.append(("✗", f"Python {pyver[0]}.{pyver[1]} 过旧：Windows 7 请安装 Python 3.8.10（最后支持 Win7 的版本），其他系统建议 3.8+"))
    else:
        note = "（注意：Python 3.9+ 不再支持 Windows 7，Win7 请停留在 3.8.10）" if (os.name == "nt" and _windows_legacy() and pyver >= (3, 9)) else ""
        rows.append(("✓", f"Python {pyver[0]}.{pyver[1]}.{pyver[2]} 兼容{note}"))
    bits = "32 位" if _is_32bit() else "64 位"
    rows.append(("✓", f"{platform.system()} {platform.release()} · {bits} · CPU {_cpu_cores} 核"))
    if _usable_ram is not None:
        cap_k = effective_trim_threshold() // 1000
        if _low_mem:
            rows.append(("⚠", f"可用内存仅 {_usable_ram:.1f} GB（已预留系统占用）→ 低内存模式：上下文上限 {cap_k}k tokens、网页历史 100 条、保存每 {_SAVE_EVERY} 轮"))
        else:
            rows.append(("✓", f"可用内存 {_usable_ram:.1f} GB（已预留系统占用）→ 上下文上限 {cap_k}k tokens"))
    free = get_disk_free_mb()
    if free is not None:
        if free < 50:
            rows.append(("✗", f"磁盘剩余仅 {free:.0f} MB，历史存档可能写失败，请清理磁盘！"))
        elif free < 200:
            rows.append(("⚠", f"磁盘剩余 {free:.0f} MB（建议 ≥200MB）"))
        else:
            rows.append(("✓", f"磁盘剩余 {free:.0f} MB"))
    hw = get_hardware_info()
    if hw.get("gpus"):
        rows.append(("✓", f"GPU: {', '.join(hw['gpus'])}（后端 {hw.get('gpu_backend')}）"))
    else:
        rows.append(("·", "GPU: 未检测到（云端推理不受影响；本地推理可用 Ollama）"))
    ok, models = detect_ollama()
    rows.append(("✓" if ok else "·", "Ollama: " + (f"可用（{', '.join(models) or '无模型'}）" if ok else "未安装（可选）")))
    net = probe_network_latency()
    lat = net.get("latency_ms")
    rows.append(("✓" if lat is not None else "·", f"API 延迟: {lat:.0f} ms" if lat is not None else "API 延迟: 未配置 Key 或暂不可测"))
    err = config_error_message()
    rows.append(("✓" if not err else "⚠", "API Key: " + ("已配置" if not err else err)))
    for mark, text in rows:
        print(f"  {mark} {text}")
    print("-" * 40)


def show_memory():
    """mem 命令：当前进程内存/CPU + 系统内存预算 + 对话内存估算"""
    print("\n[内存] 当前资源占用")
    pm = get_process_memory_mb()
    if pm is not None:
        print(f"  本进程内存: {pm:.1f} MB")
    pcpu, span = get_process_cpu()
    if pcpu is not None:
        load = "⚡ 高负载（网页轮询/保存频率将自动降频）" if pcpu > 60 else "正常"
        print(f"  本进程 CPU: {pcpu:.1f}%（{span:.0f} 秒采样）· {load}")
    if _ram_gb is not None:
        if _is_32bit():
            print(f"  系统内存: {_ram_gb:.1f} GB（32 位系统，预配上限 3.2 GB）")
        else:
            print(f"  系统内存: {_ram_gb:.1f} GB（预留系统占用后可用约 {_usable_ram:.1f} GB）")
    est = current_tokens()
    print(f"  对话占用: 约 {est:,} tokens（文本约 {est * 3 / 1024 / 1024:.1f} MB）")
    print(f"  上下文上限: {effective_trim_threshold():,} tokens（按可用内存自动收紧，接近上限自动压缩）")
    if _low_mem:
        print("  ⚡ 低内存模式: 已开启（网页历史 100 条 / 保存每 3 轮 / 连接池收紧）")
    print("-" * 40)


def run_benchmark():
    """快速基准测试：token 估算速度 + JSON 序列化速度 + 磁盘写速度。

    结果缓存进硬件信息（hw 命令显示），弱机可据此判断是否手动降级。
    """
    hw = get_hardware_info()
    bench = hw.get("bench") or {}
    if "est" in bench:
        return bench
    sample = "你好，世界 Hello world 123 " * 3000  # 约 5 万字符混合中英文
    n = 5
    t0 = time.perf_counter()
    for _ in range(n):
        estimate_tokens(sample)
    dt = max(time.perf_counter() - t0, 1e-9)
    bench["est"] = int(len(sample) * n / dt)  # 字符/秒
    payload = {"a": "中文测试", "list": list(range(1000)), "text": sample[:2000]}
    t0 = time.perf_counter()
    s = ""
    for _ in range(n):
        s = json.dumps(payload, ensure_ascii=False)
    dt = max(time.perf_counter() - t0, 1e-9)
    bench["json_mb"] = round(len(s.encode("utf-8")) * n / 1e6 / dt, 1)
    bench["disk_mb"] = _bench_disk_write()
    hw["bench"] = bench
    return bench


def _bench_disk_write():
    """测磁盘顺序写速度（MB/s）：SSD 通常 > 150，机械盘 < 80；失败返回 None"""
    import tempfile
    try:
        size = 16 * 1024 * 1024
        data = os.urandom(size)
        fd, path = tempfile.mkstemp(suffix=".chatbench")
        try:
            t0 = time.perf_counter()
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            return round(size / max(time.perf_counter() - t0, 1e-9) / 1e6, 1)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
    except Exception:
        return None


def show_hardware():
    """hw 命令：完整硬件报告 + 自适应调优结果 + 推荐本地模型"""
    hw = get_hardware_info()
    print("\n[硬件信息]")
    bits = "32 位" if _is_32bit() else "64 位"
    legacy = "（Win7/8 兼容模式）" if _windows_legacy() else ""
    print(f"  系统: {hw.get('os')} {platform.release()} · {bits}{legacy} / {hw.get('arch')} / {hw.get('cpu')}")
    print(f"  CPU 核心: {hw['cpu_cores']} 核")
    print(f"  内存: {hw.get('ram_gb') or '未知'} GB（预留系统占用后可用 {_usable_ram:.1f} GB）")
    if _low_mem:
        print("  ⚡ 低内存模式: 已开启（网页历史 100 条 / 保存每 3 轮 / 连接池收紧）")
    pyver = sys.version_info
    if os.name == "nt" and _windows_legacy() and pyver >= (3, 9):
        print("  ⚠ 注意: 当前 Python 3.9+ 不支持 Windows 7，建议安装 Python 3.8.10")
    if hw.get("gpus"):
        print(f"  GPU: {', '.join(hw['gpus'])}（后端: {hw.get('gpu_backend')}）")
        if hw.get("vram_gb"):
            print(f"  显存: {hw['vram_gb']:.1f} GB")
    else:
        print("  GPU: 未检测到独立 GPU（云端推理即可，无需本机算力）")
    print(f"  调优: 连接池 {_pool_max_conn} 连接 / 网页并发 {_web_threads} 线程 / 自动保存每 {_SAVE_EVERY} 轮")
    mode = "弱硬件模式（已自动降级：更低并发、更低保存频率，保证流畅）" if hw.get("slow_hw") else "高性能模式（多核连接池 + 后台保存 + 增量记账）"
    print(f"  加速: {mode}")
    rec = suggest_ollama_model()
    print(f"  推荐本地模型: {rec}（按显存/内存估算；provider ollama + model {rec} 即可本地推理）")
    bench = hw.get("bench")
    if bench:
        disk = bench.get("disk_mb")
        disk_label = (f"磁盘写 {disk} MB/s（{'SSD 级' if disk and disk >= 80 else '机械盘（HDD）' if disk else '未知'}）") if disk is not None else "磁盘写: 未知"
        print(f"  基准: 估算 {bench['est']:,} 字符/s · JSON {bench['json_mb']} MB/s · {disk_label}")
    print("-" * 40)


def check_latency():
    """latency 命令：测量到当前 API 服务器的连接与响应延迟"""
    err = config_error_message()
    if err:
        print(f"\n[配置缺失] {err}")
        print("-" * 40)
        return
    url = provider_base().rstrip("/") + "/models"
    print(f"\n[延迟测试] {url}")
    try:
        t0 = time.perf_counter()
        r = http_client.get(url, headers={"Authorization": f"Bearer {resolve_api_key()}"}, timeout=15)
        dt = (time.perf_counter() - t0) * 1000
        global _net_status  # 顺手更新后台探测缓存，网页状态栏立即可见
        _net_status = {"latency_ms": dt if r.status_code < 500 else None,
                       "ts": time.time(), "ok": r.status_code < 500}
        print(f"  往返耗时: {dt:.0f} ms（HTTP {r.status_code}）")
        if dt < 150:
            print("  网络质量: 优秀（流畅流式输出）")
        elif dt < 500:
            print("  网络质量: 良好")
        else:
            print("  网络质量: 一般（流式输出可能偏慢，可考虑更换更近的 API 地址）")
    except Exception as e:
        print(f"  测试失败: {e}")
    print("-" * 40)


_net_status = {"latency_ms": None, "ts": 0.0, "ok": False}  # 后台延迟探测结果
_cpu_sample = (None, 0.0)  # 进程 CPU 采样 (累计时间, 时间戳)
_net_probe_started = False


def probe_network_latency():
    """测量并缓存到当前 API 服务器的往返延迟（30 秒内复用，供网页状态栏展示）"""
    global _net_status
    now = time.time()
    if now - _net_status["ts"] < 30:
        return _net_status
    err = config_error_message()
    if err:
        _net_status = {"latency_ms": None, "ts": now, "ok": False}
        return _net_status
    url = provider_base().rstrip("/") + "/models"
    try:
        t0 = time.perf_counter()
        r = http_client.get(url, headers={"Authorization": f"Bearer {resolve_api_key()}"}, timeout=5)
        lat = (time.perf_counter() - t0) * 1000
        _net_status = {"latency_ms": lat if r.status_code < 500 else None, "ts": now, "ok": r.status_code < 500}
    except Exception:
        _net_status = {"latency_ms": None, "ts": now, "ok": False}
    return _net_status


def start_net_probe():
    """后台守护线程：定期探测 API 延迟（弱机 120s / 电池 300s / 强机 60s）"""
    global _net_probe_started
    if _net_probe_started:
        return
    _net_probe_started = True

    def _loop():
        time.sleep(4)  # 启动后稍等，避免与首轮对话抢网络
        while True:
            try:
                probe_network_latency()
            except Exception:
                pass
            interval = 300 if power_save_active() else (120 if _slow_hw else 60)
            time.sleep(interval)

    threading.Thread(target=_loop, daemon=True).start()


def _valid_import_msg(m):
    """导入消息校验：角色合法 + 内容为字符串且长度受限（防超长消息撑爆上下文）"""
    if not isinstance(m, dict) or m.get("role") not in ("user", "assistant", "system"):
        return None
    content = m.get("content")
    if not isinstance(content, str) or not content:
        return None
    if len(content) > 200_000:
        return None  # 超长消息跳过（导入侧保护，正常聊天有 MAX_MESSAGE_CHARS 限制）
    return {"role": m["role"], "content": content}


def parse_markdown_messages(text):
    """解析 Markdown 对话记录（## 你/AI/系统 分段，兼容本程序导出格式）为消息数组"""
    msgs = []
    cur_role = None
    cur_buf = []
    role_map = {"你": "user", "用户": "user", "AI": "assistant", "系统": "system"}
    def flush():
        if cur_role and cur_buf:
            content = "\n".join(cur_buf).strip()
            if content:
                msgs.append({"role": cur_role, "content": content})
    for line in (text or "").splitlines():
        m = re.match(r"^#{1,6}\s*(你|用户|AI|系统)(?:（[^）]*）)?\s*$", line.strip())
        if m:
            flush()
            cur_role = role_map[m.group(1)]
            cur_buf = []
        else:
            cur_buf.append(line)
    flush()
    return msgs


def import_history(path, merge=False):
    """从 JSON 文件导入消息到当前会话。

    merge=False 替换当前内容；merge=True 追加到现有对话末尾（合并多份导出）。
    """
    global messages, _tokens_total
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[导入失败] 无法读取文件: {e}")
        return False
    msgs = data if isinstance(data, list) else (data.get("messages") if isinstance(data, dict) else None)
    if not isinstance(msgs, list):
        print('[导入失败] 格式不正确（需要消息数组，或 {"messages": [...]} 对象）')
        return False
    valid = []
    skipped = 0
    for m in msgs:
        v = _valid_import_msg(m)
        if v is None:
            if isinstance(m, dict) and m.get("role") in ("user", "assistant", "system") \
                    and isinstance(m.get("content"), str) and len(m.get("content") or "") > 200_000:
                skipped += 1  # 超长消息单独计数提示
            continue
        valid.append(v)
    if skipped:
        print(f"[提示] {skipped} 条超长消息（>200k 字符）已跳过")
    if not valid:
        print("[导入失败] 没有有效的消息")
        return False
    if len(valid) > IMPORT_MAX_MESSAGES:
        valid = valid[-IMPORT_MAX_MESSAGES:]
        print(f"[提示] 消息过多，仅导入最近 {IMPORT_MAX_MESSAGES} 条")
    with _chat_lock:
        if merge:
            messages = messages + valid
            print(f"[已导入] 追加 {len(valid)} 条消息 → 当前会话「{current_session}」"
                  f"（共 {len(messages)} 条，已保存）")
        else:
            messages = valid
            print(f"[已导入] {len(valid)} 条消息 → 当前会话「{current_session}」（已保存）")
        _tokens_total = recount_tokens()
        persist_current_session()
    return True


_models_cache = (None, None, 0.0)  # (厂商, 模型列表, 时间戳) —— /models 代理 60 秒缓存


def fetch_model_list():
    """查询当前厂商的模型列表（60 秒缓存，按厂商隔离；失败返回 None）"""
    global _models_cache
    now = time.time()
    if _models_cache[2] and now - _models_cache[2] < 60 and _models_cache[0] == current_provider:
        return _models_cache[1]  # 换厂商后缓存自动失效，避免返回旧厂商的模型
    err = config_error_message()
    if err:
        return None
    url = provider_base().rstrip("/") + "/models"
    try:
        r = http_client.get(url, headers={"Authorization": f"Bearer {resolve_api_key()}"}, timeout=10)
        if r.status_code == 200:
            names = [str(m.get("id")) for m in r.json().get("data", [])
                     if isinstance(m, dict) and m.get("id")]
            _models_cache = (current_provider, names, now)
            return names
    except Exception:
        pass
    return None


def log_error(tag):
    """把异常写入 chat_error.log（时间戳 + 堆栈；文件超 1MB 保留最近一半）"""
    try:
        lines = f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {tag}\n{traceback.format_exc()}"
        with open("chat_error.log", "a", encoding="utf-8") as f:
            f.write(lines)
        if os.path.getsize("chat_error.log") > 1_000_000:
            with open("chat_error.log", "r", encoding="utf-8") as f:
                content = f.read()
            with open("chat_error.log", "w", encoding="utf-8") as f:
                f.write(content[-500_000:])
    except Exception:
        pass


def run_cli_turn(user_input, append_user=True):
    """执行一轮 CLI 对话（加锁 / 流式打印 / 统计）。

    append_user=False 表示复用已有的最后一条用户消息（redo 重新生成场景），
    此时出错或中断不回滚该用户消息。
    未配置 API Key 时给出引导提示，不追加消息、不退出。
    """
    global messages, _tokens_total
    err = config_error_message()
    if err:
        print(f"\n[配置缺失] {err}")
        print(f"  输入 web 打开浏览器，在『系统提示词 / 高级设置』面板填写后点『应用设置』，"
              f"填写完成两边立即可用")
        print("-" * 40)
        return False
    if _chat_lock.locked():
        print("[等待] 网页版对话正在进行中，请稍候…", flush=True)
    with _chat_lock:
        if append_user:
            # 加入用户消息（增量记账，避免每轮全量重扫历史）
            messages.append(_new_msg("user", user_input))
            add_tokens(user_input)

        printer = TableAwarePrinter()
        thinking_printed = False
        ai_header_printed = False
        assistant_content = ""
        stats = None

        try:
            # 自动裁剪（账本超限才扫描；早期对话压缩为摘要）——同样纳入 Ctrl+C 保护
            auto_trim()

            for ev in stream_reply(thinking=thinking_enabled, level=thinking_level):
                if ev["type"] == "retry":
                    print(f"\n[重试 {ev['attempt']}/{ev['max']}] 请求失败: {ev['message']}，"
                          f"{ev['wait']:.0f} 秒后重试…", flush=True)
                elif ev["type"] == "reasoning":
                    if not thinking_printed:
                        print("\n[思考链] ", end="", flush=True)
                        thinking_printed = True
                    # 终端支持颜色时用暗色显示思考链，与正式回复区分开（美观）
                    if USE_ANSI:
                        print("\033[2m" + ev["text"] + "\033[0m", end="", flush=True)
                    else:
                        print(ev["text"], end="", flush=True)
                elif ev["type"] == "content":
                    if not ai_header_printed:
                        if thinking_printed:
                            print("\n[回复] ", end="", flush=True)
                        else:
                            print("AI: ", end="", flush=True)
                        ai_header_printed = True
                    printer.feed(ev["text"])
                elif ev["type"] == "done":
                    assistant_content = ev["content"]
                    stats = ev["stats"]
            printer.finish()

        except KeyboardInterrupt:
            # 流式生成中按 Ctrl+C：取消本次回复，回滚本轮新加的用户消息与账本
            print("\n[已中断] 本次回复已取消")
            if append_user and messages and messages[-1]["role"] == "user":
                messages.pop()
                _tokens_total -= estimate_tokens(user_input)
            print("-" * 40)
            return False

        except Exception as e:
            if append_user and messages and messages[-1]["role"] == "user":
                messages.pop()
                _tokens_total -= estimate_tokens(user_input)
            print(f"\n[出错] {e}")
            return False

        if assistant_content:
            messages.append(_new_msg("assistant", assistant_content))
            add_tokens(assistant_content)

        # 后台自动保存：与用户下一轮输入并行，不阻塞
        save_history_async()

        print("\n" + "-" * 40)
        if stats:
            if stats["ttft_ms"] is not None:
                print(f"  TTFT: {stats['ttft_ms']:.0f} ms")
            print(f"  总耗时: {stats['total_sec']:.1f} 秒")
            if stats["prompt_tokens"]:
                print(f"  输入 Token: {stats['prompt_tokens']:,}")
                print(f"  输出 Token: {stats['completion_tokens']:,}")
        print("-" * 40)
        return True


def chat_loop():
    global thinking_enabled, thinking_level, messages, _tokens_total
    global current_provider, model_override, system_prompt, temperature, usage_total

    print("=" * 60)
    print(f"{provider_meta()['name']} · {current_model()} · 近似无限对话（自动裁剪 + 历史存储）")
    print("命令：exit / balance / thinking / level / provider / model / system")
    print("      temp / usage / find / session / new / redo / clear / tokens")
    print("      save / load / history / export / hw / bench / latency / web / help")
    print("提示：行尾输入 \\ 可续行；Ctrl+C 中断回复 / 退出；网页版与终端互通")
    print("=" * 60)

    # 硬件检测（说明：模型推理在云端执行，本地负责连接复用/后台保存/增量记账；有 Ollama 可本地推理）
    hw = get_hardware_info()
    mode = "低内存模式" if _low_mem else ("弱硬件模式" if hw.get("slow_hw") else "高性能模式")
    ram = hw.get("ram_gb")
    bits = "32位" if _is_32bit() else "64位"
    print(f"[硬件] {hw.get('os')} {platform.release()} · {bits} · CPU {hw['cpu_cores']} 核" +
          (f" · 内存 {ram} GB" if ram else "") +
          f" | 加速: {mode}（连接池 {_pool_max_conn} · 网页并发 {_web_threads} · 保存每 {_SAVE_EVERY} 轮）")
    if hw.get("gpus"):
        vram = f"，显存 {hw['vram_gb']:.1f} GB" if hw.get("vram_gb") else ""
        print(f"[硬件] GPU: {', '.join(hw['gpus'])}{vram}（后端: {hw.get('gpu_backend')}）")
    else:
        print("[硬件] 未检测到独立 GPU（云端推理即可，无需本机算力；本地推理可用 Ollama）")
    ollama_ok, ollama_models = detect_ollama()
    if ollama_ok:
        print(f"[硬件] 检测到本机 Ollama：{', '.join(ollama_models) or '无模型'}（推荐: {suggest_ollama_model()}；provider ollama 切换本地推理）")

    while True:
        try:
            user_input = read_multiline_input()
        except KeyboardInterrupt:
            # 在输入提示处按 Ctrl+C：保存后优雅退出
            print("\n已退出。")
            if messages:
                save_history()
            return
        except EOFError:
            # Linux/macOS 下按 Ctrl+D 触发 EOF：同样按退出处理
            print("\n已退出。")
            if messages:
                save_history()
            return

        cmd = user_input.strip().lower()
        if not cmd:
            print("[提示] 输入为空，未发送（输入 help 查看命令）")
            continue

        if cmd in ["exit", "quit", "q"]:
            # 退出前自动保存
            if messages:
                save_history()
            print("已退出。")
            break

        if cmd in ("balance", "b"):
            check_balance()
            continue

        if cmd == "thinking":
            if not provider_meta().get("thinking"):
                thinking_enabled = False
                print(f"[提示] {provider_meta()['name']} 不支持思考模式，已关闭")
            else:
                thinking_enabled = not thinking_enabled
                print(f"[思考模式已{'开启' if thinking_enabled else '关闭'}]")
            continue

        if cmd.startswith("level "):
            level = user_input.split(" ", 1)[1].strip().lower()
            if level in ["low", "medium", "high", "max"]:
                thinking_level = level
                print(f"[思考等级已设为: {level}]")
            else:
                print("[错误] 等级只能是 low / medium / high / max")
            continue

        if cmd in ("undo", "u"):
            # 撤销最后一轮：删除最后一条 AI 回复及其对应问题（弱机长会话快速清理）
            with _chat_lock:
                removed = 0
                if messages and messages[-1]["role"] == "assistant":
                    messages.pop()
                    removed += 1
                if messages and messages[-1]["role"] == "user":
                    messages.pop()
                    removed += 1
                if removed:
                    _tokens_total = recount_tokens()
            print(f"[已撤销] 删除 {removed} 条消息（当前 {len(messages)} 条）" if removed
                  else "[提示] 没有可撤销的消息")
            continue

        if cmd == "summarize":
            # 手动压缩：把除最后 2 条外的全部消息交给 API 压成一条摘要，保留近期上下文
            if len(messages) < 4:
                print("[提示] 对话太短（至少 4 条消息），暂不需要压缩")
                continue
            dropped, keep = messages[:-2], messages[-2:]
            print(f"[压缩] 正在把前 {len(dropped)} 条消息压缩为摘要…", flush=True)
            summary = summarize_conversation(dropped)
            if summary:
                applied = False
                with _chat_lock:
                    cur = list(messages)
                    if len(cur) < 4:
                        # 压缩期间网页端/其他线程修改了对话：放弃本次结果，避免覆盖
                        print("[压缩跳过] 压缩期间对话被修改，未应用")
                    else:
                        messages = [{"role": "system", "content": "【对话摘要】" + summary}] + cur[-2:]
                        _tokens_total = recount_tokens()
                        persist_current_session()
                        applied = True
                if applied:
                    print(f"[压缩完成] 早期 {len(messages) - 2} 条消息已压缩为摘要，"
                          f"上下文约 {current_tokens():,} tokens")
                else:
                    continue
            else:
                print("[压缩失败] 未能生成摘要（网络或模型问题），未做任何改动")
            continue

        if cmd.startswith("findall "):
            keyword = user_input.split(" ", 1)[1].strip()
            if not keyword:
                print("[提示] 用法: findall <关键词>")
                continue
            kw = keyword.lower()
            store = _read_store()
            hits = 0
            print(f"\n[跨会话搜索] 关键词: {keyword}")
            # 当前内存中的消息也参与搜索（可能尚未落盘）
            sessions_view = (store.get("sessions") or {}).copy()
            sessions_view[current_session] = {"messages": messages}
            for sname, sdata in sessions_view.items():
                for m in (sdata.get("messages") or []):
                    content = str(m.get("content") or "")
                    if kw in content.lower():
                        hits += 1
                        snippet = content.replace("\n", " ")[:80]
                        print(f"  「{sname}」[{m.get('role')}] {snippet}")
                        if hits >= 20:
                            break
                if hits >= 20:
                    break
            print(f"[搜索结果] 共 {hits} 条" if hits else "[未找到] 关键词: " + keyword)
            print("-" * 40)
            continue

        if cmd.startswith("import "):
            arg = user_input.split(" ", 1)[1].strip()
            if arg.lower().startswith("merge "):
                import_history(arg.split(" ", 1)[1].strip(), merge=True)
            else:
                import_history(arg)
            continue

        if cmd in ("models", "model list"):
            print("\n[模型列表] 正在查询 " + provider_meta()["name"] + " …", flush=True)
            names = fetch_model_list()
            if not names:
                print("  获取失败（配置缺失、网络错误或厂商不支持 /models 接口）")
            else:
                print(f"  共 {len(names)} 个模型：")
                for n in names:
                    mark = "→ " if n == current_model() else "  "
                    print(f"  {mark}{n}")
                print("  用法: model <名称> 切换模型")
            print("-" * 40)
            continue

        if cmd == "power":
            on_battery, percent = get_power_status()
            mode = "低功耗模式（电池供电：轮询/保存频率已降低）" if on_battery else "交流供电（正常模式）"
            pct = f"，电量 {percent}%" if percent is not None else ""
            print(f"\n[电源] {'🔋 电池供电' if on_battery else '🔌 交流供电'}{pct}")
            print(f"  模式: {mode}")
            print(f"  自动保存频率: 每 {_SAVE_EVERY} 轮（弱机长会话）")
            print("-" * 40)
            continue

        if cmd in ("health", "sysreq", "check"):
            health_check()
            continue

        if cmd in ("mem", "memory"):
            show_memory()
            continue

        if cmd in ("clear", "c"):
            with _chat_lock:
                messages = []
                _tokens_total = 0
                persist_current_session()  # 同步落盘：重启后不会“复活”旧消息
            print("[历史已清空]（已同步保存）")
            continue

        if cmd in ("tokens", "t"):
            print(f"[上下文占用] 约 {current_tokens():,} / {current_window():,} tokens"
                  f"（本机内存上限 {effective_trim_threshold():,}，接近上限自动压缩摘要）")
            continue

        if cmd == "web":
            url = f"http://{WEB_HOST}:{WEB_PORT}"
            if WEB_ENABLED:
                try:
                    import webbrowser
                    webbrowser.open(url)
                    print(f"[已打开浏览器] {url}")
                except Exception as e:
                    print(f"[无法打开浏览器] 请手动访问 {url}（{e}）")
            else:
                print(f"[网页版未启用] 请在配置中设 WEB_ENABLED = True，或手动访问 {url}")
            continue

        if cmd == "ollama":
            ok, models = detect_ollama()
            rec = suggest_ollama_model()
            if not ok:
                print("[Ollama] 未检测到本机 Ollama 服务（默认地址 http://127.0.0.1:11434）")
                print(f"  本地推理可显著利用 CPU/GPU：安装 Ollama 后本命令自动生效（建议模型: {rec}）")
            else:
                print(f"[Ollama] 服务可用，本地模型：{', '.join(models) or '（无模型，先 ollama pull 拉取）'}")
                print(f"  推荐模型: {rec}（按本机显存/内存估算）")
                if rec not in models:
                    print(f"  提示: ollama pull {rec} 拉取推荐模型后，provider ollama + model {rec} 即可对话")
                print("  用法: provider ollama 切换本地推理；model <名称> 选择模型")
            print("-" * 40)
            continue

        if cmd in ("hw", "hardware"):
            show_hardware()
            continue

        if cmd in ("bench", "benchmark"):
            print("\n[基准测试] 正在运行（约 1-2 秒）…", flush=True)
            b = run_benchmark()
            disk = b.get("disk_mb")
            disk_label = (f"{disk} MB/s（{'SSD 级' if disk and disk >= 80 else '机械盘（HDD）'}）"
                          if disk is not None else "未知")
            print(f"  token 估算速度: {b['est']:,} 字符/秒")
            print(f"  JSON 序列化: {b['json_mb']} MB/s")
            print(f"  磁盘写速度: {disk_label}")
            print("  （调优参数已按硬件自动生效：连接池 / 网页并发 / 保存频率）")
            print("-" * 40)
            continue

        if cmd in ("latency", "ping"):
            check_latency()
            continue

        if cmd == "suggest":
            hw = get_hardware_info()
            rec = suggest_ollama_model()
            basis = f"显存 {hw['vram_gb']:.1f} GB" if hw.get("vram_gb") else f"内存 {hw.get('ram_gb') or '?'} GB（无独立 GPU）"
            print(f"\n[模型推荐] 依据: {basis}")
            print(f"  推荐本地模型: {rec}")
            print(f"  使用: provider ollama 后输入 model {rec}（或网页版设置面板填写）")
            print("-" * 40)
            continue

        if cmd == "provider":
            print("\n[厂商列表]")
            for k, v in PROVIDERS.items():
                mark = "→ " if k == current_provider else "  "
                print(f"  {mark}{k:<12}{v['name']:<24}模型: {v['model']}")
            print("  用法: provider <名称> 切换厂商（输入 provider 查看列表）")
            print("-" * 40)
            continue

        if cmd.startswith("provider "):
            name = user_input.split(" ", 1)[1].strip().lower()
            if name in PROVIDERS:
                current_provider = name
                model_override = None
                runtime_settings["model"] = ""  # 切厂商后恢复该厂商默认模型
                save_runtime_config()
                _clients.clear()
                global _net_status
                _net_status = {"latency_ms": None, "ts": 0.0, "ok": False}  # 换厂商后延迟缓存作废
                print(f"[已切换] {PROVIDERS[name]['name']}（模型: {current_model()}）")
            else:
                print(f"[错误] 未知厂商: {name}（输入 provider 查看列表）")
            continue

        if cmd == "model":
            print(f"[当前模型] {current_model()}（厂商: {provider_meta()['name']}）")
            print("  用法: model <名称> 切换模型（持久化保存）")
            continue

        if cmd.startswith("model "):
            set_runtime_model(user_input.split(" ", 1)[1].strip())
            print(f"[模型已设为] {current_model()}（已保存到 {CONFIG_FILE}）")
            continue

        if cmd == "save":
            save_history()
            continue

        if cmd == "load":
            load_history()
            continue

        if cmd in ("history", "history all", "history full"):
            total = len(messages)
            show_all = user_input.strip().lower() in ("history all", "history full")
            start = 0 if show_all else max(0, total - 20)  # 默认只显示最近 20 条，避免刷屏
            print(f"\n[对话历史] 共 {total} 条消息" + ("" if show_all else f"（显示最近 {total - start} 条，输入 history all 查看全部）") + "：")
            role_names = {"user": "你", "assistant": "AI", "system": "系统"}
            for i in range(start, total):
                m = messages[i]
                role = role_names.get(m.get("role"), str(m.get("role")))
                content = m.get("content") or ""
                content_preview = content[:80]
                if len(content) > 80:
                    content_preview += "..."
                print(f"  {i+1}. [{role}] {content_preview}")
            print("-" * 40)
            continue

        if cmd == "export":
            export_history()
            continue

        if cmd.startswith("export "):
            fmt = user_input.split(" ", 1)[1].strip().lower()
            if fmt == "all":
                export_all_sessions()
            else:
                export_history("json" if fmt == "json" else "md")
            continue

        if cmd == "session":
            store = _read_store()
            sessions = sorted((store.get("sessions") or {}).keys())
            print("\n[会话列表]")
            if sessions:
                for name in sessions:
                    sdata = (store.get("sessions") or {}).get(name) or {}
                    msgs = sdata.get("messages") or []
                    n = len(msgs)
                    ts = ""
                    if msgs:
                        t = msgs[-1].get("ts") or ""
                        ts = f" · 最后 {t[5:16]}" if t else ""
                    mark = "→ " if name == current_session else "  "
                    print(f"  {mark}{name}（{n} 条{ts}）")
            else:
                print("  （无会话）")
            print("  用法: session new <名称> 新建 / session <名称> 切换 / session del <名称> 删除")
            print("-" * 40)
            continue

        if cmd.startswith("session "):
            parts = user_input.split(" ", 2)
            sub = (parts[1] or "").strip().lower()
            arg = parts[2].strip() if len(parts) > 2 else ""
            with _chat_lock:
                if sub == "new" and arg:
                    create_session(arg)
                elif sub == "del" and arg:
                    delete_session(arg)
                elif sub in ("ren", "rename") and arg:
                    rename_session(arg)
                elif sub in ("cp", "copy") and arg:
                    copy_session(arg)
                elif sub not in ("new", "del", "ren", "rename", "cp", "copy") and sub:
                    switch_session(sub)
                else:
                    print("[用法] session new <名称> | session <名称> | session del <名称> | session ren <新名称> | session cp <新名称>")
            continue

        if cmd == "prompts":
            templates = load_prompt_templates()
            print("\n[提示词模板]")
            if templates:
                for k, v in templates.items():
                    preview = str(v)[:40]
                    print(f"  · {k}: {preview}{'…' if len(str(v)) > 40 else ''}")
            else:
                print("  （无模板）")
            print("  用法: prompt save <名>（保存当前系统提示词）/ prompt apply <名> / prompt del <名>")
            print("-" * 40)
            continue

        if cmd.startswith("prompt "):
            parts = user_input.split(" ", 2)
            sub = (parts[1] or "").strip().lower()
            arg = parts[2].strip() if len(parts) > 2 else ""
            if sub == "save" and arg:
                if system_prompt:
                    print(f"[已保存模板] {arg}" if prompt_save(arg, system_prompt) else "[保存失败]")
                else:
                    print("[提示] 当前未设置系统提示词（先 system <内容> 再保存）")
            elif sub == "apply" and arg:
                print(f"[已应用模板] {arg}" if prompt_apply(arg) else f"[错误] 模板不存在: {arg}")
            elif sub == "del" and arg:
                print(f"[已删除模板] {arg}" if prompt_delete(arg) else f"[错误] 模板不存在: {arg}")
            else:
                print("[用法] prompt save <名> | prompt apply <名> | prompt del <名>")
            continue

        if cmd in ("version", "ver", "v"):
            print(f"\n[版本] DeepSeek Chat v{VERSION}")
            print(f"  功能: 多厂商对话 / 会话管理 / 思考链 / 硬件自适应加速 / 网页版")
            print(f"  环境: Python {sys.version.split()[0]} · {platform.system()} {platform.release()} · {_cpu_cores} 核")
            print("-" * 40)
            continue

        if cmd in ("help", "h", "?"):
            print(HELP_TEXT)
            continue

        if cmd == "system":
            if system_prompt:
                print(f"[系统提示词]\n{system_prompt}")
            else:
                print("[系统提示词] 未设置（用法: system <内容>；system off 清除）")
            continue

        if cmd.startswith("system "):
            val = user_input.split(" ", 1)[1].strip()
            if val.lower() in ("off", "clear", "none"):
                system_prompt = ""
                print("[系统提示词已清除]")
            else:
                system_prompt = val
                print(f"[系统提示词已设置]（{len(system_prompt)} 字，随每次请求发送）")
            continue

        if cmd == "temp":
            print(f"[温度] {temperature if temperature is not None else '厂商默认'}")
            print("  用法: temp <0-2>（如 temp 0.7；temp default 恢复默认）")
            continue

        if cmd.startswith("temp "):
            val = user_input.split(" ", 1)[1].strip().lower()
            if val in ("default", "off", "none"):
                temperature = None
                print("[温度已恢复厂商默认]")
            else:
                try:
                    t = float(val)
                    if 0 <= t <= 2:
                        temperature = t
                        print(f"[温度已设为] {t}")
                    else:
                        print("[错误] 温度范围 0-2")
                except ValueError:
                    print("[错误] 温度需为 0-2 的数字")
            continue

        if cmd == "usage":
            s = usage_summary()
            print(f"[用量统计] 回合: {s['turns']} | 输入: {s['prompt']:,} | 输出: {s['completion']:,} tokens")
            print(f"          参考成本: ${s['cost']:.4f}（按输入 ${PRICE_INPUT}/M、输出 ${PRICE_OUTPUT}/M 估算）")
            continue

        if cmd == "usage reset":
            usage_total = {"prompt": 0, "completion": 0, "turns": 0}
            print("[用量统计已重置]")
            continue

        if cmd == "usage reset":
            usage_total = {"prompt": 0, "completion": 0, "turns": 0}
            print("[用量统计已重置]")
            continue

        if cmd == "stats":
            n_user = sum(1 for m in messages if m.get("role") == "user")
            n_ai = sum(1 for m in messages if m.get("role") == "assistant")
            n_sys = sum(1 for m in messages if m.get("role") == "system")
            total_chars = sum(len(str(m.get("content") or "")) for m in messages)
            avg_ai = (total_chars / n_ai) if n_ai else 0
            print("\n[对话统计]")
            print(f"  消息: 共 {len(messages)} 条（你 {n_user} · AI {n_ai} · 系统 {n_sys}）")
            print(f"  文本: 约 {total_chars:,} 字符，AI 平均回复 {avg_ai:.0f} 字")
            print(f"  用量: 累计 {usage_total['turns']} 轮 · 输入 {usage_total['prompt']:,} · 输出 {usage_total['completion']:,} tokens")
            pct = current_tokens() * 100 // max(effective_trim_threshold(), 1)
            print(f"  上下文: {current_tokens():,} / {effective_trim_threshold():,} tokens（{pct}%）")
            print("-" * 40)
            continue


        if cmd.startswith("find "):
            keyword = user_input.split(" ", 1)[1].strip()
            if not keyword:
                print("[提示] 用法: find <关键词>")
                continue
            results = search_messages(keyword)
            if not results:
                print(f"[未找到] 关键词: {keyword}")
            else:
                print(f"\n[搜索结果] 关键词: {keyword}（{len(results)} 条）")
                for i, role, snippet in results:
                    if USE_ANSI:
                        snippet = snippet.replace(keyword, "\033[1;33m" + keyword + "\033[0m")
                    print(f"  #{i} [{role}] {snippet}")
            print("-" * 40)
            continue

        if cmd == "new":
            with _chat_lock:
                if messages:
                    save_history()
                # 旧对话自动存为独立会话，标题带上首条用户消息（自动命名，方便找回）
                first_user = next((str(m.get("content") or "").strip() for m in messages
                                   if m.get("role") == "user"), "")
                title = "".join(ch for ch in first_user if ch not in '/\\:*?"<>|')[:16]
                archive = "存档-" + time.strftime("%m%d-%H%M%S") + ("-" + title if title else "")
                with _save_lock:
                    store = _read_store()
                    store.setdefault("sessions", {})[archive] = _session_snapshot()
                    store.setdefault("sessions", {})[current_session] = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "provider": current_provider, "model": current_model(),
                        "system_prompt": system_prompt, "temperature": temperature,
                        "usage": {"prompt": 0, "completion": 0, "turns": 0},
                        "messages": [],
                    }
                    store["current"] = current_session  # 存档指针跟随当前会话
                    _write_store(store)
                messages = []
                _tokens_total = 0
                usage_total = {"prompt": 0, "completion": 0, "turns": 0}
            print(f"[已开新对话] 旧对话已存档为「{archive}」，用量统计已重置")
            continue

        if cmd == "redo":
            # 重新生成最后一条回复：先撤掉最后一条 AI 回复，再用原问题重跑一轮
            if not messages:
                print("[提示] 没有可重新生成的消息")
                continue
            popped = None
            with _chat_lock:
                if messages and messages[-1]["role"] == "assistant":
                    popped = messages.pop()
                    _tokens_total -= estimate_tokens(popped.get("content") or "")
            if messages and messages[-1]["role"] == "user":
                run_cli_turn(messages[-1]["content"], append_user=False)
            else:
                if popped is not None:
                    # 没有可重生成的问题：把刚撤下的回复放回去，避免丢失
                    messages.append(popped)
                    _tokens_total += estimate_tokens(popped.get("content") or "")
                print("[提示] 没有可重新生成的消息")
            continue

        run_cli_turn(user_input)


WEB_INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DeepSeek Chat · Beta</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' fill='%231f6feb'/%3E%3Ctext x='16' y='22' font-size='16' text-anchor='middle' fill='white' font-family='sans-serif'%3E%E2%9C%93%3C/text%3E%3C/svg%3E">
<style>
  :root {
    color-scheme: dark;
    --bg: #0f1419; --panel: #161b22; --panel2: #21262d; --border: #2d333b; --border2: #3d444d;
    --text: #e6e6e6; --text2: #8b949e; --accent: #238636; --accent-h: #2ea043;
    --blue: #1f6feb; --danger: #f85149; --warn: #d29922; --warnText: #e3b341;
    --codeBg: #0d1117; --codeText: #79c0ff; --shadow: rgba(0,0,0,.35);
    --btnGray: #30363d; --btnGrayH: #3d444d;
  }
  body[data-theme="light"] {
    color-scheme: light;
    --bg: #f6f8fa; --panel: #ffffff; --panel2: #eaeef2; --border: #d0d7de; --border2: #c4ccd4;
    --text: #1f2328; --text2: #57606a; --accent: #1a7f37; --accent-h: #2da44e;
    --blue: #0969da; --danger: #cf222e; --warn: #9a6700; --warnText: #7d4e00;
    --codeBg: #f3f4f6; --codeText: #0550ae; --shadow: rgba(0,0,0,.12);
    --btnGray: #e4e7eb; --btnGrayH: #d4d9df;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: "Segoe UI", "Microsoft YaHei", system-ui, sans-serif; background: var(--bg); color: var(--text); height: 100vh; display: flex; flex-direction: column; transition: background .2s ease, color .2s ease; }
  header { padding: 10px 16px; background: var(--panel); border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  header h1 { font-size: 16px; margin: 0; }
  .beta { background: #7c3aed; color: #fff; font-size: 11px; padding: 2px 8px; border-radius: 10px; }
  #statusBar { margin-left: auto; font-size: 12px; color: var(--text2); }
  button { background: var(--accent); color: #fff; border: none; border-radius: 10px; padding: 7px 16px; cursor: pointer; font-size: 13px; transition: background .15s ease, transform .06s ease, box-shadow .15s ease; }
  button:hover { background: var(--accent-h); box-shadow: 0 2px 10px var(--shadow); }
  button:active { transform: scale(.96); }
  button:focus-visible { outline: 2px solid var(--blue); outline-offset: 2px; }
  button:disabled { background: var(--border2); color: var(--text2); cursor: not-allowed; box-shadow: none; transform: none; }
  #clearBtn, #exportBtn, #themeBtn { background: var(--btnGray); }
  #clearBtn:hover, #exportBtn:hover, #themeBtn:hover { background: var(--btnGrayH); box-shadow: 0 2px 10px var(--shadow); }
  #log { flex: 1; overflow-y: auto; padding: 16px; scrollbar-width: thin; scrollbar-color: var(--border2) transparent; }
  #log::-webkit-scrollbar { width: 8px; }
  #log::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 4px; }
  #log::-webkit-scrollbar-thumb:hover { background: var(--border); }
  .msg { margin-bottom: 14px; max-width: 88%; animation: msgIn .25s ease; content-visibility: auto; contain-intrinsic-size: auto 120px; }
  @keyframes msgIn { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: none; } }
  .msg .who { font-size: 12px; color: var(--text2); margin-bottom: 4px; display: flex; align-items: center; gap: 6px; }
  .msg .who::before { content: ''; width: 8px; height: 8px; border-radius: 50%; background: var(--border2); flex: none; }
  .msg.user .who::before { background: var(--blue); }
  .msg:not(.user) .who::before { background: var(--accent-h); }
  .msg .body { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 10px 12px; white-space: pre-wrap; word-break: break-word; font-size: 14px; line-height: 1.6; }
  .msg.user { margin-left: auto; }
  .msg.user .body { background: color-mix(in srgb, var(--blue) 10%, transparent); border-color: color-mix(in srgb, var(--blue) 32%, transparent); }
  .msg.error .body { background: color-mix(in srgb, var(--danger) 10%, transparent); border-color: color-mix(in srgb, var(--danger) 35%, transparent); color: var(--danger); }
  .msg.info .body { background: color-mix(in srgb, var(--warn) 10%, transparent); border-color: color-mix(in srgb, var(--warn) 35%, transparent); color: var(--warnText); font-size: 12px; }
  .msg .body table { border-collapse: collapse; margin: 6px 0; }
  .msg .body th, .msg .body td { border: 1px solid var(--border2); padding: 4px 10px; font-size: 13px; }
  .msg .body th { background: var(--panel2); }
  .msg .body code { background: var(--panel2); border-radius: 4px; padding: 1px 5px; font-family: Consolas, monospace; font-size: 13px; color: var(--codeText); }
  .msg .body pre { background: var(--codeBg); border: 1px solid var(--border); border-radius: 8px; padding: 10px; overflow-x: auto; }
  .msg .body pre code { background: none; padding: 0; }
  .msg .body p { margin: 4px 0; }
  .msg .body a { color: var(--blue); text-decoration: none; }
  .msg .body a:hover { text-decoration: underline; }
  .msg .body s { opacity: .7; }
  .msg .body li.task { list-style: none; margin-left: -20px; }
  .msg .body li.task input { margin-right: 6px; }
  details { background: var(--panel2); border-radius: 8px; padding: 6px 10px; margin: 6px 0; }
  details summary { cursor: pointer; color: var(--text2); font-size: 12px; }
  #thinkText { white-space: pre-wrap; color: var(--text2); font-size: 13px; margin-top: 6px; }
  .stats { font-size: 12px; color: var(--text2); margin-top: 6px; }
  #inputBar { padding: 12px 16px; background: var(--panel); border-top: 1px solid var(--border); }
  textarea { width: 100%; background: var(--codeBg); color: var(--text); border: 1px solid var(--border); border-radius: 12px; padding: 10px; font-size: 14px; resize: none; font-family: inherit; transition: border-color .15s ease, box-shadow .15s ease; }
  textarea:focus { outline: none; border-color: var(--blue); box-shadow: 0 0 0 3px color-mix(in srgb, var(--blue) 20%, transparent); }
  @supports not (background: color-mix(in srgb, red 50%, transparent)) {
    /* 旧浏览器（Win7 时代 Chrome/Edge）不支持 color-mix：纯色回退，保证可读性 */
    .msg.user .body { background: var(--panel); border-color: var(--blue); }
    .msg.error .body { background: var(--panel); border-color: var(--danger); color: var(--danger); }
    .msg.info .body { background: var(--panel); border-color: var(--warn); color: var(--warnText); font-size: 12px; }
    textarea:focus { box-shadow: none; }
  }
  #advPanel { margin-bottom: 8px; background: var(--panel2); border-radius: 8px; padding: 6px 10px; }
  #advPanel summary { cursor: pointer; color: var(--text2); font-size: 12px; }
  #advPanel textarea { margin-top: 6px; font-size: 13px; }
  #advRow, #advRow2, #advRow3, #advRow4 { display: flex; align-items: center; gap: 10px; margin-top: 6px; font-size: 13px; color: var(--text2); flex-wrap: wrap; }
  .miniBtn { background: var(--btnGray); font-size: 11px; padding: 3px 10px; }
  .miniBtn:hover { background: var(--btnGrayH); box-shadow: none; }
  #promptSel { max-width: 180px; }
  #keyInput, #baseUrlInput, #modelInput, #tempInput, select { background: var(--codeBg); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 5px; font-size: 13px; }
  #keyInput { width: 220px; }
  #baseUrlInput { width: 240px; }
  #modelInput { width: 130px; }
  #tempInput { width: 70px; }
  #configHint { margin-top: 6px; font-size: 12px; color: var(--warnText); background: color-mix(in srgb, var(--warn) 12%, transparent); border: 1px solid color-mix(in srgb, var(--warn) 40%, transparent); border-radius: 8px; padding: 6px 10px; display: none; }
  #configHint.show { display: block; }
  #searchInput { background: var(--codeBg); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px; font-size: 12px; width: 150px; }
  #searchInput:focus { outline: none; border-color: var(--blue); }
  #searchCount { font-size: 11px; color: var(--text2); min-width: 58px; }
  mark { background: #ffd700; color: #1f2328; border-radius: 2px; padding: 0 1px; }
  #ctxBar { height: 3px; background: var(--accent); width: 0%; transition: width .4s ease, background .4s ease; }
  #ctxBar.warn { background: var(--warn); }
  #ctxBar.danger { background: var(--danger); }
  #fullOverlay { position: fixed; inset: 0; background: rgba(0,0,0,.82); z-index: 100; display: flex; align-items: center; justify-content: center; padding: 24px; }
  #fullOverlay.hidden { display: none; }
  #fullOverlay pre { max-width: 100%; max-height: 90vh; overflow: auto; background: var(--codeBg); border: 1px solid var(--border2); border-radius: 10px; padding: 16px; font-family: Consolas, monospace; font-size: 13px; color: var(--text); white-space: pre-wrap; word-break: break-word; }
  .codeHead { display: flex; align-items: center; gap: 6px; margin-bottom: 4px; }
  .codeLang { font-size: 11px; color: var(--text2); background: var(--panel2); border-radius: 4px; padding: 1px 6px; }
  .codeHead .codeCopy { margin-top: 0; }
  .msgToken { font-size: 11px; color: var(--text2); margin-top: 4px; }
  #importBtn { background: var(--btnGray); }
  #importBtn:hover { background: var(--btnGrayH); box-shadow: 0 2px 10px var(--shadow); }
  #busyTag { font-size: 12px; color: var(--warnText); }
  #busyTag.hidden { display: none; }
  #downBtn { position: fixed; right: 22px; bottom: 130px; z-index: 50; border-radius: 50%; width: 40px; height: 40px; font-size: 18px; line-height: 1; box-shadow: 0 2px 10px var(--shadow); }
  #downBtn.hidden { display: none; }
  #charCount { font-size: 11px; color: var(--text2); }
  #fontMinusBtn, #fontPlusBtn { background: var(--btnGray); font-size: 12px; padding: 3px 9px; }
  #fontMinusBtn:hover, #fontPlusBtn:hover { background: var(--btnGrayH); }
  #keyHint { margin-top: 6px; font-size: 11px; color: var(--text2); opacity: .75; }
  .msg .body.collapsed { max-height: 240px; overflow: hidden; position: relative; }
  .msg .body.collapsed::after { content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 40px; background: linear-gradient(transparent, var(--panel)); pointer-events: none; }
  @media (hover: hover) {
    .msgActions { opacity: 0; transition: opacity .15s ease; }
    .msg:hover .msgActions { opacity: 1; }
  }
  .codeWrap pre code .tk { color: #ff7b72; }
  .codeWrap pre code .st { color: #a5d6ff; }
  .codeWrap pre code .cm { color: #8b949e; }
  .codeWrap pre code .nu { color: #79c0ff; }
  body[data-theme="light"] .codeWrap pre code .tk { color: #cf222e; }
  body[data-theme="light"] .codeWrap pre code .st { color: #0a3069; }
  body[data-theme="light"] .codeWrap pre code .cm { color: #6e7781; }
  body[data-theme="light"] .codeWrap pre code .nu { color: #0550ae; }
  @media (prefers-reduced-motion: reduce) {
    .msg { animation: none; }
    * { transition: none !important; }
  }
  .msg .body h3, .msg .body h4, .msg .body h5, .msg .body h6 { margin: 8px 0 4px; }
  .msg .body h3 { font-size: 16px; } .msg .body h4 { font-size: 15px; } .msg .body h5, .msg .body h6 { font-size: 14px; }
  .msg .body ul, .msg .body ol { margin: 4px 0; padding-left: 22px; }
  .msg .body blockquote { margin: 6px 0; padding: 2px 12px; border-left: 3px solid var(--border2); color: var(--text2); }
  #opts { display: flex; align-items: center; gap: 10px; margin-top: 8px; font-size: 13px; color: var(--text2); flex-wrap: wrap; }
  .copyBtn { background: var(--btnGray); font-size: 11px; padding: 2px 8px; margin-top: 6px; }
  .copyBtn:hover { background: var(--btnGrayH); box-shadow: none; }
  .msgActions { display: flex; gap: 8px; margin-top: 6px; }
  .msgActions .copyBtn { margin-top: 0; }
  .dots { color: var(--text2); }
  .dots::after { content: ''; animation: dots 1.2s steps(4, end) infinite; }
  @keyframes dots { 0% { content: ''; } 25% { content: '·'; } 50% { content: '··'; } 75% { content: '···'; } }
  .codeWrap { position: relative; margin: 6px 0; }
  .codeWrap pre { margin: 0; }
  .codeCopy { position: absolute; top: 6px; right: 6px; background: var(--btnGray); font-size: 11px; padding: 2px 8px; }
  .codeCopy:hover { background: var(--btnGrayH); box-shadow: none; }
  @media (max-width: 640px) {
    .msg { max-width: 100%; }
    header { gap: 8px; padding: 8px 12px; }
    #inputBar { padding: 10px; }
    #keyInput, #baseUrlInput, #searchInput { width: 100%; }
  }
</style>
</head>
<body>
<header>
  <h1>DeepSeek V4 Flash 对话</h1><span class="beta">Beta</span>
  <span id="statusBar">—</span>
  <input id="searchInput" type="search" placeholder="搜索对话…" title="在已加载的消息中搜索关键词">
  <span id="searchCount"></span>
  <button id="themeBtn">☀️ 浅色</button>
  <button id="exportBtn">导出</button>
  <button id="exportJsonBtn">导出JSON</button>
  <button id="importBtn">导入</button>
  <input id="importFile" type="file" accept=".json,.md,application/json,text/markdown" style="display:none">
  <span id="busyTag" class="hidden">⚙ 对话进行中…</span>
  <button id="fontMinusBtn" class="miniBtn" title="减小字号">A−</button>
  <button id="fontPlusBtn" class="miniBtn" title="增大字号">A+</button>
  <button id="clearBtn">清空对话</button>
</header>
<div id="ctxBar"></div>
<div id="log"></div>
<div id="fullOverlay" class="hidden"><pre id="fullCode"></pre></div>
<button id="downBtn" class="hidden" title="回到底部">↓</button>
<div id="inputBar">
  <details id="advPanel" open><summary>系统提示词 / 高级设置</summary>
    <div id="advRow3">
      <label>API Key <input id="keyInput" type="password" placeholder="留空则用环境变量/配置" autocomplete="off">
      <button id="keyToggle" class="miniBtn">显示</button></label>
      <label>Base URL <input id="baseUrlInput" placeholder="留空用厂商默认"></label>
      <button id="applyBtn">应用设置</button>
    </div>
    <div id="configHint">⚠ 尚未配置 API Key：上方填写后点『应用设置』，网页版与终端立即可用（无需改代码）
      <button id="ollamaBtn" style="display:none; margin-left:8px;">改用本地 Ollama 推理</button>
    </div>
    <div id="hwText"></div>
    <div id="advRow4">
      <label>提示词模板 <select id="promptSel"></select></label>
      <button id="promptApplyBtn" class="miniBtn">应用</button>
      <button id="promptSaveBtn" class="miniBtn">存为模板</button>
      <button id="promptDelBtn" class="miniBtn">删除</button>
    </div>
    <textarea id="sysInput" rows="2" placeholder="可选：设定 AI 的角色与行为，随每次请求发送"></textarea>
    <div id="advRow">
      <label>温度 <input id="tempInput" type="number" min="0" max="2" step="0.1" placeholder="默认"></label>
    </div>
    <div id="advRow2">
      <label>会话 <select id="sessionSel"></select></label>
      <button id="renameSessionBtn" class="miniBtn">重命名</button>
      <button id="copySessionBtn" class="miniBtn">复制</button>
      <button id="deleteSessionBtn" class="miniBtn">删除</button>
      <button id="newSessionBtn">新建会话</button>
    </div>
  </details>
  <textarea id="input" rows="2" maxlength="50000" placeholder="输入消息，Enter 发送，Shift+Enter 换行"></textarea>
  <div id="opts">
    <span id="charCount" title="输入字符数/上限">0/50000</span>
    <label><input type="checkbox" id="think"> 思考模式</label>
    <select id="level">
      <option value="low">low</option>
      <option value="medium" selected>medium</option>
      <option value="high">high</option>
      <option value="max">max</option>
    </select>
    <label>厂商 <select id="providerSel"></select></label>
    <label>模型 <input id="modelInput" size="16" placeholder="模型名" list="modelList"></label>
    <datalist id="modelList"></datalist>
    <button id="sendBtn">发送</button>
  </div>
  <div id="keyHint">Enter 发送 · Shift+Enter 换行 · Esc 停止生成/关闭全屏 · 页面设置自动保存</div>
</div>
<script>
"use strict";
const log = document.getElementById('log');
const input = document.getElementById('input');
const sendBtn = document.getElementById('sendBtn');
const clearBtn = document.getElementById('clearBtn');
const statusBar = document.getElementById('statusBar');
const providerSel = document.getElementById('providerSel');
const modelInput = document.getElementById('modelInput');
const sysInput = document.getElementById('sysInput');
const tempInput = document.getElementById('tempInput');
const keyInput = document.getElementById('keyInput');
const baseUrlInput = document.getElementById('baseUrlInput');
const configHint = document.getElementById('configHint');
const sessionSel = document.getElementById('sessionSel');
const newSessionBtn = document.getElementById('newSessionBtn');
const renameSessionBtn = document.getElementById('renameSessionBtn');
const deleteSessionBtn = document.getElementById('deleteSessionBtn');
const promptSel = document.getElementById('promptSel');
const themeBtn = document.getElementById('themeBtn');
const keyToggle = document.getElementById('keyToggle');
const searchInput = document.getElementById('searchInput');
const searchCount = document.getElementById('searchCount');
let busy = false;
let abortCtl = null;
let lastMsgCount = null;
let globalMsgCount = 0;      // 服务端消息总数（编辑/删除索引换算）
let powerSave = false;       // 低功耗模式（电池供电时轮询降频）
let _modelsFor = '';         // 上次拉取模型列表的厂商（换厂商才重新拉取）
let histOffset = 0;          // 当前已加载历史窗口的起始索引（增量加载用）
const currentTexts = {};     // 消息索引 → 原文（编辑时回填）
let histLimit = WEAK ? 200 : HISTORY_LIMIT;  // 弱机页面加载即用小窗口（服务端随后按内存校正）
let curSession = '';           // 当前会话名（草稿按会话独立保存）
// 弱硬件自适应：低核数设备降低轮询频率与历史渲染量，保证页面流畅
const hc = navigator.hardwareConcurrency || 4;
const WEAK = hc <= 2;
const POLL_MS = WEAK ? 15000 : 10000;
const HISTORY_LIMIT = WEAK ? 200 : 500;

// 安全存储：localStorage 不可用（隐私模式/被禁用）时退回内存，避免整页报错
const store = (()=>{
  try{
    localStorage.setItem('__dsh_t','1'); localStorage.removeItem('__dsh_t');
    return localStorage;
  }catch(e){
    const m={};
    return { getItem:k=>m[k]??null, setItem:(k,v)=>{m[k]=String(v);}, removeItem:k=>{delete m[k];} };
  }
})();

// 主题（深色/浅色）：未手动选择时跟随系统偏好
function systemTheme(){ return window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark'; }
function applyTheme(t){
  document.body.setAttribute('data-theme', t);
  themeBtn.textContent = t==='light' ? '🌙 深色' : '☀️ 浅色';
}
let theme = store.getItem('dshTheme') || systemTheme();
applyTheme(theme);
themeBtn.addEventListener('click', ()=>{
  theme = theme==='light' ? 'dark' : 'light';
  store.setItem('dshTheme', theme);
  applyTheme(theme);
});
// API Key 显示/隐藏切换
keyToggle.addEventListener('click', ()=>{
  const showing = keyInput.type==='text';
  keyInput.type = showing ? 'password' : 'text';
  keyToggle.textContent = showing ? '显示' : '隐藏';
});
// 无 Key 且本机有 Ollama 时：一键切换本地推理
document.getElementById('ollamaBtn').addEventListener('click', async ()=>{
  try{
    const r=await fetch('/api/provider',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({provider:'ollama'})});
    if(!r.ok){ addMsg('error','切换到 Ollama 失败'); return; }
    const d=await r.json();
    addMsg('info','已切换到本地 Ollama（'+d.providerName+' · '+d.model+'），无需 Key 即可对话');
    refreshStatus();
  }catch(e){ addMsg('error','切换到 Ollama 失败: '+e); }
});

function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function renderInline(s){
  s = esc(s);
  s = s.replace(/\*\*(.+?)\*\*/g, '<b>$1</b>');
  s = s.replace(/~~(.+?)~~/g, '<s>$1</s>');
  s = s.replace(/`([^`\n]+)`/g, '<code>$1</code>');
  s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  return s;
}
function cellsOf(line){
  line=line.trim();
  if(line.startsWith('|')) line=line.slice(1);
  if(line.endsWith('|')) line=line.slice(0,-1);
  const out=[]; let cur='';
  for(let i=0;i<line.length;i++){
    const ch=line[i];
    if(ch==='\\'&&line[i+1]==='|'){ cur+='|'; i++; continue; }
    if(ch==='|'){ out.push(cur.trim()); cur=''; continue; }
    cur+=ch;
  }
  out.push(cur.trim());
  return out;
}
function isSep(cells){return cells.length>0 && cells.every(c=>/^:?-{2,}:?$/.test(c));}
function tableFrom(rows){
  let n=0;
  const parsed = rows.map(r=>{const c=cellsOf(r); n=Math.max(n,c.length); return c;});
  let h='<table>';
  parsed.forEach((c,i)=>{ if(isSep(c)) return; h+='<tr>'; c.forEach((cell,j)=>{ const tag=(i===0)?'th':'td'; h+='<'+tag+'>'+renderInline(cell)+'</'+tag+'>'; }); for(let j=c.length;j<n;j++){ h+=(i===0?'<th>':'<td>')+'</'+(i===0?'th':'td')+'>'; } h+='</tr>'; });
  return h+'</table>';
}
function renderMd(text){
  const lines = text.split('\n');
  let html='', i=0;
  while(i<lines.length){
    const t=lines[i].trim();
    if(/^\|.*\|$/.test(t)){ const rows=[]; while(i<lines.length && /^\|.*\|$/.test(lines[i].trim())){ rows.push(lines[i]); i++; } html+=tableFrom(rows); }
    else if(t.startsWith('```')){
      const lm=lines[i].match(/^```([\w+-]*)/);
      const lang=lm&&lm[1]?lm[1]:'';
      const buf=[]; i++;
      while(i<lines.length && !lines[i].trim().startsWith('```')){ buf.push(esc(lines[i])); i++; }
      i++;
      html+='<pre data-lang="'+esc(lang)+'"><code>'+buf.join('\n')+'</code></pre>';
    }
    else if(/^#{1,6}\s+/.test(t)){ const lvl=Math.min(6,t.match(/^#+/)[0].length); const htag='h'+Math.min(6,lvl+2); html+='<'+htag+'>'+renderInline(t.replace(/^#+\s*/,''))+'</'+htag+'>'; i++; }
    else if(/^>\s?/.test(t)){ const buf=[]; while(i<lines.length && /^>\s?/.test(lines[i].trim())){ buf.push(renderInline(lines[i].replace(/^>\s?/,''))); i++; } html+='<blockquote>'+buf.join('<br>')+'</blockquote>'; }
    else if(/^[-*]\s+/.test(t)){
      const buf=[];
      while(i<lines.length && /^[-*]\s+/.test(lines[i].trim())){
        const ln=lines[i].trim();
        const tm=ln.match(/^[-*]\s+\[( |x|X)\]\s+(.*)$/);
        if(tm){ buf.push('<li class="task"><input type="checkbox" disabled'+(tm[1]!==' '?' checked':'')+'>'+renderInline(tm[2])+'</li>'); }
        else { buf.push('<li>'+renderInline(ln.replace(/^[-*]\s+/,''))+'</li>'); }
        i++;
      }
      html+='<ul>'+buf.join('')+'</ul>';
    }
    else if(/^\d+\.\s+/.test(t)){ const buf=[]; while(i<lines.length && /^\d+\.\s+/.test(lines[i].trim())){ buf.push('<li>'+renderInline(lines[i].replace(/^\d+\.\s+/,''))+'</li>'); i++; } html+='<ol>'+buf.join('')+'</ol>'; }
    else if(t===''){ i++; }
    else { html+='<p>'+renderInline(lines[i])+'</p>'; i++; }
  }
  return html;
}
// markdown 渲染缓存：历史重载/流式重绘时复用结果，弱机长会话显著提速（只缓存中等长度文本）
const mdCache = new Map();
const MD_CACHE_MAX = 120;
function renderMdCached(text){
  if(mdCache.has(text)) return mdCache.get(text);
  const html = renderMd(text);
  if(text.length <= 20000){
    if(mdCache.size >= MD_CACHE_MAX) mdCache.delete(mdCache.keys().next().value);
    mdCache.set(text, html);
  }
  return html;
}
// 轻量语法高亮：字符串/注释先占位保护，再高亮关键字与数字（弱机自动关闭，长代码跳过）
const hlCache = new Map();
const HL_CACHE_MAX = 200;
function highlightCode(code){
  if(code.length > 50000) return esc(code);
  if(hlCache.has(code)) return hlCache.get(code);
  const subs = [];
  let s = esc(code);
  const BT = String.fromCharCode(96);
  const SQ = String.fromCharCode(39);
  const DQ3 = String.fromCharCode(34).repeat(3);
  const strPat = '(' + DQ3 + '[\\s\\S]*?' + DQ3 + '|"(?:[^"\\\\\\n]|\\\\.)*"|' + SQ + '(?:[^' + SQ + '\\\\\\n]|\\\\.)*' + SQ + '|' + BT + '(?:[^' + BT + '\\\\]|\\\\.)*' + BT + '|#[^\\n]*|\\/\\/[^\\n]*|\\/\\*[\\s\\S]*?\\*\\/)';
  s = s.replace(new RegExp(strPat, 'g'),
    (m)=>{ subs.push(m); return '\u0000'+(subs.length-1)+'\u0000'; });
  s = s.replace(/\b(?:def|class|import|from|return|if|elif|else|for|while|try|except|finally|with|as|lambda|pass|break|continue|global|nonlocal|yield|raise|assert|del|in|is|not|and|or|None|True|False|self|function|const|let|var|new|typeof|instanceof|async|await|export|default|static|void|do|switch|case|package|private|public|protected|interface|enum|extends|implements|throw|catch|using|namespace|struct|union|unsigned|signed|short|long|int|float|double|char|bool|string|list|dict|set|tuple|echo|print|printf|nil|true|false)\b/g, '<span class="tk">$&</span>');
  s = s.replace(/\b(\d+(?:\.\d+)?)\b/g, '<span class="nu">$1</span>');
  s = s.replace(/\u0000(\d+)\u0000/g, (m, i)=>{
    const t = subs[+i];
    const kind = (t[0]==='#'||t.indexOf('//')===0||t.indexOf('/*')===0) ? 'cm' : 'st';
    return '<span class="'+kind+'">'+t+'</span>';
  });
  return s;
}
function nowTs(){
  const d=new Date();
  return d.getHours().toString().padStart(2,'0')+':'+d.getMinutes().toString().padStart(2,'0')+':'+d.getSeconds().toString().padStart(2,'0');
}
function tokenEst(text){
  const cjk=(text.match(/[\u4e00-\u9fff]/g)||[]).length;
  const words=(text.match(/[A-Za-z0-9]+/g)||[]).length;
  return cjk + Math.round(words*1.3);
}
function fmtTs(ts){
  if(!ts) return nowTs();
  const d=new Date(String(ts).replace(' ','T'));
  if(isNaN(d.getTime())) return nowTs();
  const hm=d.getHours().toString().padStart(2,'0')+':'+d.getMinutes().toString().padStart(2,'0')+':'+d.getSeconds().toString().padStart(2,'0');
  const today=new Date();
  if(d.toDateString()===today.toDateString()) return hm;
  return (d.getMonth()+1).toString().padStart(2,'0')+'-'+d.getDate().toString().padStart(2,'0')+' '+hm;
}
function makeMsg(who, text, idx, ts){
  const div=document.createElement('div');
  div.className='msg '+who;
  const tsS=fmtTs(ts);
  if(who==='user'){
    div.innerHTML='<div class="who">你 · '+tsS+'</div><div class="body">'+esc(text)+'</div>';
    const tb=document.createElement('div'); tb.className='msgToken';
    tb.textContent='≈'+tokenEst(text).toLocaleString()+' tokens';
    div.appendChild(tb);
    if(idx!=null){
      const row=document.createElement('div'); row.className='msgActions';
      const eb=document.createElement('button'); eb.className='copyBtn'; eb.textContent='编辑';
      eb.addEventListener('click', ()=>{ editMsg(idx); });
      const db=document.createElement('button'); db.className='copyBtn'; db.textContent='删除';
      db.addEventListener('click', ()=>{ deleteMsg(idx); });
      row.appendChild(eb); row.appendChild(db); div.appendChild(row);
    }
  }
  else if(who==='ai'){ div.innerHTML='<div class="who">AI · '+tsS+'</div><div class="body"></div>'; }
  else { div.innerHTML='<div class="body">'+esc(text)+'</div>'; }
  return div;
}
function editMsg(idx){
  if(busy) return;
  const nv=prompt('编辑这条消息（将覆盖原内容）：', currentTexts[idx]||'');
  if(nv==null || !nv.trim()) return;
  fetch('/api/msg',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action:'edit', index:idx, content:nv.trim()})})
    .then(r=>r.json()).then(d=>{
      if(!d.ok){ addMsg('error', d.error||'编辑失败'); return; }
      log.innerHTML=''; loadHistory(); refreshStatus();
    }).catch(e=>addMsg('error','编辑失败: '+e));
}
function deleteMsg(idx){
  if(busy) return;
  if(!confirm('删除这条消息？')) return;
  fetch('/api/msg',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action:'delete', index:idx})})
    .then(r=>r.json()).then(d=>{
      if(!d.ok){ addMsg('error', d.error||'删除失败'); return; }
      log.innerHTML=''; loadHistory(); refreshStatus();
    }).catch(e=>addMsg('error','删除失败: '+e));
}
function addMsg(who, text){ const div=makeMsg(who,text); log.appendChild(div); scroll(); return div; }
// 安全复制：navigator.clipboard 不可用（非安全上下文/权限拒绝）时降级为 execCommand，绝不抛错
function copyText(t, btn, label){
  const restore=()=>setTimeout(()=>btn.textContent=label,1500);
  const ok=()=>{ btn.textContent='已复制'; restore(); };
  const fail=()=>{ btn.textContent='复制失败'; restore(); };
  try{
    if(navigator.clipboard && navigator.clipboard.writeText){
      navigator.clipboard.writeText(t).then(ok, fail);
    }else{
      const ta=document.createElement('textarea');
      ta.value=t; ta.style.position='fixed'; ta.style.opacity='0';
      document.body.appendChild(ta); ta.select();
      try{ document.execCommand('copy'); ok(); }catch(e){ fail(); }
      document.body.removeChild(ta);
    }
  }catch(e){ fail(); }
}
// 全局兜底：任何未捕获错误只记录到控制台，不影响页面其他功能
window.addEventListener('error', e=>{ console.error('[页面错误]', e.message); });
window.addEventListener('unhandledrejection', e=>{ console.error('[异步错误]', e.reason); });

// 给代码块加"语言标签 / 复制 / 全屏"工具条
function showFull(code){
  document.getElementById('fullCode').textContent=code;
  document.getElementById('fullOverlay').classList.remove('hidden');
}
function enhanceCode(container){
  container.querySelectorAll('pre').forEach(pre=>{
    if(pre.parentElement.classList.contains('codeWrap')) return;
    const wrap=document.createElement('div'); wrap.className='codeWrap';
    const codeEl=pre.querySelector('code');
    if(codeEl && codeEl.textContent && !WEAK){ codeEl.innerHTML=highlightCode(codeEl.textContent); }
    const head=document.createElement('div'); head.className='codeHead';
    const lang=pre.getAttribute('data-lang')||'';
    if(lang){ const lb=document.createElement('span'); lb.className='codeLang'; lb.textContent=lang; head.appendChild(lb); }
    const cp=document.createElement('button'); cp.className='codeCopy'; cp.textContent='复制';
    cp.addEventListener('click', ()=>{ copyText(pre.innerText, cp, '复制'); });
    const fsb=document.createElement('button'); fsb.className='codeCopy'; fsb.textContent='全屏';
    fsb.addEventListener('click', ()=>{ showFull(pre.innerText); });
    head.appendChild(cp); head.appendChild(fsb);
    pre.parentNode.insertBefore(wrap, pre); wrap.appendChild(head); wrap.appendChild(pre);
  });
}
document.getElementById('fullOverlay').addEventListener('click', ()=>{ document.getElementById('fullOverlay').classList.add('hidden'); });
document.addEventListener('keydown', e=>{ if(e.key==='Escape') document.getElementById('fullOverlay').classList.add('hidden'); });
document.getElementById('downBtn').addEventListener('click', ()=>{ log.scrollTop=log.scrollHeight; document.getElementById('downBtn').classList.add('hidden'); });
// 字号调节（弱机可调小字号减少渲染开销；持久化）
(function(){
  const step=1, minSize=12, maxSize=22;
  let size=parseInt(store.getItem('dshFontSize'))||14;
  const apply=()=>{ log.style.fontSize=size+'px'; store.setItem('dshFontSize', String(size)); };
  apply();
  document.getElementById('fontMinusBtn').addEventListener('click', ()=>{ size=Math.max(minSize, size-step); apply(); });
  document.getElementById('fontPlusBtn').addEventListener('click', ()=>{ size=Math.min(maxSize, size+step); apply(); });
})();
function scroll(){
  // 仅在接近底部时跟随滚动：用户向上翻阅历史时不被拽回
  const nearBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 80;
  if(nearBottom) log.scrollTop = log.scrollHeight;
  const downBtn=document.getElementById('downBtn');
  if(downBtn) downBtn.classList.toggle('hidden', nearBottom);
}
function setBusy(b){ busy=b; sendBtn.disabled=b; sendBtn.textContent=b?'停止':'发送'; }

async function refreshStatus(){
  try{
    const r=await fetch('/api/status'); const s=await r.json();
    statusBar.textContent=((!navigator.onLine)?'● 离线 · ':'')+(s.configMissing?'⚠ 未配置 Key · ':(s.apiKeySet?'✓ Key 已配置 · ':'✓ 本地推理 · '))+'会话 '+s.session+' · 厂商 '+s.providerName+' · 模型 '+s.model+' · 上下文 '+s.tokens.toLocaleString()+' / '+s.window.toLocaleString()+' tokens · '+s.messages+' 条消息'+(s.usage?' · 累计 '+s.usage.turns+' 轮':'')+(s.net&&s.net.latency_ms!=null?' · 延迟 '+Math.round(s.net.latency_ms)+'ms':'')+(s.power&&s.power.battery?' · 🔋 '+(s.power.percent!=null?s.power.percent+'%':'电池'):'')+(s.hw&&s.hw.lowMem?' · ⚡低内存':'')+(s.hw&&s.hw.procMemMb?' · 进程 '+(s.hw.procMemMb/1024).toFixed(1)+'GB':'');
    globalMsgCount=s.messages;
    powerSave=!!s.powerSave;
    if(s.hw && s.hw.historyLimit) histLimit=s.hw.historyLimit;
    if(s.session!==curSession){  // 切换会话时载入该会话的草稿
      curSession=s.session;
      input.value=store.getItem('dshDraft:'+curSession)||'';
      updateCharCount();
    }
    document.title=s.session+' · '+s.model+' · DeepSeek Chat';
    statusBar.textContent+=' · v'+s.version;
    input.placeholder='给 '+s.providerName+' 发送消息（Enter 发送，Shift+Enter 换行）';
    if(s.systemPrompt && statusBar.textContent.indexOf('系统提示词')<0 && !s._sysTagged){
      statusBar.textContent+=' · 系统提示词 ✓';
      s._sysTagged=true;
    }
    const ctxBar=document.getElementById('ctxBar');
    if(ctxBar){ const cap=s.effWindow||s.window||1; const pct=Math.min(100, s.tokens/cap*100); ctxBar.style.width=pct+'%'; ctxBar.className=pct>85?'danger':(pct>60?'warn':''); }
    const busyTag=document.getElementById('busyTag');
    if(busyTag) busyTag.classList.toggle('hidden', !(s.busy && !busy));
    configHint.classList.toggle('show', !!s.configMissing);
    const ollamaBtn = document.getElementById('ollamaBtn');
    if(ollamaBtn){ ollamaBtn.style.display = (s.configMissing && s.ollama && s.ollama.available) ? '' : 'none'; }
    const hwText = document.getElementById('hwText');
    if(hwText && s.hw){
      const h=s.hw;
      let html='<div style="margin-top:6px;font-size:12px;color:var(--text2);line-height:1.8">' +
        '硬件: CPU ' + h.cpu + ' 核' + (h.cpuName?'（'+esc(h.cpuName)+'）':'') +
        (h.ram?' · 内存 '+h.ram+' GB'+(h.usableRam?'（可用 '+h.usableRam.toFixed(1)+' GB）':''):'') +
        (h.diskFreeMb!=null?' · 磁盘 '+(h.diskFreeMb/1024).toFixed(0)+' GB':'') +
        (h.gpu?' · GPU: '+esc(h.gpu)+(h.vram?'（显存 '+h.vram.toFixed(1)+' GB）':''):' · GPU: 未检测到') +
        (h.lowMem?'<br>⚡ 低内存模式：网页历史 '+h.historyLimit+' 条 / 保存每 '+h.saveEvery+' 轮':'') +
        (h.slow?'<br>⚡ 弱硬件模式：已自动降低并发与保存频率，保证流畅':'<br>⚡ 加速已启用：多核连接池 + 后台保存 + 增量记账');
      if(s.ollama && s.ollama.available){
        html += '<br>本机 Ollama 可用 · 推荐模型: '+esc(h.suggestModel)+
          ' <button id="useLocalBtn" class="miniBtn">一键切换</button>';
      }
      html += '</div>';
      hwText.innerHTML = html;
      const ub=document.getElementById('useLocalBtn');
      if(ub) ub.addEventListener('click', async ()=>{
        try{
          const r=await fetch('/api/provider',{method:'POST',headers:{'Content-Type':'application/json'},
            body:JSON.stringify({provider:'ollama', model:h.suggestModel})});
          if(!r.ok){ addMsg('error','切换本地模型失败'); return; }
          addMsg('info','已切换到本地推理：'+h.suggestModel);
          refreshStatus();
        }catch(e){ addMsg('error','切换本地模型失败: '+e); }
      });
    }
    keyInput.value='';  // 不回显 Key（安全）；留空表示保持现有配置
    baseUrlInput.value=s.baseUrl||'';
    // 多标签页互通：其他标签页发了新消息时，自动重载对话
    if(lastMsgCount!==null && lastMsgCount!==s.messages && !busy){
      log.innerHTML=''; loadHistory();
    }
    lastMsgCount=s.messages;
    if(providerSel.options.length===0 && s.providers){
      for(const k of Object.keys(s.providers)){ const o=document.createElement('option'); o.value=k; o.textContent=s.providers[k]+'（'+k+'）'; providerSel.appendChild(o); }
    }
    providerSel.value=s.provider;
    if(document.activeElement!==modelInput) modelInput.value=s.model;   // 正在输入时不被轮询覆盖
    if(document.activeElement!==sysInput) sysInput.value=s.systemPrompt||'';
    if(document.activeElement!==tempInput) tempInput.value=(s.temperature===null||s.temperature===undefined)?'':s.temperature;
    if(s.sessions && Array.isArray(s.sessions)){
      const key=s.sessions.join(',');
      if(key!==sessionSel._key){  // 会话列表没变化时不重建下拉（避免打断选择）
        sessionSel._key=key;
        sessionSel.innerHTML='';
        const counts=s.sessionCounts||{};
        for(const n of s.sessions){
          const o=document.createElement('option'); o.value=n;
          o.textContent=(counts[n]!=null)?n+'（'+counts[n]+' 条）':n;
          sessionSel.appendChild(o);
        }
        sessionSel.value=(s.sessions.indexOf(sessionSel.value)>=0)?sessionSel.value:s.session;
      } else if(s.sessions.indexOf(sessionSel.value)<0){ sessionSel.value=s.session; }
    }
    if(s.templates && Array.isArray(s.templates)){
      const key2=s.templates.join(',');
      if(key2!==promptSel._key){
        promptSel._key=key2;
        promptSel.innerHTML='';
        for(const n of s.templates){ const o=document.createElement('option'); o.value=n; o.textContent=n; promptSel.appendChild(o); }
      }
    }
    // 模型联想：每换厂商拉取一次 /models 填充 datalist（失败静默，不影响其他功能）
    if(s.provider!==_modelsFor){
      _modelsFor=s.provider;
      fetch('/api/models').then(r=>r.json()).then(d=>{
        if(d && d.ok && Array.isArray(d.models)){
          const dl=document.getElementById('modelList'); dl.innerHTML='';
          for(const n of d.models.slice(0,200)){ const o=document.createElement('option'); o.value=n; dl.appendChild(o); }
        }
      }).catch(()=>{});
    }
  }catch(e){ statusBar.textContent='—'; }
}
// 分片渲染：每帧最多 CHUNK 条，弱机长会话不卡死主线程；返回渲染完成后的 promise
function renderMessages(msgs, baseOffset, container){
  const CHUNK=WEAK?30:40;
  const frag=document.createDocumentFragment();
  return new Promise(resolve=>{
    let i=0;
    (function next(){
      const end=Math.min(i+CHUNK, msgs.length);
      for(; i<end; i++){
        const m=msgs[i]; const c=m.content||''; const idx=baseOffset+i;
        if(m.role==='user'){ currentTexts[idx]=c; frag.appendChild(makeMsg('user', c, idx, m.ts)); }
        else if(m.role==='assistant'){
          const d=makeMsg('ai','',null,m.ts);
          const bodyEl=d.querySelector('.body');
          if(c.length>3000){
            bodyEl.classList.add('collapsed');
            bodyEl.innerHTML=renderMdCached(c); enhanceCode(d);
            const btn=document.createElement('button'); btn.className='copyBtn';
            btn.textContent='展开完整回复（'+c.length+' 字）';
            btn.addEventListener('click', ()=>{
              if(bodyEl.classList.contains('collapsed')){ bodyEl.classList.remove('collapsed'); btn.textContent='收起回复'; }
              else { bodyEl.classList.add('collapsed'); btn.textContent='展开完整回复（'+c.length+' 字）'; }
            });
            d.appendChild(btn);
          } else {
            bodyEl.innerHTML=renderMdCached(c); enhanceCode(d);
          }
          frag.appendChild(d);
        }
      }
      if(i<msgs.length){ requestAnimationFrame(next); }
      else { container.appendChild(frag); resolve(); }
    })();
  });
}
async function loadHistory(){
  try{
    const r=await fetch('/api/history?limit='+histLimit); const h=await r.json();
    const total=h.total||0;
    const offset=Math.max(0, total-histLimit);  // 只加载最近 histLimit 条
    const r2=await fetch('/api/history?limit='+(total-offset)+'&offset='+offset);
    const h2=await r2.json();
    const msgs=h2.messages||[];
    log.innerHTML='';
    Object.keys(currentTexts).forEach(k=>{ delete currentTexts[k]; });  // 清理失效索引
    histOffset=offset;
    if(offset>0){
      const bar=document.createElement('div'); bar.className='msg info'; bar.id='moreBar';
      bar.innerHTML='<div class="body">对话较长，已显示最近 '+msgs.length+' 条（共 '+total+' 条）</div>';
      const b=document.createElement('button'); b.className='copyBtn'; b.textContent='加载更早消息（还有 '+offset+' 条）';
      b.addEventListener('click', loadOlder);
      bar.appendChild(b); log.appendChild(bar);
    }
    await renderMessages(msgs, offset, log);
    globalMsgCount=total;
    scroll();
    if(searchInput && searchInput.value.trim()) doSearch();  // 重载后保持搜索过滤
  }catch(e){ console.error('[历史加载失败]', e); }
}
async function loadOlder(){
  if(histOffset<=0) return;
  const olderOffset=Math.max(0, histOffset-histLimit);
  try{
    const r=await fetch('/api/history?limit='+(histOffset-olderOffset)+'&offset='+olderOffset);
    const h=await r.json(); const msgs=h.messages||[];
    const tmp=document.createElement('div');
    const hBefore=log.scrollHeight;
    await renderMessages(msgs, olderOffset, tmp);
    const bar=document.getElementById('moreBar');
    if(bar){ log.insertBefore(tmp, bar.nextSibling); }
    else { log.insertBefore(tmp, log.firstChild); }
    log.scrollTop += (log.scrollHeight - hBefore);  // 顶部插入后补偿滚动量，避免跳动
    histOffset=olderOffset;
    const DOM_CAP = WEAK ? 400 : 600;  // DOM 节点上限：弱机长会话防止节点无限膨胀
    let extra = log.querySelectorAll('.msg').length - DOM_CAP;
    const nodes = log.querySelectorAll('.msg');
    for(let k=0; k<nodes.length && extra>0; k++){
      if(nodes[k].id==='moreBar') continue;
      nodes[k].parentNode.removeChild(nodes[k]);
      extra--;
    }
    const b=bar?bar.querySelector('button'):null;
    if(b){
      b.disabled=false;
      if(histOffset>0){ b.textContent='加载更早消息（还有 '+histOffset+' 条）'; }
      else { b.textContent='已加载全部消息'; b.disabled=true; }
    }
    if(searchInput && searchInput.value.trim()) doSearch();
  }catch(e){ console.error('[加载更早失败]', e); const b=document.getElementById('moreBar'); if(b){ const bb=b.querySelector('button'); if(bb){ bb.disabled=false; bb.textContent='加载失败，点击重试'; } } }
}
async function send(textOverride){
  const text=(textOverride!=null)?textOverride:input.value.trim();
  if(!text||busy) return;
  let autoRetried=false;  // 断线自动重试最多一次，防止重复
  setBusy(true);
  lastMsgCount=null;  // 本次发送由自己渲染，无需重载
  input.value='';
  store.setItem('dshDraft:'+curSession, '');
  store.removeItem('dshDraft');
  input.style.height='auto';
  updateCharCount();
  const myIdx=globalMsgCount;  // 新用户消息的绝对索引（编辑/删除用）
  currentTexts[myIdx]=text;
  globalMsgCount++;
  histSave(text);  // 记录到输入历史
  histIdx=-1;
  addMsg('user', text, myIdx);
  let aiDiv=addMsg('ai','');
  aiDiv.querySelector('.body').innerHTML='<span class="dots"></span>';
  let thinkBox=null;
  let aiText='';
  const streamStats=document.createElement('div'); streamStats.className='stats';
  aiDiv.appendChild(streamStats);
  const tStart=performance.now();
  // rAF 合并渲染：SSE 事件再密集也最多每帧重绘一次，长回复不卡顿
  let renderQueued=false;
  const scheduleRender=()=>{
    if(renderQueued) return;
    renderQueued=true;
    requestAnimationFrame(()=>{
      renderQueued=false;
      aiDiv.querySelector('.body').innerHTML=renderMdCached(aiText); enhanceCode(aiDiv); scroll();
      const secs=(performance.now()-tStart)/1000;
      const cps=secs>0.3 ? Math.round(aiText.length/secs) : 0;
      streamStats.textContent=aiText.length+' 字 · '+cps+' 字/秒 · '+secs.toFixed(1)+' 秒';
    });
  };
  abortCtl=new AbortController();
  let gotDone=false;
  try{
    const resp=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:text, thinking:document.getElementById('think').checked, level:document.getElementById('level').value}),
      signal:abortCtl.signal});
    if(!resp.ok||!resp.body){ addMsg('error','请求失败 HTTP '+resp.status); return; }
    const reader=resp.body.getReader();
    const dec=new TextDecoder();
    let buf='';
    while(true){
      const {done,value}=await reader.read();
      if(done) break;
      buf+=dec.decode(value,{stream:true});
      let idx;
      while((idx=buf.indexOf('\n\n'))>=0){
        const raw=buf.slice(0,idx); buf=buf.slice(idx+2);
        const line=raw.split('\n').find(l=>l.startsWith('data:'));
        if(!line) continue;
        let ev; try{ ev=JSON.parse(line.slice(5)); }catch(e){ continue; }
        if(ev.type==='content'){ aiText+=ev.text; scheduleRender(); }
        else if(ev.type==='reasoning'){
          if(!thinkBox){
            aiDiv.querySelector('.body').innerHTML='';
            const d=document.createElement('details');
            d.innerHTML='<summary>思考链</summary><div id="thinkText"></div>';
            const cpb=document.createElement('button'); cpb.className='codeCopy'; cpb.textContent='复制思考';
            cpb.addEventListener('click', ()=>{ copyText(thinkBox?thinkBox.textContent:'', cpb, '复制思考'); });
            d.querySelector('summary').appendChild(cpb);
            aiDiv.querySelector('.body').prepend(d);
            thinkBox=d.querySelector('#thinkText');
          }
          thinkBox.textContent+=ev.text;
        }
        else if(ev.type==='info'){ addMsg('info', ev.message); }
        else if(ev.type==='error'){ addMsg('error', ev.message); }
        else if(ev.type==='done'){
          gotDone=true;
          streamStats.remove();  // 流结束：换成带 TTFT/token 的最终统计行
          aiDiv.querySelector('.body').innerHTML=renderMdCached(aiText);
          enhanceCode(aiDiv);
          if(ev.stats){ const s=ev.stats; let t=''; if(s.ttft_ms!=null)t+='TTFT '+s.ttft_ms+' ms · '; if(s.prompt_tokens)t+='输入 '+s.prompt_tokens.toLocaleString()+' · 输出 '+s.completion_tokens.toLocaleString(); if(t){ const st=document.createElement('div'); st.className='stats'; st.textContent=t; aiDiv.appendChild(st); } }
          const row=document.createElement('div'); row.className='msgActions';
          const cb=document.createElement('button'); cb.className='copyBtn'; cb.textContent='复制回复';
          cb.addEventListener('click', ()=>{ copyText(aiText, cb, '复制回复'); });
          const rg=document.createElement('button'); rg.className='copyBtn'; rg.textContent='重新生成';
          rg.addEventListener('click', regenerate);
          row.appendChild(cb); row.appendChild(rg); aiDiv.appendChild(row);
        }
      }
    }
    // 流结束但没收到 done：连接中途断开，提示并允许重新生成
    if(!gotDone){
      if(aiText){ aiDiv.querySelector('.body').innerHTML=renderMd(aiText); enhanceCode(aiDiv); }
      else { aiDiv.querySelector('.body').innerHTML=''; }
      addMsg('error','连接中断，回复可能不完整');
      addRegenBtn(aiDiv);
    }
  }catch(e){
    if(e.name==='AbortError'){
      if(aiText){ aiDiv.querySelector('.body').innerHTML=renderMdCached(aiText); enhanceCode(aiDiv); }
      else { aiDiv.querySelector('.body').innerHTML=''; }
      addMsg('info','已停止生成（你的消息已保留在对话中）');
      addRegenBtn(aiDiv);
    }
    else if(!aiText && !autoRetried && !textOverride){
      // 零内容断流：自动重试一次（先查服务端状态防重复消息）
      autoRetried=true;
      addMsg('info','网络中断，正在自动重试…');
      setTimeout(()=>{ autoRetry(text); }, 1200);
    }
    else addMsg('error','连接中断: '+e);
  }
  finally{ abortCtl=null; setBusy(false); refreshStatus(); input.focus(); }
}
// 断线自动重试：先查服务端，最后一条是本次文本则转重新生成，否则直接重发
async function autoRetry(text){
  try{
    const s=await (await fetch('/api/status')).json();
    if(s.busy) return;
    const h=await (await fetch('/api/history?limit=1')).json();
    const msgs=h.messages||[];
    const last=msgs[msgs.length-1];
    if(last && last.role==='user' && last.content===text){
      addMsg('info','消息已在服务端，转为重新生成…');
      regenerate();
    } else {
      addMsg('info','重新发送中…');
      send(text);
    }
  }catch(e){ addMsg('error','自动重试失败，可点击「重新生成」'); }
}
// 在消息下方追加"重新生成"按钮（断流/停止后可用）
function addRegenBtn(aiDiv){
  const row=document.createElement('div'); row.className='msgActions';
  const rg=document.createElement('button'); rg.className='copyBtn'; rg.textContent='重新生成';
  rg.addEventListener('click', regenerate);
  row.appendChild(rg); aiDiv.appendChild(row);
}
async function regenerate(){
  if(busy) return;
  setBusy(true);
  lastMsgCount=null;
  addMsg('info','重新生成中…（点发送按钮可停止）');
  abortCtl=new AbortController();
  let errMsg=null;
  try{
    const resp=await fetch('/api/regenerate',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({thinking:document.getElementById('think').checked, level:document.getElementById('level').value}),
      signal:abortCtl.signal});
    if(!resp.ok||!resp.body){ errMsg='重新生成失败 HTTP '+resp.status; return; }
    const reader=resp.body.getReader(); const dec=new TextDecoder(); let buf='';
    while(true){
      const {done,value}=await reader.read();
      if(done) break;
      buf+=dec.decode(value,{stream:true});
      let idx;
      while((idx=buf.indexOf('\n\n'))>=0){
        const raw=buf.slice(0,idx); buf=buf.slice(idx+2);
        const line=raw.split('\n').find(l=>l.startsWith('data:'));
        if(!line) continue;
        let ev; try{ ev=JSON.parse(line.slice(5)); }catch(e){ continue; }
        if(ev.type==='error') errMsg=ev.message;
      }
    }
  }catch(e){ if(e.name!=='AbortError') errMsg='连接中断: '+e; }
  finally{
    abortCtl=null; log.innerHTML=''; loadHistory(); refreshStatus(); setBusy(false);
    if(errMsg) addMsg('error', errMsg);
  }
}
// 设置本地持久化：刷新后保留思考模式/等级/温度
(function(){
  const thinkCb=document.getElementById('think');
  const levelSel=document.getElementById('level');
  const tInp=document.getElementById('tempInput');
  if(store.getItem('dshThink')==='1') thinkCb.checked=true;
  const lv=store.getItem('dshLevel');
  if(['low','medium','high','max'].indexOf(lv)>=0) levelSel.value=lv;
  const tv=store.getItem('dshTemp');
  if(tv) tInp.value=tv;
  thinkCb.addEventListener('change', ()=>{ store.setItem('dshThink', thinkCb.checked?'1':'0'); });
  levelSel.addEventListener('change', ()=>{ store.setItem('dshLevel', levelSel.value); });
  tInp.addEventListener('change', ()=>{ store.setItem('dshTemp', tInp.value); });
})();
sendBtn.addEventListener('click', ()=>{ if(busy){ if(abortCtl) abortCtl.abort(); return; } send(); });
document.getElementById('applyBtn').addEventListener('click', async ()=>{
  try{
    const tv=tempInput.value.trim();
    const r=await fetch('/api/provider',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({provider:providerSel.value, model:modelInput.value.trim(),
        system:sysInput.value, temperature: tv===''?null:parseFloat(tv),
        apiKey:keyInput.value.trim(), baseUrl:baseUrlInput.value.trim()})});
    if(!r.ok){ addMsg('error','设置保存失败 HTTP '+r.status); return; }
    const d=await r.json();
    keyInput.value='';
    addMsg('info', d.configMissing?'设置已保存，仍需填写 API Key':'设置已保存，网页版与终端已生效');
    refreshStatus();
  }catch(e){ addMsg('error','设置保存失败: '+e); }
});
clearBtn.addEventListener('click', async ()=>{
  if(busy) return;
  if(!confirm('清空当前对话？此操作会同步删除存档中的消息，且不可恢复。')) return;
  try{
    await fetch('/api/clear',{method:'POST'});
    log.innerHTML=''; refreshStatus();
  }catch(e){ addMsg('error','清空失败: '+e); }
});
document.getElementById('exportBtn').addEventListener('click', async ()=>{
  try{
    const r=await fetch('/api/export');
    if(!r.ok){ const d=await r.json().catch(()=>null); addMsg('error', (d&&d.error)||('导出失败 HTTP '+r.status)); return; }
    const blob=await r.blob();
    const dateS=new Date().toISOString().slice(0,10);
    const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='chat_export_'+dateS+'.md'; a.click(); URL.revokeObjectURL(a.href);
  }catch(e){ addMsg('error','导出失败: '+e); }
});
document.getElementById('exportJsonBtn').addEventListener('click', async ()=>{
  try{
    const r=await fetch('/api/export?format=json');
    if(!r.ok){ const d=await r.json().catch(()=>null); addMsg('error', (d&&d.error)||('导出失败 HTTP '+r.status)); return; }
    const blob=await r.blob();
    const dateS=new Date().toISOString().slice(0,10);
    const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='chat_export_'+dateS+'.json'; a.click(); URL.revokeObjectURL(a.href);
  }catch(e){ addMsg('error','导出失败: '+e); }
});
// 导入对话（JSON）：支持本程序导出的 chat_export.json，替换当前会话内容
document.getElementById('importBtn').addEventListener('click', ()=>{ document.getElementById('importFile').click(); });
document.getElementById('importFile').addEventListener('change', async ()=>{
  const f=document.getElementById('importFile').files[0];
  document.getElementById('importFile').value='';
  if(!f) return;
  try{
    const txt=await f.text();
    let body;
    if(f.name.toLowerCase().endsWith('.md')){ body={format:'md', text:txt}; }
    else { body=JSON.parse(txt); }
    const r=await fetch('/api/import',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify(body)});
    const d=await r.json();
    if(!d.ok){ addMsg('error', d.error||'导入失败'); return; }
    addMsg('info','已导入 '+d.count+' 条消息（约 '+d.tokens.toLocaleString()+' tokens）');
    log.innerHTML=''; loadHistory(); refreshStatus();
  }catch(e){ addMsg('error','导入失败: '+e); }
});
function updateCharCount(){
  const cc=document.getElementById('charCount');
  if(cc) cc.textContent=input.value.length+'/50000';
  sendBtn.disabled = busy || input.value.trim().length===0;  // 空输入禁用发送
}
input.addEventListener('input', ()=>{ input.style.height='auto'; input.style.height=Math.min(input.scrollHeight,180)+'px'; if(input.value.length<=20000) store.setItem('dshDraft:'+curSession, input.value); updateCharCount(); });
input.value = store.getItem('dshDraft:'+curSession) || store.getItem('dshDraft') || '';  // 恢复该会话上次未发送的草稿
updateCharCount();
// IME 修复：中文输入法选词时按回车不应发送（isComposing / keyCode 229）
// 输入历史：↑/↓ 浏览最近发送的消息（localStorage 50 条，IME 选词时不触发）
const inputHist = (()=>{
  try{ const h=JSON.parse(store.getItem('dshInputHist')||'[]'); return Array.isArray(h)?h:[]; }catch(e){ return []; }
})();
let histIdx = -1, histDraft = '';
function histSave(text){
  if(!text) return;
  const i=inputHist.indexOf(text);
  if(i>=0) inputHist.splice(i,1);
  inputHist.unshift(text);
  if(inputHist.length>50) inputHist.length=50;
  store.setItem('dshInputHist', JSON.stringify(inputHist));
}
function histNav(dir){
  if(inputHist.length===0) return;
  if(histIdx<0) histDraft=input.value;
  histIdx+=dir;
  if(histIdx<0){ histIdx=-1; input.value=histDraft; }
  else if(histIdx>=inputHist.length){ histIdx=inputHist.length-1; }
  else { input.value=inputHist[histIdx]; }
  updateCharCount();
}
input.addEventListener('keydown', e=>{
  if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing&&e.keyCode!==229){ e.preventDefault(); send(); }
  if(e.key==='ArrowUp'&&!e.isComposing){ e.preventDefault(); histNav(1); }
  if(e.key==='ArrowDown'&&!e.isComposing){ e.preventDefault(); histNav(-1); }
});
document.addEventListener('keydown', e=>{
  if(e.ctrlKey&&e.key==='Enter'){ e.preventDefault(); if(!busy) send(); }
  if(e.ctrlKey&&(e.key==='k'||e.key==='K')){ e.preventDefault(); input.focus(); }
  if(e.key==='/' && document.activeElement!==input && document.activeElement!==searchInput){
    e.preventDefault(); searchInput.focus();
  }
});
document.addEventListener('keydown', e=>{ if(e.key==='Escape'&&busy&&abortCtl){ abortCtl.abort(); } });
sessionSel.addEventListener('change', async ()=>{
  if(busy) return;
  try{
    const r=await fetch('/api/session',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({action:'switch', name:sessionSel.value})});
    if(!r.ok){ addMsg('error','切换会话失败'); return; }
    log.innerHTML=''; loadHistory(); refreshStatus();
  }catch(e){ addMsg('error','切换会话失败: '+e); }
});
newSessionBtn.addEventListener('click', async ()=>{
  if(busy) return;
  const n=prompt('新会话名称（≤32 字符）');
  if(!n||!n.trim()) return;
  try{
    const r=await fetch('/api/session',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({action:'new', name:n.trim()})});
    if(!r.ok){ addMsg('info','新建会话失败'); return; }
    log.innerHTML=''; loadHistory(); refreshStatus();
  }catch(e){ addMsg('error','新建会话失败: '+e); }
});
// 提示词模板：应用 / 存为模板 / 删除
document.getElementById('promptApplyBtn').addEventListener('click', async ()=>{
  const n=promptSel.value; if(!n) return;
  try{
    const r=await fetch('/api/prompt',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({action:'apply', name:n})});
    if(!r.ok){ addMsg('error','应用模板失败'); return; }
    const d=await r.json();
    sysInput.value=d.systemPrompt||'';
    addMsg('info','已应用模板「'+n+'」到系统提示词');
    refreshStatus();
  }catch(e){ addMsg('error','应用模板失败: '+e); }
});
document.getElementById('promptSaveBtn').addEventListener('click', async ()=>{
  const text=sysInput.value.trim();
  if(!text){ addMsg('error','系统提示词为空，无法保存模板'); return; }
  const n=prompt('模板名称（≤32 字符）');
  if(!n||!n.trim()) return;
  try{
    const r=await fetch('/api/prompt',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({action:'save', name:n.trim(), text:text})});
    if(!r.ok){ addMsg('error','保存模板失败'); return; }
    addMsg('info','模板「'+n.trim()+'」已保存');
    refreshStatus();
  }catch(e){ addMsg('error','保存模板失败: '+e); }
});
document.getElementById('promptDelBtn').addEventListener('click', async ()=>{
  const n=promptSel.value; if(!n) return;
  if(!confirm('删除模板「'+n+'」？')) return;
  try{
    const r=await fetch('/api/prompt',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({action:'delete', name:n})});
    if(!r.ok){ addMsg('error','删除模板失败'); return; }
    addMsg('info','模板「'+n+'」已删除');
    refreshStatus();
  }catch(e){ addMsg('error','删除模板失败: '+e); }
});
// 会话重命名 / 删除
renameSessionBtn.addEventListener('click', async ()=>{
  if(busy) return;
  const n=prompt('新会话名称（≤32 字符）');
  if(!n||!n.trim()) return;
  try{
    const r=await fetch('/api/session',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({action:'rename', name:n.trim()})});
    if(!r.ok){ addMsg('error','重命名失败'); return; }
    log.innerHTML=''; loadHistory(); refreshStatus();
  }catch(e){ addMsg('error','重命名失败: '+e); }
});
document.getElementById('copySessionBtn').addEventListener('click', async ()=>{
  if(busy) return;
  const n=prompt('复制当前会话为新名称（≤32 字符）：', sessionSel.value+'-副本');
  if(!n||!n.trim()) return;
  try{
    const r=await fetch('/api/session',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({action:'copy', name:n.trim()})});
    if(!r.ok){ addMsg('error','复制失败（名称非法或已存在）'); return; }
    addMsg('info','已复制会话「'+sessionSel.value+'」→「'+n.trim()+'」');
    refreshStatus();
  }catch(e){ addMsg('error','复制失败: '+e); }
});
deleteSessionBtn.addEventListener('click', async ()=>{
  if(busy) return;
  if(!confirm('删除会话「'+sessionSel.value+'」？此操作不可恢复。')) return;
  try{
    const r=await fetch('/api/session',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({action:'delete', name:sessionSel.value})});
    if(!r.ok){ addMsg('error','删除失败（当前会话不可删）'); return; }
    log.innerHTML=''; loadHistory(); refreshStatus();
  }catch(e){ addMsg('error','删除失败: '+e); }
});
// 网络在线/离线提示
window.addEventListener('online', ()=>{ refreshStatus(); addMsg('info','网络已恢复'); });
window.addEventListener('offline', ()=>{ addMsg('error','网络已断开，请检查连接'); });
// 对话搜索：过滤已加载消息，命中的正文用 <mark> 高亮（不重新请求，弱机也流畅）
function clearMarks(){
  log.querySelectorAll('mark').forEach(m=>{
    const t=document.createTextNode(m.textContent);
    m.parentNode.replaceChild(t,m);
  });
}
function markIn(el,q){
  const walker=document.createTreeWalker(el,NodeFilter.SHOW_TEXT);
  const nodes=[];
  while(walker.nextNode()){
    const t=walker.currentNode;
    if(t.nodeValue && t.nodeValue.toLowerCase().indexOf(q)>=0) nodes.push(t);
  }
  nodes.forEach(t=>{
    const lower=t.nodeValue.toLowerCase();
    const frag=document.createDocumentFragment();
    let idx=0,pos;
    while((pos=lower.indexOf(q,idx))>=0){
      frag.appendChild(document.createTextNode(t.nodeValue.slice(idx,pos)));
      const m=document.createElement('mark'); m.textContent=t.nodeValue.slice(pos,pos+q.length);
      frag.appendChild(m);
      idx=pos+q.length;
    }
    frag.appendChild(document.createTextNode(t.nodeValue.slice(idx)));
    t.parentNode.replaceChild(frag,t);
  });
}
function doSearch(){
  const q=searchInput.value.trim().toLowerCase();
  clearMarks();
  if(!q){
    log.querySelectorAll('.msg').forEach(m=>{ m.style.display=''; });
    searchCount.textContent='';
    return;
  }
  let hits=0;
  log.querySelectorAll('.msg').forEach(m=>{
    const body=m.querySelector('.body');
    const ok=body && (body.textContent||'').toLowerCase().indexOf(q)>=0;
    m.style.display=ok?'':'none';
    if(ok){ hits++; markIn(m,q); }
  });
  searchCount.textContent=hits+' 条匹配';
}
let searchTimer=null;
searchInput.addEventListener('input', ()=>{
  clearTimeout(searchTimer);
  searchTimer=setTimeout(doSearch,250);  // 防抖：输入停顿后再搜
});
searchInput.addEventListener('keydown', e=>{ if(e.key==='Escape'){ searchInput.value=''; doSearch(); } });
refreshStatus();
loadHistory();
// 自适应轮询：弱机降频、电池供电再降频，终端操作网页这边同步看到
function schedulePoll(){
  const iv=powerSave ? POLL_MS+10000 : POLL_MS;
  setTimeout(()=>{
    Promise.resolve(refreshStatus()).then(schedulePoll).catch(()=>schedulePoll());
  }, iv);
}
schedulePoll();
</script>
</body>
</html>
"""


class ChatHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 按硬件能力限制并发请求线程数：多核机器开更多并发，弱机防线程爆炸
        self._req_sem = threading.BoundedSemaphore(_web_threads)

    def process_request(self, request, client_address):
        if not self._req_sem.acquire(blocking=False):
            # 并发已达上限：直接关闭连接（浏览器会自动重试），不让线程数失控
            try:
                request.close()
            except OSError:
                pass
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._req_sem.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._req_sem.release()

    def get_request(self):
        """为每个连接关闭 Nagle 算法：SSE 小包即时发出，不攒包"""
        sock, addr = super().get_request()
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        return sock, addr


class ChatHandler(BaseHTTPRequestHandler):
    server_version = "DeepSeekChat/0.1"
    protocol_version = "HTTP/1.1"  # 保持连接：状态轮询/页面资源复用 TCP，不再每次握手

    def __init__(self, *args, **kwargs):
        # 注意：BaseHTTPRequestHandler.__init__ 内部会立即调用 self.handle()，
        # 所以属性必须在 super().__init__() 之前初始化，否则请求处理时访问不到
        self._sse_buf = ""        # SSE 合并缓冲区（content 增量攒批冲刷）
        self._sse_last_flush = 0.0
        super().__init__(*args, **kwargs)

    def log_message(self, fmt, *args):
        pass  # 安静：不污染 CLI 输出

    # ---------- 工具 ----------

    def _send_body(self, body, ctype, code=200, extra=None):
        """通用响应写回：大响应自动 gzip（浏览器 Accept-Encoding: gzip 时生效）。

        弱机/慢网显著减小传输量（历史列表可达数 MB，gzip 后通常只剩 10-20%）。
        SSE 流式响应不走这里（无法预知长度）。
        """
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        if len(body) > 512 and "gzip" in (self.headers.get("Accept-Encoding") or "").lower():
            import gzip
            body = gzip.compress(body, _gz_level)  # 弱机用 2 级更快，强机 4 级更省流量
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload, code=200):
        self._send_body(json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                        "application/json; charset=utf-8", code)

    def _sse(self, payload, flush_now=False):
        """写一条 SSE 事件；客户端断开返回 False。

        加速：正文增量（content）先在缓冲区攒着，每 ≥0.1 秒或 ≥8KB 合并
        成一次 write+flush（弱机减少系统调用；浏览器端 rAF 本来就按帧合并渲染，
        感知延迟不变）。info/reasoning/error/done/retry 等事件立即冲刷。
        """
        data = json.dumps(payload, ensure_ascii=False)
        try:
            self._sse_buf += f"data: {data}\n\n"
            now = time.time()
            force = flush_now or payload.get("type") in ("reasoning", "info", "error", "done", "retry") \
                    or len(self._sse_buf) > 8192 or (now - self._sse_last_flush) >= 0.1
            if force:
                self.wfile.write(self._sse_buf.encode("utf-8"))
                self.wfile.flush()
                self._sse_buf = ""
                self._sse_last_flush = now
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            return False

    def _rollback_user_msg(self, text):
        """回滚最后一条用户消息（请求失败/中断时保持历史一致）"""
        global messages, _tokens_total
        if messages and messages[-1]["role"] == "user":
            messages.pop()
            _tokens_total -= estimate_tokens(text)

    # ---------- GET ----------

    def do_GET(self):
        """统一兜底：任何未预料异常都返回明确 500，而不是断开连接"""
        try:
            self._route_get()
        except Exception as e:
            traceback.print_exc()
            log_error("GET " + self.path)
            try:
                self._send_json({"ok": False, "error": f"服务器内部错误: {e}"}, 500)
            except Exception:
                pass

    def _route_get(self):
        if self.path in ("/", "/index.html", "/chat"):
            self._send_body(WEB_INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8",
                            extra={"Cache-Control": "no-store"})
        elif self.path == "/api/status":
            self._send_json({
                "session": current_session,
                "sessions": list_sessions(),
                "sessionCounts": {k: len((v or {}).get("messages") or [])
                                  for k, v in (_read_store().get("sessions") or {}).items()},
                "provider": current_provider,
                "providerName": provider_meta()["name"],
                "model": current_model(),
                "providers": {k: v["name"] for k, v in PROVIDERS.items()},
                "tokens": current_tokens(),
                "messages": len(messages),
                "window": current_window(),
                "thinking": thinking_enabled,
                "level": thinking_level,
                "systemPrompt": system_prompt,
                "temperature": temperature,
                "apiKeySet": bool(resolve_api_key()),
                "baseUrl": runtime_settings["base_url"],
                "configMissing": config_error_message() is not None,
                "usage": dict(usage_total),
                "templates": list(load_prompt_templates().keys()),
                "hw": {
                    "cpu": get_hardware_info()["cpu_cores"],
                    "cpuName": get_hardware_info().get("cpu") or "",
                    "arch": get_hardware_info().get("arch") or "",
                    "ram": get_hardware_info().get("ram_gb"),
                    "usableRam": _usable_ram,
                    "gpu": ", ".join(get_hardware_info().get("gpus") or []),
                    "vram": get_hardware_info().get("vram_gb"),
                    "backend": get_hardware_info().get("gpu_backend"),
                    "slow": bool(get_hardware_info().get("slow_hw")),
                    "lowMem": bool(_low_mem),
                    "bits": "32" if _is_32bit() else "64",
                    "historyLimit": web_history_limit(),
                    "procMemMb": get_process_memory_mb(),
                    "procCpu": get_process_cpu()[0],
                    "diskFreeMb": get_disk_free_mb(),
                    "suggestModel": suggest_ollama_model(),
                    "saveEvery": _SAVE_EVERY,
                },
                "ollama": dict(zip(("available", "models"), detect_ollama())),
                "net": {"latency_ms": _net_status.get("latency_ms"),
                        "ok": bool(_net_status.get("ok"))},
                "power": dict(zip(("battery", "percent"), get_power_status())),
                "powerSave": power_save_active(),
                "effWindow": effective_trim_threshold(),
                "busy": _chat_lock.locked(),
                "version": VERSION,
            })
        elif self.path == "/api/session":
            self._send_json({"current": current_session, "sessions": list_sessions()})
        elif self.path == "/api/prompt":
            self._send_json({"templates": load_prompt_templates()})
        elif self.path == "/api/models":
            try:
                names = fetch_model_list()
                if names is None:
                    self._send_json({"ok": False, "error": "无法获取模型列表（配置缺失或网络错误）"}, 400)
                else:
                    self._send_json({"ok": True, "models": names, "provider": current_provider})
            except Exception:
                self._send_json({"ok": False, "error": "模型列表获取失败"}, 500)
        elif self.path == "/api/ping":
            self._send_json({"ok": True, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        elif self.path.startswith("/api/history"):
            # 支持 ?limit=N&offset=M：网页版增量加载历史，长会话/弱机不整包传输。
            # 只带 limit 时默认取「最近 N 条」（与历史行为一致）；带 offset 才取指定窗口。
            qs = self.path.partition("?")[2]
            limit = None
            offset = None
            for part in qs.split("&"):
                if part.startswith("limit="):
                    try:
                        limit = max(1, min(10000, int(part[6:])))
                    except ValueError:
                        pass
                elif part.startswith("offset="):
                    try:
                        offset = max(0, int(part[7:]))
                    except ValueError:
                        pass
            total = len(messages)
            if offset is None:
                offset = max(0, total - limit) if limit else 0
            msgs = messages[offset:offset + limit] if limit else messages[offset:]
            self._send_json({"messages": msgs, "total": total, "offset": offset})
        elif self.path.startswith("/api/export"):
            if not messages:
                self._send_json({"ok": False, "error": "对话为空，无可导出内容"}, 400)
                return
            qs = self.path.partition("?")[2]
            date = time.strftime("%Y-%m-%d")
            if "format=json" in qs:
                body = json.dumps(messages, ensure_ascii=False, indent=2).encode("utf-8")
                self._send_body(body, "application/json; charset=utf-8",
                                extra={"Content-Disposition": f'attachment; filename="chat_export_{date}.json"'})
            else:
                self._send_body(build_export_markdown().encode("utf-8"), "text/markdown; charset=utf-8",
                                extra={"Content-Disposition": f'attachment; filename="chat_export_{date}.md"'})
        else:
            self.send_error(404)

    # ---------- POST ----------

    def do_POST(self):
        """统一兜底：请求体过大返回 413；任何未预料异常返回明确 500"""
        try:
            if int(self.headers.get("Content-Length") or 0) > MAX_BODY_BYTES:
                self.close_connection = True  # 丢弃未读请求体，避免残留字节污染下一个 keep-alive 请求
                self._send_json({"ok": False, "error": "请求体过大"}, 413)
                return
            self._route_post()
        except Exception as e:
            traceback.print_exc()
            log_error("POST " + self.path)
            try:
                self._send_json({"ok": False, "error": f"服务器内部错误: {e}"}, 500)
            except Exception:
                pass

    def _route_post(self):
        if self.path == "/api/chat":
            self._handle_chat()
        elif self.path == "/api/regenerate":
            self._handle_regenerate()
        elif self.path == "/api/provider":
            self._handle_provider()
        elif self.path == "/api/msg":
            self._handle_msg()
        elif self.path == "/api/import":
            self._handle_import()
        elif self.path == "/api/session":
            self._handle_session()
        elif self.path == "/api/prompt":
            self._handle_prompt()
        elif self.path == "/api/clear":
            if not _chat_lock.acquire(blocking=False):
                self._send_json({"ok": False, "error": "对话进行中，请稍后再试"}, 409)
                return
            try:
                global messages, _tokens_total
                messages = []
                _tokens_total = 0
                persist_current_session()  # 同步落盘：重启后不会“复活”旧消息
            finally:
                _chat_lock.release()
            self._send_json({"ok": True})
        else:
            self.send_error(404)

    def _handle_session(self):
        """网页版会话管理：{action: switch|new|delete, name}（与 CLI 互通）"""
        global current_session
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        try:
            data = json.loads(body or b"{}")
        except json.JSONDecodeError:
            self._send_json({"ok": False, "error": "请求格式错误"}, 400)
            return
        action = str(data.get("action") or "").strip()
        name = str(data.get("name") or "").strip()
        if not _chat_lock.acquire(blocking=False):
            self._send_json({"ok": False, "error": "对话进行中，请稍后再试"}, 409)
            return
        try:
            if action == "new":
                if not create_session(name):
                    self._send_json({"ok": False, "error": "非法会话名"}, 400)
                    return
            elif action == "switch":
                if not switch_session(name):
                    self._send_json({"ok": False, "error": "会话不存在"}, 404)
                    return
            elif action == "delete":
                if not delete_session(name):
                    self._send_json({"ok": False, "error": "无法删除（当前会话不可删）"}, 400)
                    return
            elif action == "rename":
                if not rename_session(name):
                    self._send_json({"ok": False, "error": "重命名失败（非法名称）"}, 400)
                    return
            elif action == "copy":
                if not copy_session(name):
                    self._send_json({"ok": False, "error": "复制失败（非法名称或已存在）"}, 400)
                    return
            else:
                self._send_json({"ok": False, "error": "未知操作"}, 400)
                return
        finally:
            _chat_lock.release()
        self._send_json({"ok": True, "current": current_session, "sessions": list_sessions()})

    def _handle_prompt(self):
        """提示词模板管理：{action: save|apply|delete, name, text?}"""
        global system_prompt
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        try:
            data = json.loads(body or b"{}")
        except json.JSONDecodeError:
            self._send_json({"ok": False, "error": "请求格式错误"}, 400)
            return
        action = str(data.get("action") or "").strip()
        name = str(data.get("name") or "").strip()
        if action == "save":
            text = str(data.get("text") or "").strip()
            if not prompt_save(name, text):
                self._send_json({"ok": False, "error": "保存失败（非法名称或内容为空）"}, 400)
                return
        elif action == "apply":
            if not prompt_apply(name):
                self._send_json({"ok": False, "error": f"模板不存在: {name}"}, 404)
                return
        elif action == "delete":
            if not prompt_delete(name):
                self._send_json({"ok": False, "error": f"模板不存在: {name}"}, 404)
                return
        else:
            self._send_json({"ok": False, "error": "未知操作"}, 400)
            return
        self._send_json({"ok": True, "templates": load_prompt_templates(),
                         "systemPrompt": system_prompt})

    def _handle_msg(self):
        """网页版消息编辑/删除：{action: edit|delete, index, content?}

        index 为消息在会话中的绝对索引（网页端按偏移量换算），与 CLI 共享同一份历史。
        """
        global messages, _tokens_total
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        try:
            data = json.loads(body or b"{}")
        except json.JSONDecodeError:
            self._send_json({"ok": False, "error": "请求格式错误"}, 400)
            return
        action = str(data.get("action") or "").strip()
        if action not in ("edit", "delete"):
            self._send_json({"ok": False, "error": "未知操作"}, 400)
            return
        try:
            index = int(data.get("index"))
        except (TypeError, ValueError):
            self._send_json({"ok": False, "error": "索引格式错误"}, 400)
            return
        if not _chat_lock.acquire(blocking=False):
            self._send_json({"ok": False, "error": "对话进行中，请稍后再试"}, 409)
            return
        try:
            if not (0 <= index < len(messages)):
                self._send_json({"ok": False, "error": "消息不存在"}, 404)
                return
            if action == "delete":
                if _is_summary_msg(messages[index]):
                    self._send_json({"ok": False, "error": "摘要消息受保护，不能删除"}, 400)
                    return
                messages.pop(index)
                _tokens_total = recount_tokens()
            elif action == "edit":
                if messages[index]["role"] != "user":
                    self._send_json({"ok": False, "error": "只能编辑用户消息"}, 400)
                    return
                content = str(data.get("content") or "").strip()
                if not content or len(content) > MAX_MESSAGE_CHARS:
                    self._send_json({"ok": False, "error": "内容为空或过长"}, 400)
                    return
                messages[index]["content"] = content
                _tokens_total = recount_tokens()
            persist_current_session()
        finally:
            _chat_lock.release()
        self._send_json({"ok": True, "count": len(messages)})

    def _handle_import(self):
        """网页版导入对话：JSON 消息数组或 {"messages": [...]}（替换当前会话）"""
        global messages, _tokens_total
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        try:
            data = json.loads(body or b"{}")
        except json.JSONDecodeError:
            self._send_json({"ok": False, "error": "请求格式错误"}, 400)
            return
        merge = bool(data.get("merge")) if isinstance(data, dict) else False
        if isinstance(data, dict) and data.get("format") == "md":
            msgs = parse_markdown_messages(str(data.get("text") or ""))
        else:
            msgs = data if isinstance(data, list) else (data.get("messages") if isinstance(data, dict) else None)
        if not isinstance(msgs, list) or not msgs:
            self._send_json({"ok": False, "error": "没有有效消息"}, 400)
            return
        valid = [v for v in (_valid_import_msg(m) for m in msgs) if v is not None]
        if not valid:
            self._send_json({"ok": False, "error": "没有有效的消息"}, 400)
            return
        if len(valid) > IMPORT_MAX_MESSAGES:
            valid = valid[-IMPORT_MAX_MESSAGES:]
        if not _chat_lock.acquire(blocking=False):
            self._send_json({"ok": False, "error": "对话进行中，请稍后再试"}, 409)
            return
        try:
            if merge:
                messages = messages + valid
            else:
                messages = valid
            _tokens_total = recount_tokens()
            persist_current_session()
        finally:
            _chat_lock.release()
        self._send_json({"ok": True, "count": len(messages), "tokens": _tokens_total, "merge": merge})

    def _handle_provider(self):
        """网页版设置接口（厂商/模型/API Key/地址/提示词/温度，与 CLI 互通并持久化）"""
        global current_provider, model_override, system_prompt, temperature
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        try:
            data = json.loads(body or b"{}")
        except json.JSONDecodeError:
            self._send_json({"ok": False, "error": "请求格式错误"}, 400)
            return
        name = str(data.get("provider") or "").strip().lower()
        model = str(data.get("model") or "").strip()
        changed_client = False
        if name and name in PROVIDERS:
            if name != current_provider:
                changed_client = True
            current_provider = name
            model_override = model or None
        elif name:
            self._send_json({"ok": False, "error": f"未知厂商: {name}"}, 400)
            return
        else:
            model_override = model or None
        if "model" in data:
            runtime_settings["model"] = model  # 模型随请求参数生效，无需重建客户端
        if isinstance(data.get("apiKey"), str) and data["apiKey"].strip():
            new_key = data["apiKey"].strip()
            if len(new_key) > 512:
                self._send_json({"ok": False, "error": "API Key 过长"}, 400)
                return
            if new_key != runtime_settings.setdefault("api_key", {}).get(current_provider, ""):
                runtime_settings["api_key"][current_provider] = new_key
                changed_client = True
        if isinstance(data.get("baseUrl"), str):
            new_base = data["baseUrl"].strip()
            if len(new_base) > 512:
                self._send_json({"ok": False, "error": "Base URL 过长"}, 400)
                return
            if new_base != runtime_settings["base_url"]:
                runtime_settings["base_url"] = new_base
                changed_client = True
        if isinstance(data.get("system"), str):
            system_prompt = data["system"]
        if data.get("temperature") is not None:
            try:
                t = float(data["temperature"])
                temperature = t if 0 <= t <= 2 else None
            except (TypeError, ValueError):
                temperature = None
        if changed_client:
            _clients.clear()  # Key/地址/厂商变化后重建客户端
            save_runtime_config()
            global _net_status
            _net_status = {"latency_ms": None, "ts": 0.0, "ok": False}  # 延迟缓存作废，重新探测
        self._send_json({
            "ok": True,
            "provider": current_provider,
            "providerName": provider_meta()["name"],
            "model": current_model(),
            "systemPrompt": system_prompt,
            "temperature": temperature,
            "apiKeySet": bool(resolve_api_key()),
            "configMissing": config_error_message() is not None,
        })

    def _start_sse(self):
        """SSE 响应头（无 Content-Length：显式关闭连接，客户端才能判断流结束）"""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()

    @staticmethod
    def _as_bool(v, default=False):
        """宽容的布尔解析：兼容 JSON true / 字符串 "true" / 1 等"""
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return v != 0
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes", "on")
        return default

    def _handle_chat(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        try:
            data = json.loads(body or b"{}")
        except json.JSONDecodeError:
            self._send_json({"ok": False, "error": "请求格式错误"}, 400)
            return
        text = str(data.get("message") or "").strip()
        if not text:
            self._send_json({"ok": False, "error": "消息为空"}, 400)
            return
        if len(text) > MAX_MESSAGE_CHARS:
            self._send_json({"ok": False, "error": f"消息过长（{len(text):,} 字符，上限 {MAX_MESSAGE_CHARS:,}）"}, 400)
            return
        thinking = self._as_bool(data.get("thinking"), thinking_enabled)
        level = str(data.get("level") or thinking_level)
        if level not in ("low", "medium", "high", "max"):
            level = thinking_level

        # 先回 SSE 响应头，再检查配置与回合锁
        self._start_sse()

        err = config_error_message()
        if err:
            self._sse({"type": "error", "message": err + "（本页『系统提示词/高级设置』面板填写后即可对话）"})
            return

        if not _chat_lock.acquire(blocking=False):
            self._sse({"type": "error", "message": "另一个对话正在进行中，请稍候"})
            return
        try:
            self._run_turn_locked(text, thinking, level, append_user=True)
        finally:
            _chat_lock.release()

    def _handle_regenerate(self):
        """重新生成最后一条回复（复用最后一条用户消息，不重复追加）"""
        global messages, _tokens_total
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        try:
            data = json.loads(body or b"{}")
        except json.JSONDecodeError:
            data = {}
        thinking = self._as_bool(data.get("thinking"), thinking_enabled)
        level = str(data.get("level") or thinking_level)
        if level not in ("low", "medium", "high", "max"):
            level = thinking_level

        self._start_sse()

        err = config_error_message()
        if err:
            self._sse({"type": "error", "message": err + "（本页设置面板填写后即可对话）"})
            return

        if not _chat_lock.acquire(blocking=False):
            self._sse({"type": "error", "message": "另一个对话正在进行中，请稍候"})
            return
        try:
            popped = None
            if messages and messages[-1]["role"] == "assistant":
                popped = messages.pop()
                _tokens_total -= estimate_tokens(popped.get("content") or "")
            if not messages or messages[-1]["role"] != "user":
                if popped is not None:
                    # 没有可重生成的问题：把刚撤下的回复放回去，避免丢失
                    messages.append(popped)
                    _tokens_total += estimate_tokens(popped.get("content") or "")
                self._sse({"type": "error", "message": "没有可重新生成的用户消息"})
                return
            self._run_turn_locked(messages[-1]["content"], thinking, level, append_user=False)
        finally:
            _chat_lock.release()

    def _run_turn_locked(self, text, thinking, level, append_user):
        """持锁状态下执行一轮对话并推送 SSE 事件；
        append_user=False 表示复用已有的最后一条用户消息（重新生成场景）"""
        global messages, _tokens_total
        if append_user:
            messages.append(_new_msg("user", text))
            add_tokens(text)
        if current_tokens() > effective_trim_threshold() * 0.85:
            self._sse({"type": "info",
                       "message": f"上下文已用 {current_tokens():,}/{effective_trim_threshold():,} tokens，"
                                  f"正在自动压缩早期对话…"})
        auto_trim()

        assistant_content = ""
        stats = None
        disconnected = False
        try:
            for ev in stream_reply(thinking=thinking, level=level):
                if ev["type"] == "retry":
                    if not self._sse({"type": "info",
                                     "message": f"请求失败，{ev['wait']:.0f} 秒后重试（{ev['attempt']}/{ev['max']}）"}):
                        disconnected = True
                        break
                elif ev["type"] in ("reasoning", "content"):
                    if not self._sse({"type": ev["type"], "text": ev["text"]}):
                        disconnected = True
                        break
                elif ev["type"] == "done":
                    assistant_content = ev["content"]
                    stats = ev["stats"]
        except Exception as e:
            if append_user:
                self._rollback_user_msg(text)
            self._sse({"type": "error", "message": f"请求失败: {e}"})
            return

        if disconnected:
            # 客户端主动停止/断开：保留已发送的用户消息（对话历史里留着问题），
            # 只是不追加回复——与错误回滚语义区分开；强制落盘防止重启丢失
            save_history_async(force=True)
            return

        if assistant_content:
            messages.append(_new_msg("assistant", assistant_content))
            add_tokens(assistant_content)
        save_history_async(force=True)  # 网页回合必存：浏览器随时可能关闭，不依赖降频节流
        self._sse({"type": "done", "stats": stats})


def start_web_server():
    """启动网页版（Beta）服务器（守护线程，随主程序退出）"""
    try:
        server = ChatHTTPServer((WEB_HOST, WEB_PORT), ChatHandler)
    except OSError as e:
        print(f"[网页版 Beta] 启动失败（{WEB_HOST}:{WEB_PORT}）: {e}")
        return None
    threading.Thread(target=server.serve_forever, daemon=True).start()
    start_net_probe()  # 后台网络延迟监测（网页状态栏显示）
    print(f"[网页版 Beta] 已启动: http://{WEB_HOST}:{WEB_PORT}（与终端共享同一份对话历史）")
    return server


if __name__ == "__main__":
    ensure_utf8_stdio()
    # 命令行参数：--web-only（只跑网页版）/ --no-web（关网页版）/ --port N / --host H / --version
    if "--version" in sys.argv or "-V" in sys.argv:
        print(f"DeepSeek Chat v{VERSION}（Python {sys.version.split()[0]} · {platform.system()} {platform.release()} · {_cpu_cores} 核）")
        sys.exit(0)
    cli_web_only = "--web-only" in sys.argv
    cli_no_web = "--no-web" in sys.argv
    for _i, _a in enumerate(sys.argv):
        if _a == "--port" and _i + 1 < len(sys.argv):
            try:
                WEB_PORT = int(sys.argv[_i + 1])
            except ValueError:
                pass
        elif _a == "--host" and _i + 1 < len(sys.argv):
            WEB_HOST = sys.argv[_i + 1].strip() or WEB_HOST
    if sys.version_info < (3, 8):
        print("=" * 60)
        print("[兼容性] ⚠ 当前 Python 版本过旧，部分功能可能不可用。")
        print("  Windows 7 请安装 Python 3.8.10（最后一个支持 Win7 的版本）")
        print("  其他系统请安装 Python 3.8 或更高版本")
        print("=" * 60)
    if os.name == "nt" and _windows_legacy():
        pyver = sys.version_info
        if pyver >= (3, 9):
            print("[兼容性] ⚠ 检测到 Windows 7/8，但 Python 3.9+ 已不再支持该系统，"
                  "建议安装 Python 3.8.10 以获得完整兼容")
        else:
            print("[兼容性] Windows 7/8 兼容模式：已自动关闭 ANSI 彩色输出与不兼容特性")
    if _low_mem:
        print(f"[内存] ⚡ 低内存模式（可用 {_usable_ram:.1f} GB）：上下文上限 "
              f"{effective_trim_threshold() // 1000}k tokens、网页历史 {web_history_limit()} 条、保存每 {_SAVE_EVERY} 轮")
    if load_runtime_config():
        print(f"[配置] 已加载网页版保存的运行期配置（{CONFIG_FILE}）")
    enable_ansi_on_windows()
    get_hardware_info()  # 提前探测硬件（避免网页首个请求卡在探测命令上）
    # 先加载存档再启动网页服务：避免网页请求在加载完成前到达（启动竞态）
    if os.path.exists(HISTORY_FILE):
        load_history()
        print("-" * 40)
    if WEB_ENABLED and not cli_no_web:
        start_web_server()
    if cli_web_only:
        print(f"[网页版] 仅运行网页版模式（{WEB_HOST}:{WEB_PORT}），Ctrl+C 退出")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            sys.exit(0)
    if not provider_meta().get("local") and not resolve_api_key():
        url = f"http://{WEB_HOST}:{WEB_PORT}" if WEB_ENABLED else "（网页版未启用）"
        ollama_ok, ollama_models = detect_ollama()
        print("=" * 60)
        print("[提示] 尚未配置 API Key / 模型 / 地址（可全部在网页版填写，无需改代码）。")
        print(f"  1) 打开网页版: {url}（命令行输入 web 可直接打开浏览器）")
        print("  2) 在『系统提示词 / 高级设置』面板填写 API Key（可选：自定义地址、模型）")
        print("  3) 点『应用设置』确认后，网页版与终端命令行即可同时对话，无需重启")
        if ollama_ok:
            print(f"  💡 检测到本机 Ollama（模型: {', '.join(ollama_models) or '无'}）——"
                  f"输入 provider ollama 即可用本地 GPU/CPU 推理，无需任何 Key")
        print("  另：也可设置环境变量或编辑 chat_config.json 后重启")
        print("=" * 60)
    warm_up_connection()
    start_net_probe()
    chat_loop()
    try:
        input("\n程序结束，按回车关闭窗口...")
    except (KeyboardInterrupt, EOFError):
        pass
