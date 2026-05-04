import os
import json
import base64
import hashlib
import threading
import tkinter as tk
from tkinter import messagebox, filedialog
from datetime import datetime
from typing import List, Dict, Optional

import customtkinter as ctk
from cryptography.fernet import Fernet
import bcrypt
from openai import OpenAI
import pyperclip

# ------------------------------
# 配置
# ------------------------------
ctk.set_appearance_mode("System")  # 跟随系统，可手动切换
ctk.set_default_color_theme("blue")

DATA_DIR = "user_data"
os.makedirs(DATA_DIR, exist_ok=True)
USERS_FILE = os.path.join(DATA_DIR, "users.json")
CHATS_DIR = os.path.join(DATA_DIR, "chats")
os.makedirs(CHATS_DIR, exist_ok=True)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# ------------------------------
# 加密工具
# ------------------------------
def derive_key_from_password(password: str, salt: bytes = None) -> tuple[bytes, bytes]:
    """从密码派生 Fernet 密钥（32字节 base64 编码）"""
    if salt is None:
        salt = os.urandom(16)
    # 使用 PBKDF2
    kdf = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000, dklen=32)
    key = base64.urlsafe_b64encode(kdf)
    return key, salt

def encrypt_data(data: str, password: str, salt: bytes) -> str:
    key, _ = derive_key_from_password(password, salt)
    f = Fernet(key)
    return f.encrypt(data.encode()).decode()

def decrypt_data(encrypted: str, password: str, salt: bytes) -> str:
    key, _ = derive_key_from_password(password, salt)
    f = Fernet(key)
    return f.decrypt(encrypted.encode()).decode()

# ------------------------------
# 用户管理
# ------------------------------
class UserManager:
    def __init__(self):
        self.users = {}
        self.load_users()

    def load_users(self):
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r', encoding='utf-8') as f:
                self.users = json.load(f)
        else:
            self.users = {}

    def save_users(self):
        with open(USERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.users, f, indent=2)

    def register(self, username: str, password: str) -> bool:
        if username in self.users:
            return False
        # 哈希密码
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        # 生成加密盐
        salt = os.urandom(16)
        self.users[username] = {
            "password_hash": hashed,
            "encryption_salt": base64.b64encode(salt).decode(),
            "api_key_encrypted": None
        }
        self.save_users()
        return True

    def login(self, username: str, password: str) -> bool:
        if username not in self.users:
            return False
        stored = self.users[username]
        return bcrypt.checkpw(password.encode(), stored["password_hash"].encode())

    def get_user_salt(self, username: str) -> bytes:
        return base64.b64decode(self.users[username]["encryption_salt"])

    def save_api_key(self, username: str, api_key: str, password: str):
        salt = self.get_user_salt(username)
        encrypted = encrypt_data(api_key, password, salt)
        self.users[username]["api_key_encrypted"] = encrypted
        self.save_users()

    def get_api_key(self, username: str, password: str) -> Optional[str]:
        encrypted = self.users[username].get("api_key_encrypted")
        if not encrypted:
            return None
        salt = self.get_user_salt(username)
        try:
            return decrypt_data(encrypted, password, salt)
        except Exception:
            return None

# ------------------------------
# 聊天记录管理
# ------------------------------
class ChatManager:
    def __init__(self, username: str):
        self.username = username
        self.user_chat_dir = os.path.join(CHATS_DIR, username)
        os.makedirs(self.user_chat_dir, exist_ok=True)

    def list_chats(self) -> List[Dict]:
        chats = []
        for fname in os.listdir(self.user_chat_dir):
            if fname.endswith('.json'):
                path = os.path.join(self.user_chat_dir, fname)
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                chats.append({
                    "id": data.get("id", fname[:-5]),
                    "name": data.get("name", "未命名对话"),
                    "updated": data.get("updated", "")
                })
        # 按更新时间倒序
        chats.sort(key=lambda x: x["updated"], reverse=True)
        return chats

    def create_chat(self, chat_id: str = None) -> str:
        if chat_id is None:
            chat_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        path = os.path.join(self.user_chat_dir, f"{chat_id}.json")
        data = {
            "id": chat_id,
            "name": "新对话",
            "created": datetime.now().isoformat(),
            "updated": datetime.now().isoformat(),
            "messages": []
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return chat_id

    def save_chat(self, chat_id: str, messages: List[Dict], name: str = None):
        path = os.path.join(self.user_chat_dir, f"{chat_id}.json")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {"id": chat_id, "created": datetime.now().isoformat()}
        data["messages"] = messages
        data["updated"] = datetime.now().isoformat()
        if name:
            data["name"] = name
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_chat(self, chat_id: str) -> Dict:
        path = os.path.join(self.user_chat_dir, f"{chat_id}.json")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def delete_chat(self, chat_id: str):
        path = os.path.join(self.user_chat_dir, f"{chat_id}.json")
        if os.path.exists(path):
            os.remove(path)

    def rename_chat(self, chat_id: str, new_name: str):
        data = self.load_chat(chat_id)
        data["name"] = new_name
        self.save_chat(chat_id, data["messages"], name=new_name)

# ------------------------------
# 登录/注册窗口
# ------------------------------
class LoginWindow(ctk.CTkToplevel):
    def __init__(self, parent, user_manager: UserManager, on_success):
        super().__init__(parent)
        self.user_manager = user_manager
        self.on_success = on_success
        self.title("登录 / 注册")
        self.geometry("400x350")
        self.resizable(False, False)

        self.label_title = ctk.CTkLabel(self, text="DeepSeek 助手", font=ctk.CTkFont(size=20, weight="bold"))
        self.label_title.pack(pady=20)

        self.tabview = ctk.CTkTabview(self, width=350)
        self.tabview.pack(pady=10)

        self.tab_login = self.tabview.add("登录")
        self.tab_register = self.tabview.add("注册")

        # 登录页
        self.entry_username_login = ctk.CTkEntry(self.tab_login, placeholder_text="用户名")
        self.entry_username_login.pack(pady=10, padx=20, fill="x")
        self.entry_password_login = ctk.CTkEntry(self.tab_login, placeholder_text="密码", show="*")
        self.entry_password_login.pack(pady=10, padx=20, fill="x")
        self.btn_login = ctk.CTkButton(self.tab_login, text="登录", command=self.do_login)
        self.btn_login.pack(pady=10)

        # 注册页
        self.entry_username_reg = ctk.CTkEntry(self.tab_register, placeholder_text="用户名")
        self.entry_username_reg.pack(pady=10, padx=20, fill="x")
        self.entry_password_reg = ctk.CTkEntry(self.tab_register, placeholder_text="密码", show="*")
        self.entry_password_reg.pack(pady=10, padx=20, fill="x")
        self.entry_confirm_reg = ctk.CTkEntry(self.tab_register, placeholder_text="确认密码", show="*")
        self.entry_confirm_reg.pack(pady=10, padx=20, fill="x")
        self.btn_register = ctk.CTkButton(self.tab_register, text="注册", command=self.do_register)
        self.btn_register.pack(pady=10)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def do_login(self):
        username = self.entry_username_login.get().strip()
        password = self.entry_password_login.get()
        if not username or not password:
            messagebox.showerror("错误", "请输入用户名和密码")
            return
        if self.user_manager.login(username, password):
            self.on_success(username, password)
            self.destroy()
        else:
            messagebox.showerror("错误", "用户名或密码错误")

    def do_register(self):
        username = self.entry_username_reg.get().strip()
        password = self.entry_password_reg.get()
        confirm = self.entry_confirm_reg.get()
        if not username or not password:
            messagebox.showerror("错误", "请输入用户名和密码")
            return
        if password != confirm:
            messagebox.showerror("错误", "两次密码不一致")
            return
        if self.user_manager.register(username, password):
            messagebox.showinfo("成功", "注册成功，请登录")
            self.tabview.set("登录")
        else:
            messagebox.showerror("错误", "用户名已存在")

    def on_close(self):
        self.destroy()
        self.master.quit()

# ------------------------------
# API 密钥设置窗口
# ------------------------------
class APIKeyWindow(ctk.CTkToplevel):
    def __init__(self, parent, username: str, password: str, user_manager: UserManager, on_save):
        super().__init__(parent)
        self.username = username
        self.password = password
        self.user_manager = user_manager
        self.on_save = on_save
        self.title("设置 DeepSeek API 密钥")
        self.geometry("500x200")
        self.resizable(False, False)

        self.label = ctk.CTkLabel(self, text="请输入您的 DeepSeek API 密钥\n（可在 https://platform.deepseek.com/api_keys 获取）")
        self.label.pack(pady=15)

        self.entry = ctk.CTkEntry(self, width=400, placeholder_text="sk-...")
        self.entry.pack(pady=10)

        self.btn_save = ctk.CTkButton(self, text="保存并继续", command=self.save_key)
        self.btn_save.pack(pady=10)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def save_key(self):
        api_key = self.entry.get().strip()
        if not api_key:
            messagebox.showerror("错误", "请输入 API 密钥")
            return
        self.user_manager.save_api_key(self.username, api_key, self.password)
        self.on_save(api_key)
        self.destroy()

    def on_close(self):
        self.destroy()
        self.master.quit()

# ------------------------------
# 主界面
# ------------------------------
class DeepSeekApp(ctk.CTk):
    def __init__(self, username: str, password: str, user_manager: UserManager, api_key: str):
        super().__init__()
        self.username = username
        self.password = password
        self.user_manager = user_manager
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

        self.chat_manager = ChatManager(username)
        self.current_chat_id = None
        self.current_messages: List[Dict] = []
        self.is_generating = False
        self.stop_generation = False

        self.title(f"DeepSeek 助手 - {username}")
        self.geometry("1100x700")
        self.minsize(900, 500)

        # 菜单栏
        self.create_menu()

        # 布局
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 左侧对话列表
        self.sidebar_frame = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(1, weight=1)

        self.label_chats = ctk.CTkLabel(self.sidebar_frame, text="对话列表", font=ctk.CTkFont(weight="bold"))
        self.label_chats.grid(row=0, column=0, padx=10, pady=(10,5))

        self.btn_new_chat = ctk.CTkButton(self.sidebar_frame, text="+ 新建对话", command=self.new_chat)
        self.btn_new_chat.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        self.chat_list_frame = ctk.CTkScrollableFrame(self.sidebar_frame)
        self.chat_list_frame.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")
        self.chat_buttons = {}

        # 主聊天区域
        self.main_frame = ctk.CTkFrame(self, corner_radius=0)
        self.main_frame.grid(row=0, column=1, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        # 聊天显示区域（可滚动）
        self.chat_display = ctk.CTkScrollableFrame(self.main_frame)
        self.chat_display.grid(row=0, column=0, padx=10, pady=(10,5), sticky="nsew")
        self.chat_display.grid_columnconfigure(0, weight=1)

        # 输入区域
        self.input_frame = ctk.CTkFrame(self.main_frame)
        self.input_frame.grid(row=1, column=0, padx=10, pady=(0,10), sticky="ew")
        self.input_frame.grid_columnconfigure(0, weight=1)

        self.text_input = ctk.CTkTextbox(self.input_frame, height=80)
        self.text_input.grid(row=0, column=0, padx=(0,5), pady=5, sticky="ew")
        self.text_input.bind("<Control-Return>", lambda e: self.send_message())

        self.btn_frame = ctk.CTkFrame(self.input_frame, fg_color="transparent")
        self.btn_frame.grid(row=0, column=1, pady=5, sticky="ns")

        self.btn_send = ctk.CTkButton(self.btn_frame, text="发送", width=60, command=self.send_message)
        self.btn_send.pack(pady=2)
        self.btn_stop = ctk.CTkButton(self.btn_frame, text="停止", width=60, fg_color="#d9534f", hover_color="#c9302c", command=self.stop_generating)
        self.btn_stop.pack(pady=2)

        # 选项栏
        self.options_frame = ctk.CTkFrame(self.main_frame)
        self.options_frame.grid(row=2, column=0, padx=10, pady=(0,5), sticky="ew")

        self.thinking_var = ctk.BooleanVar(value=False)
        self.cb_thinking = ctk.CTkCheckBox(self.options_frame, text="深度思考模式", variable=self.thinking_var)
        self.cb_thinking.pack(side="left", padx=10)

        self.search_var = ctk.BooleanVar(value=False)
        self.cb_search = ctk.CTkCheckBox(self.options_frame, text="智能搜索", variable=self.search_var)
        self.cb_search.pack(side="left", padx=10)

        # 初始化
        self.refresh_chat_list()
        self.new_chat()  # 默认新建对话

    # ---------- 菜单 ----------
    def create_menu(self):
        menubar = tk.Menu(self)
        # 主题菜单
        theme_menu = tk.Menu(menubar, tearoff=0)
        theme_menu.add_command(label="浅色模式", command=lambda: ctk.set_appearance_mode("Light"))
        theme_menu.add_command(label="深色模式", command=lambda: ctk.set_appearance_mode("Dark"))
        theme_menu.add_command(label="跟随系统", command=lambda: ctk.set_appearance_mode("System"))
        menubar.add_cascade(label="主题", menu=theme_menu)

        # API 密钥菜单
        menubar.add_command(label="修改 API 密钥", command=self.change_api_key)

        self.configure(menu=menubar)

    def change_api_key(self):
        def on_save(new_key):
            self.api_key = new_key
            self.client = OpenAI(api_key=new_key, base_url=DEEPSEEK_BASE_URL)
        APIKeyWindow(self, self.username, self.password, self.user_manager, on_save)

    # ---------- 对话列表 ----------
    def refresh_chat_list(self):
        # 清除现有按钮
        for btn in self.chat_buttons.values():
            btn.destroy()
        self.chat_buttons.clear()

        chats = self.chat_manager.list_chats()
        for i, chat in enumerate(chats):
            btn = ctk.CTkButton(
                self.chat_list_frame,
                text=chat["name"],
                anchor="w",
                command=lambda cid=chat["id"]: self.load_chat(cid)
            )
            btn.grid(row=i, column=0, padx=5, pady=2, sticky="ew")
            # 右键菜单
            btn.bind("<Button-3>", lambda e, cid=chat["id"], name=chat["name"]: self.show_chat_menu(e, cid, name))
            self.chat_buttons[chat["id"]] = btn

    def show_chat_menu(self, event, chat_id, chat_name):
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="重命名", command=lambda: self.rename_chat_dialog(chat_id))
        menu.add_command(label="删除", command=lambda: self.delete_chat(chat_id))
        menu.tk_popup(event.x_root, event.y_root)

    def rename_chat_dialog(self, chat_id):
        dialog = ctk.CTkInputDialog(text="输入新对话名称:", title="重命名")
        new_name = dialog.get_input()
        if new_name:
            self.chat_manager.rename_chat(chat_id, new_name)
            self.refresh_chat_list()
            if self.current_chat_id == chat_id:
                self.title(f"DeepSeek 助手 - {self.username} - {new_name}")

    def delete_chat(self, chat_id):
        if messagebox.askyesno("确认", "确定删除该对话吗？"):
            self.chat_manager.delete_chat(chat_id)
            if self.current_chat_id == chat_id:
                self.new_chat()
            self.refresh_chat_list()

    def new_chat(self):
        chat_id = self.chat_manager.create_chat()
        self.load_chat(chat_id)

    def load_chat(self, chat_id):
        self.current_chat_id = chat_id
        data = self.chat_manager.load_chat(chat_id)
        self.current_messages = data.get("messages", [])
        self.title(f"DeepSeek 助手 - {self.username} - {data.get('name', '新对话')}")
        self.render_messages()
        self.refresh_chat_list()

    # ---------- 渲染消息 ----------
    def render_messages(self):
        # 清空显示区域
        for widget in self.chat_display.winfo_children():
            widget.destroy()

        for i, msg in enumerate(self.current_messages):
            self.add_message_bubble(msg["role"], msg["content"], save=False)

        # 滚动到底部
        self.chat_display._parent_canvas.yview_moveto(1.0)

    def add_message_bubble(self, role: str, content: str, save: bool = True):
        """在界面上添加一个聊天气泡，并可选保存到消息列表"""
        frame = ctk.CTkFrame(self.chat_display, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=1)
        frame.pack(fill="x", pady=5, padx=10)

        if role == "user":
            bubble = ctk.CTkFrame(frame, fg_color="#2b5b84", corner_radius=15)
            bubble.pack(side="right", padx=5)
            label = ctk.CTkLabel(bubble, text=content, wraplength=500, justify="left")
            label.pack(padx=10, pady=8)
        else:
            bubble = ctk.CTkFrame(frame, fg_color="#3a3a3a", corner_radius=15)
            bubble.pack(side="left", padx=5)
            # 检查是否有代码块
            self.render_assistant_content(bubble, content)

        if save:
            self.current_messages.append({"role": role, "content": content})
            self.chat_manager.save_chat(self.current_chat_id, self.current_messages)

        # 自动滚动
        self.chat_display._parent_canvas.yview_moveto(1.0)

    def render_assistant_content(self, parent_frame, content: str):
        """解析助手消息，分离文本和代码块"""
        import re
        pattern = r"```(\w*)\n(.*?)```"
        parts = re.split(pattern, content, flags=re.DOTALL)

        # 如果无代码块
        if len(parts) == 1:
            label = ctk.CTkLabel(parent_frame, text=content, wraplength=500, justify="left")
            label.pack(padx=10, pady=8)
            return

        # 有代码块：交替显示文本和代码框
        text_index = 0
        for i in range(0, len(parts), 4):
            # parts[i] 是代码块之前的文本
            text_part = parts[i]
            if text_part.strip():
                label = ctk.CTkLabel(parent_frame, text=text_part, wraplength=500, justify="left")
                label.pack(padx=10, pady=4, anchor="w")

            if i+3 < len(parts):
                lang = parts[i+1] or "text"
                code = parts[i+2].strip()
                # 创建代码框
                code_frame = ctk.CTkFrame(parent_frame, fg_color="#1e1e1e", corner_radius=8)
                code_frame.pack(fill="x", padx=10, pady=5)

                # 工具栏
                toolbar = ctk.CTkFrame(code_frame, fg_color="#2d2d2d", height=30)
                toolbar.pack(fill="x")
                lang_label = ctk.CTkLabel(toolbar, text=lang, font=ctk.CTkFont(size=12))
                lang_label.pack(side="left", padx=10)

                copy_btn = ctk.CTkButton(toolbar, text="复制", width=50, height=20, command=lambda c=code: self.copy_code(c))
                copy_btn.pack(side="right", padx=5)
                download_btn = ctk.CTkButton(toolbar, text="下载", width=50, height=20, command=lambda c=code, l=lang: self.download_code(c, l))
                download_btn.pack(side="right", padx=5)

                # 代码文本
                code_text = ctk.CTkTextbox(code_frame, height=min(len(code.split('\n'))*20, 200), font=("Consolas", 12))
                code_text.insert("1.0", code)
                code_text.configure(state="disabled")
                code_text.pack(fill="both", padx=5, pady=5)

    def copy_code(self, code: str):
        pyperclip.copy(code)
        messagebox.showinfo("成功", "代码已复制到剪贴板")

    def download_code(self, code: str, lang: str):
        ext_map = {"python": ".py", "javascript": ".js", "html": ".html", "css": ".css", "java": ".java", "cpp": ".cpp", "c": ".c"}
        ext = ext_map.get(lang.lower(), ".txt")
        file_path = filedialog.asksaveasfilename(defaultextension=ext, filetypes=[(f"{lang}文件", f"*{ext}"), ("所有文件", "*.*")])
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(code)
            messagebox.showinfo("成功", f"代码已保存至 {file_path}")

    # ---------- 发送消息 ----------
    def send_message(self):
        if self.is_generating:
            messagebox.showwarning("提示", "正在生成回复，请稍后或点击停止")
            return

        user_input = self.text_input.get("1.0", "end-1c").strip()
        if not user_input:
            return

        # 自动命名（如果是新对话且无名称）
        data = self.chat_manager.load_chat(self.current_chat_id)
        if data.get("name") == "新对话":
            new_name = user_input[:20] + ("..." if len(user_input)>20 else "")
            self.chat_manager.rename_chat(self.current_chat_id, new_name)
            self.title(f"DeepSeek 助手 - {self.username} - {new_name}")

        self.text_input.delete("1.0", "end")
        self.add_message_bubble("user", user_input)

        # 启动生成线程
        self.is_generating = True
        self.stop_generation = False
        self.btn_send.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        threading.Thread(target=self.generate_response, args=(user_input,), daemon=True).start()

    def stop_generating(self):
        self.stop_generation = True

    def generate_response(self, user_input: str):
        # 准备消息（仅发送当前对话历史，不含系统提示）
        messages = [{"role": m["role"], "content": m["content"]} for m in self.current_messages]

        # 添加额外参数
        extra_body = {}
        if self.thinking_var.get():
            extra_body["thinking"] = {"type": "enabled"}
        if self.search_var.get():
            extra_body["enable_search"] = True

        try:
            stream = self.client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=messages,
                stream=True,
                extra_body=extra_body if extra_body else None
            )

            # 创建助手消息气泡（占位）
            assistant_frame = ctk.CTkFrame(self.chat_display, fg_color="transparent")
            assistant_frame.grid_columnconfigure(0, weight=1)
            assistant_frame.pack(fill="x", pady=5, padx=10)
            bubble = ctk.CTkFrame(assistant_frame, fg_color="#3a3a3a", corner_radius=15)
            bubble.pack(side="left", padx=5)

            # 先用一个临时标签显示“正在思考...”
            temp_label = ctk.CTkLabel(bubble, text="正在思考...", wraplength=500, justify="left")
            temp_label.pack(padx=10, pady=8)

            full_response = ""
            thinking_content = ""
            for chunk in stream:
                if self.stop_generation:
                    break
                delta = chunk.choices[0].delta
                # 处理思考内容（如果开启）
                if hasattr(delta, 'thinking_content') and delta.thinking_content:
                    thinking_content += delta.thinking_content
                if delta.content:
                    full_response += delta.content
                    temp_label.configure(text=full_response + ("..."))
                    self.update_idletasks()

            # 移除临时标签
            temp_label.destroy()

            # 最终渲染（带代码解析）
            final_content = full_response
            if thinking_content and self.thinking_var.get():
                final_content = f"[思考过程]\n{thinking_content}\n\n[回复]\n{full_response}"
            self.render_assistant_content(bubble, final_content)

            # 保存消息
            self.current_messages.append({"role": "assistant", "content": final_content})
            self.chat_manager.save_chat(self.current_chat_id, self.current_messages)

        except Exception as e:
            messagebox.showerror("错误", f"API 调用失败: {str(e)}")
            # 移除占位
            assistant_frame.destroy()
        finally:
            self.is_generating = False
            self.btn_send.configure(state="normal")
            self.btn_stop.configure(state="disabled")
            self.refresh_chat_list()

    def on_close(self):
        self.destroy()
        self.quit()

# ------------------------------
# 启动入口
# ------------------------------
def main():
    user_manager = UserManager()

    # 创建临时根窗口用于登录
    root = ctk.CTk()
    root.withdraw()  # 隐藏

    def on_login_success(username, password):
        root.deiconify()
        # 检查 API 密钥
        api_key = user_manager.get_api_key(username, password)
        if not api_key:
            def on_key_saved(key):
                app = DeepSeekApp(username, password, user_manager, key)
                app.protocol("WM_DELETE_WINDOW", app.on_close)
                app.mainloop()
            APIKeyWindow(root, username, password, user_manager, on_key_saved)
        else:
            app = DeepSeekApp(username, password, user_manager, api_key)
            app.protocol("WM_DELETE_WINDOW", app.on_close)
            app.mainloop()

    login_window = LoginWindow(root, user_manager, on_login_success)
    root.wait_window(login_window)  # 等待登录窗口关闭
    root.destroy()

if __name__ == "__main__":
    main()