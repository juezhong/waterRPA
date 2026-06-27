import sys
import os
import time
import json
import pyautogui
import pyperclip
import traceback
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QLabel, QComboBox, QLineEdit, QScrollArea, 
                               QFileDialog, QTextEdit, QMessageBox, QFrame)
from PySide6.QtCore import Qt, QThread, Signal
from pynput.keyboard import GlobalHotKeys, Key
import cv2
import numpy as np
import mss
import atexit
from collections import OrderedDict
# --------------------------
# 核心逻辑 (原 waterRPA.py)
# --------------------------

# ---- 多屏截图定位 ----
# mss（CoreGraphics 直抓）+ OpenCV 灰度匹配，全局单例复用 avoid 重复创建开销。
# 自动适配 Retina：模板在 1.0x/0.5x 两个尺度尝试匹配。

_mss = mss.MSS()  # 模块级单例，全生命周期复用

# 模板 LRU 缓存 + 位置缓存
_NEEDLE_CACHE_SIZE = 3
_recog_cache = {
    "needles": OrderedDict(),  # img_path -> (gray_full, gray_half), LRU
    "last_hits": {},           # img_path -> (mon_idx, abs_x, abs_y)
}


_active_monitor_cache = None  # 任务开始时锁定，避免执行期间鼠标跳屏


def _reset_recog_cache():
    """每次 run_tasks 开始时清除位置缓存，重新检测当前屏幕。"""
    global _active_monitor_cache
    _recog_cache["last_hits"].clear()
    _active_monitor_cache = None  # 下次 _get_active_monitor_idx 重新计算


def _clear_all_cache():
    """程序退出时释放所有缓存。"""
    _recog_cache["needles"].clear()
    _recog_cache["last_hits"].clear()


atexit.register(_clear_all_cache)


def _get_needle(img_path):
    """LRU 缓存模板，预计算 0.5x 降采样版本，上限 3 个。"""
    if img_path in _recog_cache["needles"]:
        # 命中：移到末尾（标记最近使用）
        _recog_cache["needles"].move_to_end(img_path)
        return _recog_cache["needles"][img_path]

    # 未命中：加载 + 预计算
    full = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if full is None:
        return None, None
    half = cv2.resize(full, None, fx=0.5, fy=0.5) if min(full.shape) > 20 else full
    pair = (full, half)

    # LRU 淘汰
    if len(_recog_cache["needles"]) >= _NEEDLE_CACHE_SIZE:
        _recog_cache["needles"].popitem(last=False)

    _recog_cache["needles"][img_path] = pair
    return pair


def _get_active_monitor_idx():
    """返回任务启动时所在显示器的 mss 索引（1-based），一次计算后缓存。"""
    global _active_monitor_cache
    if _active_monitor_cache is not None:
        return _active_monitor_cache
    mx, my = pyautogui.position()
    for i, mon in enumerate(_mss.monitors[1:], start=1):
        if mon["left"] <= mx < mon["left"] + mon["width"] and \
           mon["top"] <= my < mon["top"] + mon["height"]:
            _active_monitor_cache = i
            return i
    _active_monitor_cache = 1
    return 1


def _match_on_haystack(haystack_gray, needle_full, needle_half, confidence):
    """在给定灰度图上匹配模板。返回 (abs_cx, abs_cy, label) 或 None。
    abs_cx/cy 是 haystack 内的坐标（不含显示器偏移），label 是日志标签。"""
    hh, hw = haystack_gray.shape
    nfh, nfw = needle_full.shape

    # --- 0.5x 降采样（主路径）---
    nhh, nhw = needle_half.shape
    if nhh <= hh and nhw <= hw and min(nfh, nfw) > 20:
        hay_half = cv2.resize(haystack_gray, None, fx=0.5, fy=0.5)
        result = cv2.matchTemplate(hay_half, needle_half, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if max_val >= confidence:
            cx = (max_loc[0] + nhw // 2) * 2
            cy = (max_loc[1] + nhh // 2) * 2
            return (cx, cy, f"s=0.5 c=+{max_val:.2f}")
        if min_val <= -confidence:
            cx = (min_loc[0] + nhw // 2) * 2
            cy = (min_loc[1] + nhh // 2) * 2
            return (cx, cy, f"s=0.5 c={min_val:.2f}(inv)")

    # --- 1.0x 回退 ---
    if nfh <= hh and nfw <= hw:
        result = cv2.matchTemplate(haystack_gray, needle_full, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if max_val >= confidence:
            cx = max_loc[0] + nfw // 2
            cy = max_loc[1] + nfh // 2
            return (cx, cy, f"s=1.0 c=+{max_val:.2f}")
        if min_val <= -confidence:
            cx = min_loc[0] + nfw // 2
            cy = min_loc[1] + nfh // 2
            return (cx, cy, f"s=1.0 c={min_val:.2f}(inv)")

    return None


_REGION_MARGIN = 200


def _locate_on_all_screens(img_path, confidence=0.9):
    """mss 快速截图 + OpenCV 灰度匹配，带区域缓存和屏幕优先级。"""
    needle_full, needle_half = _get_needle(img_path)
    if needle_full is None:
        return None

    # --- Phase 1: ±200px 区域优先搜索 ---
    last = _recog_cache["last_hits"].get(img_path)
    if last is not None:
        prev_idx, prev_x, prev_y = last
        if 0 <= prev_idx < len(_mss.monitors):
            mon = _mss.monitors[prev_idx]
            l = max(mon["left"], prev_x - _REGION_MARGIN)
            t = max(mon["top"], prev_y - _REGION_MARGIN)
            r = min(mon["left"] + mon["width"], prev_x + _REGION_MARGIN)
            b = min(mon["top"] + mon["height"], prev_y + _REGION_MARGIN)
            rw, rh = r - l, b - t

            if rw > needle_full.shape[1] and rh > needle_full.shape[0]:
                try:
                    reg = {"left": l, "top": t, "width": rw, "height": rh}
                    shot = _mss.grab(reg)
                    bgra = np.frombuffer(shot.bgra, dtype=np.uint8).reshape(
                        (shot.height, shot.width, 4))
                    haystack = cv2.cvtColor(bgra, cv2.COLOR_BGRA2GRAY)

                    found = _match_on_haystack(
                        haystack, needle_full, needle_half, confidence)
                    if found is not None:
                        cx, cy, label = found
                        abs_x, abs_y = l + cx, t + cy
                        _recog_cache["last_hits"][img_path] = (prev_idx, abs_x, abs_y)
                        print(f"[匹配] {os.path.basename(img_path)} "
                              f"Rgn屏{prev_idx} {label}→({abs_x},{abs_y})")
                        return pyautogui.Point(abs_x, abs_y)
                except Exception:
                    pass
            # 区域失败 → 清除缓存，走全屏搜索
            del _recog_cache["last_hits"][img_path]

    # --- Phase 2: 全屏搜索（优先跨屏）---
    active = _get_active_monitor_idx()
    indices = list(range(1, len(_mss.monitors)))
    if len(indices) > 1:
        indices = [i for i in indices if i != active] + [active]

    for i in indices:
        mon = _mss.monitors[i]
        shot = _mss.grab(mon)
        bgra = np.frombuffer(shot.bgra, dtype=np.uint8).reshape(
            (shot.height, shot.width, 4))
        haystack_gray = cv2.cvtColor(bgra, cv2.COLOR_BGRA2GRAY)

        found = _match_on_haystack(haystack_gray, needle_full, needle_half, confidence)
        if found is not None:
            cx, cy, label = found
            abs_x, abs_y = mon["left"] + cx, mon["top"] + cy
            _recog_cache["last_hits"][img_path] = (i, abs_x, abs_y)
            print(f"[匹配] {os.path.basename(img_path)} "
                  f"全屏屏{i} {label}→({abs_x},{abs_y})")
            return pyautogui.Point(abs_x, abs_y)

    # 没找到 → 清除过期缓存
    _recog_cache["last_hits"].pop(img_path, None)
    return None
# ------------------------------

SETTLE_DELAY = 0.15  # 动作后沉降时间，等待 UI 动画完成


def mouseClick(clickTimes, lOrR, img, reTry, timeout=60, stop_check=None):
    """
    reTry: 1 (一次), -1 (无限), >1 (指定次数)
    timeout: 超时时间(秒)，默认60秒。防止无限卡死。
    stop_check: 可调用对象，返回 True 时立即中断并返回。
    """
    start_time = time.time()

    def should_stop():
        return stop_check and stop_check()

    if reTry == 1:
        while True:
            # 检查停止信号
            if should_stop():
                print("收到停止信号，中断操作")
                return
            # 检查超时
            if timeout and (time.time() - start_time > timeout):
                print(f"等待图片 {img} 超时 ({timeout}秒)")
                return

            try:
                location=_locate_on_all_screens(img, confidence=0.9)
                if location is not None:
                    pyautogui.click(location.x,location.y,clicks=clickTimes,interval=0.02,duration=0.01,button=lOrR)
                    break
            except pyautogui.ImageNotFoundException:
                pass # 没找到，继续重试

            print("未找到匹配图片,重试中...")
            time.sleep(0.03)
    elif reTry == -1:
        while True:
            if should_stop():
                print("收到停止信号，中断操作")
                return
            if timeout and (time.time() - start_time > timeout):
                print(f"等待图片 {img} 超时 ({timeout}秒)")
                return

            try:
                location=_locate_on_all_screens(img, confidence=0.9)
                if location is not None:
                    pyautogui.click(location.x,location.y,clicks=clickTimes,interval=0.02,duration=0.01,button=lOrR)
            except pyautogui.ImageNotFoundException:
                pass

            time.sleep(0.03)
    elif reTry > 1:
        i = 1
        while i < reTry + 1:
            if should_stop():
                print("收到停止信号，中断操作")
                return
            if timeout and (time.time() - start_time > timeout):
                print(f"操作超时 ({timeout}秒)")
                return

            try:
                location=_locate_on_all_screens(img, confidence=0.9)
                if location is not None:
                    pyautogui.click(location.x,location.y,clicks=clickTimes,interval=0.02,duration=0.01,button=lOrR)
                    print("重复")
                    i += 1
            except pyautogui.ImageNotFoundException:
                pass

            time.sleep(0.03)

def mouseMove(img, reTry, timeout=60, stop_check=None):
    """
    鼠标悬停（移动但不点击）
    stop_check: 可调用对象，返回 True 时立即中断并返回。
    """
    start_time = time.time()
    while True:
        if stop_check and stop_check():
            print("收到停止信号，中断操作")
            return
        if timeout and (time.time() - start_time > timeout):
            print(f"等待图片 {img} 超时 ({timeout}秒)")
            return

        try:
            location = _locate_on_all_screens(img, confidence=0.9)
            if location is not None:
                pyautogui.moveTo(location.x, location.y, duration=0.01)
                break
        except pyautogui.ImageNotFoundException:
            pass

        print("未找到匹配图片,重试中...")
        time.sleep(0.03)
        if reTry == 1:
            pass
        # 注意：原mouseClick中 reTry=1 也是 while True，直到找到。这里保持一致。

class RPAEngine:
    def __init__(self):
        self.is_running = False
        self.stop_requested = False

    def stop(self):
        self.stop_requested = True
        self.is_running = False

    def run_tasks(self, tasks, loop_forever=False, callback_msg=None):
        """
        tasks: list of dict, format:
        [
            {"type": 1.0, "value": "1.png", "retry": 1},
            ...
        ]
        """
        self.is_running = True
        self.stop_requested = False
        # 重置位置缓存，模板缓存保留
        _reset_recog_cache()
        # 记录鼠标起始位置，任务结束后恢复
        start_pos = pyautogui.position()

        try:
            while True:
                for idx, task in enumerate(tasks):
                    if self.stop_requested:
                        if callback_msg: callback_msg("任务已停止")
                        return

                    cmd_type = task.get("type")
                    cmd_value = task.get("value")
                    retry = task.get("retry", 1)

                    if callback_msg:
                        callback_msg(f"执行步骤 {idx+1}: 类型={cmd_type}, 内容={cmd_value}")

                    if cmd_type == 1.0: # 单击左键
                        mouseClick(1, "left", cmd_value, retry, stop_check=lambda: self.stop_requested)
                        if callback_msg: callback_msg(f"单击左键: {cmd_value}")

                    elif cmd_type == 2.0: # 双击左键
                        mouseClick(2, "left", cmd_value, retry, stop_check=lambda: self.stop_requested)
                        if callback_msg: callback_msg(f"双击左键: {cmd_value}")

                    elif cmd_type == 3.0: # 右键
                        mouseClick(1, "right", cmd_value, retry, stop_check=lambda: self.stop_requested)
                        if callback_msg: callback_msg(f"右键单击: {cmd_value}")

                    elif cmd_type == 4.0: # 输入
                        pyperclip.copy(str(cmd_value))
                        pyautogui.hotkey('ctrl', 'v')
                        time.sleep(0.15)
                        if callback_msg: callback_msg(f"输入文本: {cmd_value}")

                    elif cmd_type == 5.0: # 等待 (可被停止信号中断)
                        sleep_time = float(cmd_value)
                        elapsed = 0.0
                        chunk = 0.1
                        while elapsed < sleep_time:
                            if self.stop_requested:
                                if callback_msg: callback_msg("等待被停止信号中断")
                                return
                            wait = min(chunk, sleep_time - elapsed)
                            time.sleep(wait)
                            elapsed += wait
                        if callback_msg: callback_msg(f"等待 {sleep_time} 秒")
                    
                    elif cmd_type == 6.0: # 滚轮
                        scroll_val = int(cmd_value)
                        pyautogui.scroll(scroll_val)
                        if callback_msg: callback_msg(f"滚轮滑动 {scroll_val}")

                    elif cmd_type == 7.0: # 系统按键 (组合键)
                        keys = str(cmd_value).lower().split('+')
                        # 去除空格
                        keys = [k.strip() for k in keys]
                        pyautogui.hotkey(*keys)
                        if callback_msg: callback_msg(f"按键组合: {cmd_value}")

                    elif cmd_type == 8.0: # 鼠标悬停
                        mouseMove(cmd_value, retry, stop_check=lambda: self.stop_requested)
                        if callback_msg: callback_msg(f"鼠标悬停: {cmd_value}")

                    elif cmd_type == 9.0: # 截图保存
                        path = str(cmd_value)
                        # 如果是目录，自动拼接时间戳文件名
                        if os.path.isdir(path):
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            filename = os.path.join(path, f"screenshot_{timestamp}.png")
                        else:
                            # 兼容旧逻辑：如果用户直接输入了带文件名的路径
                            filename = path
                            if not filename.endswith(('.png', '.jpg', '.bmp')):
                                filename += '.png'
                        
                        pyautogui.screenshot(filename)
                        if callback_msg: callback_msg(f"截图已保存: {filename}")

                    # 沉降：等待 UI 动画完成，防止下一帧匹配过早
                    if not self.stop_requested:
                        time.sleep(SETTLE_DELAY)

                if not loop_forever:
                    break

                if callback_msg: callback_msg("等待 0.1 秒进入下一轮循环...")
                time.sleep(0.03)
                
        except Exception as e:
            if callback_msg: callback_msg(f"执行出错: {e}")
            traceback.print_exc()
        finally:
            self.is_running = False
            # 鼠标回到起始位置
            try:
                pyautogui.moveTo(start_pos.x, start_pos.y, duration=0.01)
            except Exception:
                pass
            if callback_msg: callback_msg("任务结束")

# --------------------------
# GUI 界面 (原 rpa_gui.py)
# --------------------------

# 定义操作类型映射
CMD_TYPES = {
    "左键单击": 1.0,
    "左键双击": 2.0,
    "右键单击": 3.0,
    "输入文本": 4.0,
    "等待(秒)": 5.0,
    "滚轮滑动": 6.0,
    "系统按键": 7.0,
    "鼠标悬停": 8.0,
    "截图保存": 9.0
}

CMD_TYPES_REV = {v: k for k, v in CMD_TYPES.items()}

class WorkerThread(QThread):
    log_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self, engine, tasks, loop_forever):
        super().__init__()
        self.engine = engine
        self.tasks = tasks
        self.loop_forever = loop_forever

    def run(self):
        self.engine.run_tasks(self.tasks, self.loop_forever, self.log_callback)
        self.finished_signal.emit()

    def log_callback(self, msg):
        self.log_signal.emit(msg)

class HotkeyThread(QThread):
    """全局热键监听线程 — F7 开始执行，F8 停止执行"""
    start_signal = Signal()
    stop_signal = Signal()

    def __init__(self):
        super().__init__()
        self._listener = None

    def run(self):
        def on_f7():
            self.start_signal.emit()

        def on_f8():
            self.stop_signal.emit()

        with GlobalHotKeys({'<f7>': on_f7, '<f8>': on_f8}) as listener:
            self._listener = listener
            listener.join()

    def stop_listener(self):
        if self._listener is not None:
            self._listener.stop()


class TaskRow(QFrame):
    def __init__(self, parent_layout, delete_callback):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        
        # 操作类型选择
        self.type_combo = QComboBox()
        self.type_combo.addItems(list(CMD_TYPES.keys()))
        self.type_combo.currentTextChanged.connect(self.on_type_changed)
        self.layout.addWidget(self.type_combo)
        
        # 参数输入区域
        self.value_input = QLineEdit()
        self.value_input.setPlaceholderText("参数值 (如图片路径、文本、时间)")
        self.layout.addWidget(self.value_input)
        
        # 文件选择按钮 (默认隐藏)
        self.file_btn = QPushButton("选择图片")
        self.file_btn.clicked.connect(self.select_file)
        self.file_btn.setVisible(True) # 默认是左键单击，需要显示
        self.layout.addWidget(self.file_btn)
        
        # 重试次数 (默认隐藏)
        self.retry_input = QLineEdit()
        self.retry_input.setPlaceholderText("重试次数 (1=一次, -1=无限)")
        self.retry_input.setText("1")
        self.retry_input.setFixedWidth(100)
        self.retry_input.setVisible(True)
        self.layout.addWidget(self.retry_input)
        
        # 删除按钮
        self.del_btn = QPushButton("X")
        self.del_btn.setStyleSheet("color: red; font-weight: bold;")
        self.del_btn.setFixedWidth(30)
        self.del_btn.clicked.connect(lambda: delete_callback(self))
        self.layout.addWidget(self.del_btn)
        
        parent_layout.addWidget(self)

    def on_type_changed(self, text):
        cmd_type = CMD_TYPES[text]
        
        # 图片相关操作 (1, 2, 3, 8)
        if cmd_type in [1.0, 2.0, 3.0, 8.0]:
            self.file_btn.setVisible(True)
            self.file_btn.setText("选择图片")
            self.retry_input.setVisible(True)
            self.value_input.setPlaceholderText("图片路径")
        # 输入 (4)
        elif cmd_type == 4.0:
            self.file_btn.setVisible(False)
            self.retry_input.setVisible(False)
            self.value_input.setPlaceholderText("请输入要发送的文本")
        # 等待 (5)
        elif cmd_type == 5.0:
            self.file_btn.setVisible(False)
            self.retry_input.setVisible(False)
            self.value_input.setPlaceholderText("等待秒数 (如 1.5)")
        # 滚轮 (6)
        elif cmd_type == 6.0:
            self.file_btn.setVisible(False)
            self.retry_input.setVisible(False)
            self.value_input.setPlaceholderText("滚动距离 (正数向上，负数向下)")
        # 系统按键 (7)
        elif cmd_type == 7.0:
            self.file_btn.setVisible(False)
            self.retry_input.setVisible(False)
            self.value_input.setPlaceholderText("组合键 (如 ctrl+s, alt+tab)")
        # 截图保存 (9)
        elif cmd_type == 9.0:
            self.file_btn.setVisible(True)
            self.file_btn.setText("选择保存文件夹")
            self.retry_input.setVisible(False)
            self.value_input.setPlaceholderText("保存目录 (如 D:\\Screenshots)")

    def set_data(self, data):
        """用于回填数据"""
        cmd_type = data.get("type")
        value = data.get("value", "")
        retry = data.get("retry", 1)

        # 设置类型 (反向查找文本)
        if cmd_type in CMD_TYPES_REV:
            self.type_combo.setCurrentText(CMD_TYPES_REV[cmd_type])
        
        # 设置值
        self.value_input.setText(str(value))
        
        # 设置重试次数
        self.retry_input.setText(str(retry))

    def select_file(self):
        cmd_type = CMD_TYPES[self.type_combo.currentText()]
        
        # 截图保存 (9.0) -> 选择文件夹
        if cmd_type == 9.0:
            folder = QFileDialog.getExistingDirectory(self, "选择保存文件夹", os.getcwd())
            if folder:
                self.value_input.setText(folder)
        
        # 其他图片操作 (1, 2, 3, 8) -> 打开文件对话框
        else:
            filename, _ = QFileDialog.getOpenFileName(self, "选择图片", os.getcwd(), "Image Files (*.png *.jpg *.bmp)")
            if filename:
                self.value_input.setText(filename)

    def get_data(self):
        cmd_type = CMD_TYPES[self.type_combo.currentText()]
        value = self.value_input.text()
        
        # 数据校验与转换
        try:
            if cmd_type in [5.0, 6.0]:
                # 尝试转换为数字，如果失败可能会在运行时报错，这里简单处理
                if not value: value = "0"
            
            retry = 1
            if self.retry_input.isVisible():
                retry_text = self.retry_input.text()
                if retry_text:
                    retry = int(retry_text)
        except ValueError:
            pass # 保持默认

        return {
            "type": cmd_type,
            "value": value,
            "retry": retry
        }

class RPAWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("不高兴就喝水 RPA 配置工具")
        self.resize(800, 600)
        
        self.engine = RPAEngine()
        self.worker = None
        self.rows = []

        # 全局热键 F7 开始 / F8 停止
        self.hotkey_thread = HotkeyThread()
        self.hotkey_thread.start_signal.connect(self.start_task)
        self.hotkey_thread.stop_signal.connect(self.stop_task)
        self.hotkey_thread.start()

        # 主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 顶部控制栏
        top_bar = QHBoxLayout()
        
        self.add_btn = QPushButton("+ 新增指令")
        self.add_btn.clicked.connect(self.add_row)
        top_bar.addWidget(self.add_btn)

        self.save_btn = QPushButton("保存配置")
        self.save_btn.clicked.connect(self.save_config)
        top_bar.addWidget(self.save_btn)

        self.load_btn = QPushButton("导入配置")
        self.load_btn.clicked.connect(self.load_config)
        top_bar.addWidget(self.load_btn)
        
        top_bar.addStretch()
        
        self.loop_check = QComboBox()
        self.loop_check.addItems(["执行一次", "循环执行"])
        top_bar.addWidget(self.loop_check)
        
        self.start_btn = QPushButton("开始运行")
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        self.start_btn.clicked.connect(self.start_task)
        top_bar.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white;")
        self.stop_btn.clicked.connect(self.stop_task)
        self.stop_btn.setEnabled(False)
        top_bar.addWidget(self.stop_btn)

        # 快捷键提示
        hotkey_hint = QLabel("F7 启动 | F8 停止")
        hotkey_hint.setStyleSheet("color: #888; font-size: 11px; padding: 2px 8px;")
        top_bar.addWidget(hotkey_hint)

        main_layout.addLayout(top_bar)

        # 任务列表区域 (滚动)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.task_container = QWidget()
        self.task_layout = QVBoxLayout(self.task_container)
        self.task_layout.addStretch() # 弹簧，确保添加的行在顶部
        scroll.setWidget(self.task_container)
        main_layout.addWidget(scroll)

        # 日志区域
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(150)
        main_layout.addWidget(QLabel("运行日志:"))
        main_layout.addWidget(self.log_area)

        # 初始添加一行
        self.add_row()

    def add_row(self, data=None):
        # 移除底部的弹簧
        self.task_layout.takeAt(self.task_layout.count() - 1)
        
        row = TaskRow(self.task_layout, self.delete_row)
        if data:
            row.set_data(data)
        self.rows.append(row)
        
        # 加回弹簧
        self.task_layout.addStretch()

    def delete_row(self, row_widget):
        if row_widget in self.rows:
            self.rows.remove(row_widget)
            row_widget.deleteLater()
            
    def save_config(self):
        tasks = []
        for row in self.rows:
            data = row.get_data()
            # 允许保存空值，方便后续编辑
            tasks.append(data)
            
        if not tasks:
            QMessageBox.warning(self, "警告", "没有可保存的配置")
            return

        filename, _ = QFileDialog.getSaveFileName(self, "保存配置", os.getcwd(), "JSON Files (*.json);;Text Files (*.txt)")
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(tasks, f, indent=4, ensure_ascii=False)
                QMessageBox.information(self, "成功", "配置已保存！")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {e}")

    def load_config(self):
        filename, _ = QFileDialog.getOpenFileName(self, "导入配置", os.getcwd(), "JSON Files (*.json);;Text Files (*.txt)")
        if not filename:
            return
            
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                tasks = json.load(f)
            
            if not isinstance(tasks, list):
                raise ValueError("文件格式不正确")

            # 清空现有行
            for row in self.rows:
                row.deleteLater()
            self.rows.clear()
            
            # 重新添加行
            for task in tasks:
                self.add_row(task)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"导入失败: {e}")

    def start_task(self):
        tasks = []
        for row in self.rows:
            data = row.get_data()
            if not data['value']:
                QMessageBox.warning(self, "警告", "请检查有空参数的指令！")
                return
            tasks.append(data)
            
        if not tasks:
            QMessageBox.warning(self, "警告", "请至少添加一条指令！")
            return

        self.log_area.clear()
        self.log("任务开始...")
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.add_btn.setEnabled(False)
        
        loop = (self.loop_check.currentText() == "循环执行")
        
        self.worker = WorkerThread(self.engine, tasks, loop)
        self.worker.log_signal.connect(self.log)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()

        # 最小化窗口
        self.showMinimized()

    def stop_task(self):
        self.engine.stop()
        self.log("正在停止...")

    def on_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.add_btn.setEnabled(True)
        self.log("任务已结束")
        
        # 恢复窗口并置顶
        self.showNormal()
        self.activateWindow()

    def log(self, msg):
        self.log_area.append(msg)

    def closeEvent(self, event):
        """窗口关闭事件：确保线程停止，防止残留"""
        if self.worker and self.worker.isRunning():
            self.engine.stop()
            self.worker.quit()
            self.worker.wait()
        # 停止全局热键监听
        self.hotkey_thread.stop_listener()
        self.hotkey_thread.quit()
        self.hotkey_thread.wait()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = RPAWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
