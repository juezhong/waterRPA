import sys
import os
import time
import json
import pyautogui
import pyperclip
import traceback
import shutil
import logging
from datetime import datetime
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QLabel, QComboBox, QLineEdit, QScrollArea, 
                               QFileDialog, QTextEdit, QMessageBox, QFrame, QDialog)
from PySide6.QtCore import Qt, QThread, Signal
from pynput.keyboard import GlobalHotKeys, Key
from pynput.mouse import Listener as MouseListener, Button as MouseButton
import cv2
import numpy as np
import mss
import atexit
from collections import OrderedDict
# ---- 日志 ----
_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_log_date = datetime.now().strftime("%Y-%m-%d")
_log_file = os.path.join(_LOG_DIR, f"{_log_date}.log")

_logger = logging.getLogger("waterRPA")
_logger.setLevel(logging.DEBUG)
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                         datefmt="%H:%M:%S")
# 终端输出
_sh = logging.StreamHandler()
_sh.setLevel(logging.INFO)
_sh.setFormatter(_fmt)
_logger.addHandler(_sh)
# 文件输出
_fh = logging.FileHandler(_log_file, encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(_fmt)
_logger.addHandler(_fh)
# ------------------------------


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
    """在给定灰度图上匹配模板。返回 (cx, cy, label) 或 None。
    cx/cy 是 haystack 内的坐标（不含显示器偏移）。"""
    hh, hw = haystack_gray.shape
    nfh, nfw = needle_full.shape

    # --- 0.5x needle 降采样匹配（haystack 保持原分辨率，避免 resize 插值损失精度）---
    nhh, nhw = needle_half.shape
    if nhh <= hh and nhw <= hw and min(nfh, nfw) > 20:
        result = cv2.matchTemplate(haystack_gray, needle_half, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if max_val >= confidence:
            cx = max_loc[0] + nhw // 2
            cy = max_loc[1] + nhh // 2
            return (cx, cy, f"s=0.5 c=+{max_val:.2f}")
        if min_val <= -confidence:
            cx = min_loc[0] + nhw // 2
            cy = min_loc[1] + nhh // 2
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
    """mss 截图 + OpenCV 匹配，带区域缓存、屏幕优先级、主题反转适配。"""
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
                        _logger.info(f"匹配 {os.path.basename(img_path)} "
                                    f"Rgn屏{prev_idx} {label}→({abs_x},{abs_y})")
                        return pyautogui.Point(abs_x, abs_y)
                except Exception:
                    pass
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
            _logger.info(f"匹配 {os.path.basename(img_path)} "
                        f"全屏屏{i} {label}→({abs_x},{abs_y})")
            return pyautogui.Point(abs_x, abs_y)

    _recog_cache["last_hits"].pop(img_path, None)
    return None
# ------------------------------

SETTLE_DELAY = 0.35  # 动作后沉降时间，等待 UI 动画完成


def mouseClick(clickTimes, lOrR, img, reTry, timeout=60, stop_check=None):
    """
    reTry: 1 (一次), -1 (无限), >1 (指定次数)
    timeout: 超时时间(秒)，默认60秒。防止无限卡死。
    stop_check: 可调用对象，返回 True 时立即中断并返回。
    """
    # 坐标直达
    if _is_coordinate(img):
        x, y = _parse_coordinate(img)
        pyautogui.click(x, y, clicks=clickTimes, interval=0.15,
                        duration=0.25, button=lOrR)
        return

    start_time = time.time()

    def should_stop():
        return stop_check and stop_check()

    if reTry == 1:
        while True:
            # 检查停止信号
            if should_stop():
                _logger.info("收到停止信号，中断操作")
                return
            # 检查超时
            if timeout and (time.time() - start_time > timeout):
                _logger.info(f"等待图片超时: {img}")
                return

            try:
                location=_locate_on_all_screens(img, confidence=0.9)
                if location is not None:
                    pyautogui.click(location.x,location.y,clicks=clickTimes,interval=0.15,duration=0.25,button=lOrR)
                    break
            except pyautogui.ImageNotFoundException:
                pass # 没找到，继续重试

            _logger.debug("未找到匹配")
            time.sleep(0.03)
    elif reTry == -1:
        while True:
            if should_stop():
                _logger.info("收到停止信号，中断操作")
                return
            if timeout and (time.time() - start_time > timeout):
                _logger.info(f"等待图片超时: {img}")
                return

            try:
                location=_locate_on_all_screens(img, confidence=0.9)
                if location is not None:
                    pyautogui.click(location.x,location.y,clicks=clickTimes,interval=0.15,duration=0.25,button=lOrR)
            except pyautogui.ImageNotFoundException:
                pass

            time.sleep(0.03)
    elif reTry > 1:
        i = 1
        while i < reTry + 1:
            if should_stop():
                _logger.info("收到停止信号，中断操作")
                return
            if timeout and (time.time() - start_time > timeout):
                _logger.info(f"操作超时 ({timeout}s)")
                return

            try:
                location=_locate_on_all_screens(img, confidence=0.9)
                if location is not None:
                    pyautogui.click(location.x,location.y,clicks=clickTimes,interval=0.15,duration=0.25,button=lOrR)
                    _logger.debug("重复点击")
                    i += 1
            except pyautogui.ImageNotFoundException:
                pass

            time.sleep(0.03)

def mouseMove(img, reTry, timeout=60, stop_check=None):
    """
    鼠标悬停（移动但不点击）
    stop_check: 可调用对象，返回 True 时立即中断并返回。
    """
    # 坐标直达
    if _is_coordinate(img):
        x, y = _parse_coordinate(img)
        pyautogui.moveTo(x, y, duration=0.25)
        return

    start_time = time.time()
    while True:
        if stop_check and stop_check():
            _logger.info("收到停止信号，中断操作")
            return
        if timeout and (time.time() - start_time > timeout):
            _logger.info(f"等待图片超时: {img}")
            return

        try:
            location = _locate_on_all_screens(img, confidence=0.9)
            if location is not None:
                pyautogui.moveTo(location.x, location.y, duration=0.25)
                break
        except pyautogui.ImageNotFoundException:
            pass

        _logger.debug("未找到匹配")
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

    def _execute_drag(self, value, callback_msg=None):
        """执行拖移：value 格式 "sx,sy -> mx,my -> ex,ey"。"""
        parts = [p.strip() for p in value.split("->")]
        if len(parts) < 2:
            return
        points = [_parse_coordinate(p) for p in parts]
        sx, sy = points[0]
        pyautogui.moveTo(sx, sy, duration=0.2)
        pyautogui.mouseDown(button="left")
        for px, py in points[1:]:
            if self.stop_requested:
                pyautogui.mouseUp(button="left")
                return
            pyautogui.moveTo(px, py, duration=0.15)
        pyautogui.mouseUp(button="left")

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

                    # ── 坐标直接操作 (12-16, 11) ──
                    if cmd_type == 12.0:  # 坐标左键单击
                        x, y = _parse_coordinate(cmd_value)
                        pyautogui.click(x, y, clicks=1, interval=0.15,
                                        duration=0.25, button="left")
                        if callback_msg: callback_msg(f"坐标单击: {cmd_value}")
                    elif cmd_type == 13.0:  # 坐标左键双击
                        x, y = _parse_coordinate(cmd_value)
                        pyautogui.click(x, y, clicks=2, interval=0.15,
                                        duration=0.25, button="left")
                        if callback_msg: callback_msg(f"坐标双击: {cmd_value}")
                    elif cmd_type == 14.0:  # 坐标右键单击
                        x, y = _parse_coordinate(cmd_value)
                        pyautogui.click(x, y, clicks=1, interval=0.15,
                                        duration=0.25, button="right")
                        if callback_msg: callback_msg(f"坐标右键: {cmd_value}")
                    elif cmd_type == 15.0:  # 坐标中键单击
                        x, y = _parse_coordinate(cmd_value)
                        pyautogui.click(x, y, clicks=1, interval=0.15,
                                        duration=0.25, button="middle")
                        if callback_msg: callback_msg(f"坐标中键: {cmd_value}")
                    elif cmd_type == 16.0:  # 坐标鼠标悬停
                        x, y = _parse_coordinate(cmd_value)
                        pyautogui.moveTo(x, y, duration=0.25)
                        if callback_msg: callback_msg(f"坐标悬停: {cmd_value}")
                    elif cmd_type == 11.0:  # 坐标左键拖移
                        self._execute_drag(cmd_value, callback_msg)
                        if callback_msg: callback_msg(f"拖移: {cmd_value}")

                    # ── 图片识别操作 (1,2,3,10,8) ──
                    elif cmd_type == 1.0:
                        mouseClick(1, "left", cmd_value, retry, stop_check=lambda: self.stop_requested)
                        if callback_msg: callback_msg(f"图片单击: {cmd_value}")
                    elif cmd_type == 2.0:
                        mouseClick(2, "left", cmd_value, retry, stop_check=lambda: self.stop_requested)
                        if callback_msg: callback_msg(f"图片双击: {cmd_value}")
                    elif cmd_type == 3.0:
                        mouseClick(1, "right", cmd_value, retry, stop_check=lambda: self.stop_requested)
                        if callback_msg: callback_msg(f"图片右键: {cmd_value}")
                    elif cmd_type == 10.0:
                        mouseClick(1, "middle", cmd_value, retry, stop_check=lambda: self.stop_requested)
                        if callback_msg: callback_msg(f"图片中键: {cmd_value}")
                    elif cmd_type == 8.0:
                        mouseMove(cmd_value, retry, stop_check=lambda: self.stop_requested)
                        if callback_msg: callback_msg(f"图片悬停: {cmd_value}")

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
                pyautogui.moveTo(start_pos.x, start_pos.y, duration=0.25)
            except Exception:
                pass
            if callback_msg: callback_msg("任务结束")

# --------------------------
# GUI 界面 (原 rpa_gui.py)
# --------------------------

# 定义操作类型映射
CMD_TYPES = {
    # ── 鼠标坐标操作 ──
    "坐标左键单击": 12.0,
    "坐标左键双击": 13.0,
    "坐标右键单击": 14.0,
    "坐标中键单击": 15.0,
    "坐标鼠标悬停": 16.0,
    "坐标左键拖移": 11.0,
    # ── 图片识别操作 ──
    "图片左键单击": 1.0,
    "图片左键双击": 2.0,
    "图片右键单击": 3.0,
    "图片中键单击": 10.0,
    "图片鼠标悬停": 8.0,
    # ── 其他 ──
    "输入文本": 4.0,
    "等待(秒)": 5.0,
    "滚轮滑动": 6.0,
    "系统按键": 7.0,
    "截图保存": 9.0,
}

CMD_TYPES_REV = {v: k for k, v in CMD_TYPES.items()}

# 图片类操作：value 存图片路径，save_config 时迁移
_IMG_TYPES = {1.0, 2.0, 3.0, 8.0, 10.0}
# 坐标类操作：value 存坐标字符串，无需图像识别
_COORD_TYPES = {11.0, 12.0, 13.0, 14.0, 15.0, 16.0}
# 点击类：坐标 + 图片都有
_CLICK_TYPES = {1.0, 2.0, 3.0, 10.0, 12.0, 13.0, 14.0, 15.0}

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

import re


def _is_coordinate(value):
    """检测 value 是否为坐标格式 "x,y"。"""
    return isinstance(value, str) and bool(
        re.match(r'^\s*\d+\s*,\s*\d+\s*$', value))


def _parse_coordinate(value):
    """解析坐标字符串为 (x, y) 整数元组。"""
    x, y = value.split(",")
    return int(x.strip()), int(y.strip())


# 每类操作的参数名提示
_TYPE_PARAM_LABEL = {
    1.0: "图片路径", 2.0: "图片路径", 3.0: "图片路径",
    8.0: "图片路径", 10.0: "图片路径",
    11.0: "拖移坐标", 12.0: "点击坐标", 13.0: "点击坐标",
    14.0: "点击坐标", 15.0: "点击坐标", 16.0: "悬停坐标",
    4.0: "文本内容", 5.0: "等待秒数", 6.0: "滚动距离",
    7.0: "组合键", 9.0: "保存路径",
}

_TYPE_PLACEHOLDER = {
    1.0: "图片路径", 2.0: "图片路径", 3.0: "图片路径",
    8.0: "图片路径", 10.0: "图片路径",
    11.0: "起点x,起点y -> 中点x,中点y -> 终点x,终点y",
    12.0: "x,y  (如 500,300)", 13.0: "x,y",
    14.0: "x,y", 15.0: "x,y", 16.0: "x,y",
    4.0: "请输入要发送的文本", 5.0: "等待秒数 (如 1.5)",
    6.0: "滚动距离 (正数向上，负数向下)", 7.0: "组合键 (如 ctrl+s)",
    9.0: "保存目录 (如 D:\\Screenshots)",
}

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
    """全局热键 + 鼠标监听 — F7/F8 控制执行，F4 录制，F5 悬停"""
    start_signal = Signal()
    stop_signal = Signal()
    record_signal = Signal(str, int, int, int, int, int, int)
    recording_toggled = Signal(bool)

    def __init__(self):
        super().__init__()
        self._kb_listener = None
        self._ms_listener = None
        self.recording = False
        self._drag_start = None      # (x, y) 拖移起点
        self._drag_mid = None        # (x, y) 拖移中间点
        self._last_click_ts = 0.0    # 上次左键释放时间，双击判定

    def run(self):
        DBL_THRESH = 0.4   # 双击间隔上限
        DRAG_THRESH = 10   # 拖移判定最小像素

        def on_f4():
            if not self.recording:
                # 短暂延迟让窗口隐藏+listener就绪
                time.sleep(0.05)
            self.recording = not self.recording
            self._drag_start = None
            self._last_click_ts = 0.0
            self.recording_toggled.emit(self.recording)

        def on_f5():
            if self.recording:
                x, y = pyautogui.position()
                self.record_signal.emit("hover", x, y, 0, 0, 0, 0)

        def on_f7():
            if not self.recording:
                self.start_signal.emit()

        def on_f8():
            if not self.recording:
                self.stop_signal.emit()

        def on_mouse_click(x, y, button, pressed):
            if not self.recording:
                return True

            if button == MouseButton.left:
                if pressed:
                    self._drag_start = (x, y)
                    self._drag_mid = (x, y)
                else:
                    sx, sy = self._drag_start or (x, y)
                    dist = ((x - sx) ** 2 + (y - sy) ** 2) ** 0.5
                    if dist > DRAG_THRESH:
                        mx, my = self._drag_mid or ((sx + x) // 2, (sy + y) // 2)
                        self.record_signal.emit("drag", sx, sy, mx, my, x, y)
                    else:
                        # 单击 / 双击判定
                        now = time.time()
                        if self._last_click_ts and now - self._last_click_ts < DBL_THRESH:
                            self.record_signal.emit("dblclick", x, y, 0, 0, 0, 0)
                            self._last_click_ts = 0.0
                        else:
                            self.record_signal.emit("click", x, y, 0, 0, 0, 0)
                            self._last_click_ts = now
                    self._drag_start = None

            elif button == MouseButton.right:
                if pressed:
                    self.record_signal.emit("right", x, y, 0, 0, 0, 0)
            elif button == MouseButton.middle:
                if pressed:
                    self.record_signal.emit("middle", x, y, 0, 0, 0, 0)
            return True

        def on_mouse_move(x, y):
            if self.recording and self._drag_start:
                self._drag_mid = (x, y)
            return True

        kb = GlobalHotKeys({
            '<f4>': on_f4, '<f5>': on_f5, '<f7>': on_f7, '<f8>': on_f8,
        })
        ms = MouseListener(on_click=on_mouse_click, on_move=on_mouse_move)
        self._kb_listener = kb
        self._ms_listener = ms
        ms.start()
        kb.run()

    def stop_listener(self):
        if self._kb_listener is not None:
            self._kb_listener.stop()
        if self._ms_listener is not None:
            self._ms_listener.stop()


class TaskRow(QFrame):
    def __init__(self, parent_layout, delete_callback):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)

        # 操作类型选择
        self.type_combo = QComboBox()
        items = list(CMD_TYPES.keys())
        self.type_combo.addItems(items)
        self.type_combo.currentTextChanged.connect(self.on_type_changed)
        self.layout.addWidget(self.type_combo)

        # 参数标签
        self.param_label = QLabel("图片路径")
        self.param_label.setStyleSheet("color: #666; font-size: 12px;")
        self.param_label.setFixedWidth(55)
        self.layout.addWidget(self.param_label)

        # 参数输入区域
        self.value_input = QLineEdit()
        self.layout.addWidget(self.value_input)

        # 文件选择按钮
        self.file_btn = QPushButton("选择图片")
        self.file_btn.clicked.connect(self.select_file)
        self.layout.addWidget(self.file_btn)

        # 重试次数 + 说明
        self.retry_label = QLabel("重试:")
        self.retry_label.setStyleSheet("color: #888; font-size: 11px;")
        self.retry_label.setFixedWidth(30)
        self.layout.addWidget(self.retry_label)
        self.retry_input = QLineEdit("1")
        self.retry_input.setFixedWidth(35)
        self.layout.addWidget(self.retry_input)
        self.retry_hint = QLabel("1=一次, -1=无限")
        self.retry_hint.setStyleSheet("color: #aaa; font-size: 10px;")
        self.layout.addWidget(self.retry_hint)

        # 删除按钮
        self.del_btn = QPushButton("X")
        self.del_btn.setStyleSheet("color: red; font-weight: bold;")
        self.del_btn.setFixedWidth(30)
        self.del_btn.clicked.connect(lambda: delete_callback(self))
        self.layout.addWidget(self.del_btn)

        parent_layout.addWidget(self)
        # 初始化为第一个类型
        self.on_type_changed(items[0])

    def on_type_changed(self, text):
        cmd_type = CMD_TYPES[text]

        # 参数标签 & placeholder
        self.param_label.setText(
            _TYPE_PARAM_LABEL.get(cmd_type, "参数"))
        self.value_input.setPlaceholderText(
            _TYPE_PLACEHOLDER.get(cmd_type, ""))

        # 图片类：显示选择按钮 + 重试
        is_image = cmd_type in _IMG_TYPES
        is_screenshot = cmd_type == 9.0
        show_file_btn = is_image or is_screenshot
        show_retry = is_image

        self.file_btn.setVisible(show_file_btn)
        if is_screenshot:
            self.file_btn.setText("选择文件夹")
        else:
            self.file_btn.setText("选择图片")

        self.retry_label.setVisible(show_retry)
        self.retry_input.setVisible(show_retry)
        self.retry_hint.setVisible(show_retry)

    def set_data(self, data):
        cmd_type = data.get("type")
        value = data.get("value", "")
        retry = data.get("retry", 1)
        if cmd_type in CMD_TYPES_REV:
            self.type_combo.setCurrentText(CMD_TYPES_REV[cmd_type])
        self.value_input.setText(str(value))
        self.retry_input.setText(str(retry))

    def select_file(self):
        cmd_type = CMD_TYPES[self.type_combo.currentText()]
        if cmd_type == 9.0:
            folder = QFileDialog.getExistingDirectory(
                self, "选择保存文件夹", os.getcwd())
            if folder:
                self.value_input.setText(folder)
        else:
            filename, _ = QFileDialog.getOpenFileName(
                self, "选择图片", os.getcwd(),
                "Image Files (*.png *.jpg *.bmp)")
            if filename:
                self.value_input.setText(filename)

    def get_data(self):
        cmd_type = CMD_TYPES[self.type_combo.currentText()]
        value = self.value_input.text()
        try:
            if cmd_type in (5.0, 6.0):
                if not value:
                    value = "0"
            retry = 1
            if self.retry_input.isVisible():
                retry_text = self.retry_input.text()
                if retry_text:
                    retry = int(retry_text)
        except ValueError:
            pass
        return {"type": cmd_type, "value": value, "retry": retry}

class SettingsDialog(QDialog):
    """Cmd+, 设置窗口。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setFixedSize(300, 150)
        layout = QVBoxLayout(self)

        log_btn = QPushButton("查看日志")
        log_btn.clicked.connect(self._open_log_viewer)
        layout.addWidget(log_btn)

        layout.addStretch()

    def _open_log_viewer(self):
        viewer = LogViewer(self)
        viewer.exec()


class LogViewer(QDialog):
    """日志查看窗口。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("运行日志")
        self.resize(700, 500)
        layout = QVBoxLayout(self)

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setStyleSheet("font-family: Menlo, monospace; font-size: 12px;")
        layout.addWidget(self.text)

        btn_layout = QHBoxLayout()
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self._load_log)
        btn_layout.addWidget(refresh_btn)

        clear_btn = QPushButton("清空日志文件")
        clear_btn.clicked.connect(self._clear_log)
        btn_layout.addWidget(clear_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        self._load_log()

    def _load_log(self):
        try:
            if os.path.exists(_log_file):
                with open(_log_file, 'r', encoding='utf-8') as f:
                    self.text.setPlainText(f.read())
            else:
                self.text.setPlainText("（暂无日志）")
        except Exception as e:
            self.text.setPlainText(f"读取失败: {e}")

    def _clear_log(self):
        try:
            open(_log_file, 'w').close()
            self.text.setPlainText("（已清空）")
        except Exception as e:
            self.text.setPlainText(f"清空失败: {e}")


class RPAWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("不高兴就喝水 RPA 配置工具")
        self.resize(800, 600)

        self.engine = RPAEngine()
        self.worker = None
        self.rows = []

        # 全局热键
        self._rec_action_map = {
            "click": (12.0, "坐标左键单击"), "dblclick": (13.0, "坐标左键双击"),
            "right": (14.0, "坐标右键单击"), "middle": (15.0, "坐标中键单击"),
            "hover": (16.0, "坐标鼠标悬停"), "drag": (11.0, "坐标左键拖移"),
        }
        self.hotkey_thread = HotkeyThread()
        self.hotkey_thread.start_signal.connect(self.start_task)
        self.hotkey_thread.stop_signal.connect(self.stop_task)
        self.hotkey_thread.record_signal.connect(self._on_record)
        self.hotkey_thread.recording_toggled.connect(self._on_recording_toggled)
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

        self.clear_btn = QPushButton("清空指令")
        self.clear_btn.clicked.connect(self.clear_rows)
        top_bar.addWidget(self.clear_btn)

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
        self.hotkey_hint = QLabel("F4录制 | F7启动 | F8停止")
        self.hotkey_hint.setStyleSheet("color: #888; font-size: 11px; padding: 2px 8px;")
        top_bar.addWidget(self.hotkey_hint)

        main_layout.addLayout(top_bar)

        # 任务列表区域 (滚动)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.task_container = QWidget()
        self.task_layout = QVBoxLayout(self.task_container)
        self.task_layout.addStretch()
        scroll.setWidget(self.task_container)
        main_layout.addWidget(scroll)

        # 日志区域（默认隐藏）
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(150)
        self.log_area.setVisible(False)
        self.log_label = QLabel("运行日志:")
        self.log_label.setVisible(False)
        main_layout.addWidget(self.log_label)
        main_layout.addWidget(self.log_area)

        # Cmd+, 设置快捷键
        from PySide6.QtGui import QAction, QKeySequence
        pref_action = QAction("设置...", self)
        pref_action.setShortcut(QKeySequence.Preferences)
        pref_action.triggered.connect(self._open_settings)
        self.addAction(pref_action)

        # 初始添加一行
        self.add_row()

    def _on_record(self, action, x, y, x2, y2, x3, y3):
        if action not in self._rec_action_map:
            return
        cmd_type, label = self._rec_action_map[action]
        if action == "drag":
            value = f"{x},{y} -> {x2},{y2} -> {x3},{y3}"
        else:
            value = f"{x},{y}"
        self.add_row({"type": cmd_type, "value": value, "retry": 1})
        _logger.info(f"录制 {label} → {value}")

    def _on_recording_toggled(self, entering):
        if entering:
            empty = [r for r in self.rows
                     if not r.get_data().get("value", "").strip()]
            for r in empty:
                self.rows.remove(r)
                r.deleteLater()
            self.hotkey_hint.setText("● 录制 F4退出|F5悬停|鼠标操作自动记录")
            self.hotkey_hint.setStyleSheet(
                "color: #f44336; font-size: 11px; padding: 2px 8px;")
            self.hide()
        else:
            self.hotkey_hint.setText("F4录制 | F7启动 | F8停止")
            self.hotkey_hint.setStyleSheet(
                "color: #888; font-size: 11px; padding: 2px 8px;")
            self.show()
            self.activateWindow()

    def _open_settings(self):
        dlg = SettingsDialog(self)
        dlg.exec()

    def add_row(self, data=None):
        self.task_layout.takeAt(self.task_layout.count() - 1)
        row = TaskRow(self.task_layout, self.delete_row)
        if data:
            row.set_data(data)
        self.rows.append(row)
        self.task_layout.addStretch()

    def delete_row(self, row_widget):
        if row_widget in self.rows:
            self.rows.remove(row_widget)
            row_widget.deleteLater()

    def clear_rows(self):
        for row in self.rows:
            row.deleteLater()
        self.rows.clear()
        self.add_row()

    def save_config(self):
        tasks = []
        for row in self.rows:
            data = row.get_data()
            tasks.append(data)

        if not tasks:
            QMessageBox.warning(self, "警告", "没有可保存的配置")
            return

        from PySide6.QtWidgets import QInputDialog

        profiles_dir = os.path.join(_PROJECT_DIR, "profiles")
        images_dir = os.path.join(profiles_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        name, ok = QInputDialog.getText(
            self, "保存配置", "配置名称（不含扩展名）:")
        if not ok or not name.strip():
            return
        name = "".join(c for c in name.strip() if c not in r'\/:*?"<>|')

        for task in tasks:
            cmd_type = task.get("type")
            value = task.get("value", "")
            if cmd_type not in _IMG_TYPES or not value:
                continue
            src = os.path.abspath(value)
            if not os.path.isfile(src):
                continue
            if os.path.commonpath([src, os.path.abspath(images_dir)]) == os.path.abspath(images_dir):
                task["value"] = os.path.join("images", os.path.basename(src))
                continue

            basename = os.path.basename(src)
            dest = os.path.join(images_dir, basename)
            if os.path.exists(dest) and not os.path.samefile(src, dest):
                stem, ext = os.path.splitext(basename)
                i = 2
                while os.path.exists(dest):
                    dest = os.path.join(images_dir, f"{stem}_{i}{ext}")
                    i += 1

            shutil.move(src, dest)
            task["value"] = os.path.join("images", os.path.basename(dest))

        filepath = os.path.join(profiles_dir, f"{name}.json")
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(tasks, f, indent=4, ensure_ascii=False)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {e}")

    def load_config(self):
        profiles_dir = os.path.join(_PROJECT_DIR, "profiles")
        filename, _ = QFileDialog.getOpenFileName(
            self, "导入配置", profiles_dir, "JSON Files (*.json)")
        if not filename:
            return

        try:
            with open(filename, 'r', encoding='utf-8') as f:
                tasks = json.load(f)

            if not isinstance(tasks, list):
                raise ValueError("文件格式不正确")

            config_dir = os.path.dirname(os.path.abspath(filename))
            for task in tasks:
                value = task.get("value", "")
                if value and not os.path.isabs(value) and not _is_coordinate(value):
                    task["value"] = os.path.normpath(
                        os.path.join(config_dir, value))

            for row in self.rows:
                row.deleteLater()
            self.rows.clear()

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

        self.showMinimized()

    def stop_task(self):
        self.engine.stop()
        self.log("正在停止...")

    def on_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.add_btn.setEnabled(True)
        self.log("任务已结束")
        self.showNormal()
        self.activateWindow()

    def log(self, msg):
        self.log_area.append(msg)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.engine.stop()
            self.worker.quit()
            self.worker.wait()
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
