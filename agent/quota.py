"""M10: 基于 IP 的配额管理器"""

import json
import threading
from datetime import datetime, timedelta
from pathlib import Path


class QuotaManager:
    """基于 IP 的配额管理器
    
    功能:
    - 每用户每天限制 N 个任务
    - 白名单 IP 不限制
    - JSON 文件存储，7 天自动清理
    """

    def __init__(self, data_dir: str, daily_limit: int, whitelist: set[str]):
        self.data_file = Path(data_dir) / "quota_usage.json"
        self.daily_limit = daily_limit
        self.whitelist = whitelist
        self._lock = threading.Lock()
        
        # 确保数据目录存在
        self.data_file.parent.mkdir(parents=True, exist_ok=True)

    def check_quota(self, client_ip: str) -> tuple[bool, dict]:
        """
        检查配额
        
        Returns:
            (allowed, info)
            - allowed: 是否允许提交
            - info: {"used": 3, "limit": 5, "remaining": 2, "is_whitelisted": False}
        """
        # 白名单用户不限制
        if client_ip in self.whitelist:
            return True, {
                "used": 0,
                "limit": -1,
                "remaining": -1,
                "is_whitelisted": True,
            }
        
        with self._lock:
            data = self._load_data()
            today = datetime.now().strftime("%Y-%m-%d")
            
            # 获取今日该 IP 的使用情况
            today_data = data.get(today, {})
            ip_data = today_data.get(client_ip, {})
            used = ip_data.get("count", 0)
            
            remaining = max(0, self.daily_limit - used)
            allowed = used < self.daily_limit
            
            return allowed, {
                "used": used,
                "limit": self.daily_limit,
                "remaining": remaining,
                "is_whitelisted": False,
            }

    def record_usage(self, client_ip: str, task_id: str):
        """记录一次任务使用"""
        # 白名单用户不记录
        if client_ip in self.whitelist:
            return
        
        with self._lock:
            data = self._load_data()
            today = datetime.now().strftime("%Y-%m-%d")
            
            # 初始化今日数据
            if today not in data:
                data[today] = {}
            
            # 初始化该 IP 的数据
            if client_ip not in data[today]:
                data[today][client_ip] = {
                    "count": 0,
                    "tasks": [],
                    "last_used": None,
                }
            
            # 更新计数
            data[today][client_ip]["count"] += 1
            data[today][client_ip]["tasks"].append(task_id)
            data[today][client_ip]["last_used"] = datetime.now().isoformat()
            
            # 清理旧数据
            self._cleanup_old_data(data)
            
            # 保存
            self._save_data(data)

    def get_usage_stats(self, client_ip: str) -> dict:
        """获取当前用户的配额使用情况"""
        # 白名单用户
        if client_ip in self.whitelist:
            return {
                "used": 0,
                "limit": -1,
                "remaining": -1,
                "is_whitelisted": True,
            }
        
        with self._lock:
            data = self._load_data()
            today = datetime.now().strftime("%Y-%m-%d")
            
            today_data = data.get(today, {})
            ip_data = today_data.get(client_ip, {})
            used = ip_data.get("count", 0)
            
            remaining = max(0, self.daily_limit - used)
            
            return {
                "used": used,
                "limit": self.daily_limit,
                "remaining": remaining,
                "is_whitelisted": False,
            }

    def _cleanup_old_data(self, data: dict = None):
        """清理 7 天前的数据"""
        if data is None:
            with self._lock:
                data = self._load_data()
                self._cleanup_old_data(data)
                self._save_data(data)
            return
        
        cutoff = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        dates_to_remove = [d for d in data.keys() if d < cutoff]
        for d in dates_to_remove:
            del data[d]

    def _load_data(self) -> dict:
        """从 JSON 文件加载数据"""
        if not self.data_file.exists():
            return {}
        
        try:
            with open(self.data_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def _save_data(self, data: dict):
        """保存数据到 JSON 文件"""
        with open(self.data_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
