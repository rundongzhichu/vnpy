"""
Global setting of the trading platform.
"""

from logging import INFO
from tzlocal import get_localzone_name

from .utility import load_json


SETTINGS: dict = {
    "font.family": "微软雅黑",
    "font.size": 12,

    "log.active": True,
    "log.level": INFO,
    "log.console": True,
    "log.file": True,

    "email.server": "smtp.qq.com",
    "email.port": 465,
    "email.username": "",
    "email.password": "",
    "email.sender": "",
    "email.receiver": "",

    # rqdata
    # "datafeed.name": "rqdata",
    # "datafeed.username": "license",
    # "datafeed.password": "GpYr7c0DJbnD3yBZ7XEX9CihZI0zYLW07hVHKfZP1Uvz-M2UKPdVyS10IQ-0_b8PvrKShMwTfbyLLb1K0JK1aRR84EDBB-ezyCmH-h9eQGSvmpuyRGmK8iuah-zt-U_ZmKcDCbhdMw18mLsis-CzfTLXT99SNaG0Ars8Sa_e9r0=VisWwnAEqprczOQhvgNhwLFZ7Kp2a1fZELOmsBNVy1Pf0gES6fTGJBEi2bHXVk3RQsQQ-ghuFI8Cp7M2XxcKbx5LRRYlH5shOWosjRkrmLhLlKPM3PP4eWbmOBz_IkEh83lcZc5H2EaiYKFmeEtDe9fDjycT9ACRhyHn33WPC0M=",

    # 天勤
    "datafeed.name": "tqsdk",
    "datafeed.username": "shoucai",
    "datafeed.password": "Wssc1314520",


    "database.timezone": get_localzone_name(),
    "database.name": "sqlite",
    "database.database": "database.db",
    "database.host": "",
    "database.port": 0,
    "database.user": "",
    "database.password": ""
}


# Load global setting from json file.
SETTING_FILENAME: str = "vt_setting.json"
SETTINGS.update(load_json(SETTING_FILENAME))
