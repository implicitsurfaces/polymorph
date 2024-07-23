import json
from pathlib import Path

ICON_DIR = Path(__file__).parent.parent.parent.joinpath("fonts")
ICON_PATH = str(ICON_DIR.joinpath("lucide.ttf"))

ICON_DIR = Path(__file__).parent.parent.parent.joinpath("fonts")
ICON_MAP = {
    icon: chr(int(description["unicode"].strip("&#;")))
    for icon, description in json.load(
        ICON_DIR.joinpath("icon_map.json").open()
    ).items()
}

ICON_MIN_VALUE = min(ord(v) for v in ICON_MAP.values())
ICON_MAX_VALUE = max(ord(v) for v in ICON_MAP.values())


class IconRetriever:
    def __getattr__(self, name):
        return ICON_MAP[name.lower().replace("_", "-")]


# You can search for the icons here: https://lucide.dev/icons/
Icon = IconRetriever()
