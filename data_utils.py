from dataclasses import dataclass
from typing import List, Dict, Any
import json

@dataclass
class ClassItem:
    id: int
    name: str
    teacher: str
    group: str
    size: int
    room_type: str = "standard"

@dataclass
class Room:
    id: int
    name: str
    capacity: int
    room_type: str = "standard"

def load_from_dict(d: Dict[str, Any]):
    classes = [ClassItem(**c) for c in d.get("classes", [])]
    rooms = [Room(**r) for r in d.get("rooms", [])]
    timeslots = d.get("timeslots", [])
    return classes, rooms, timeslots

def load_from_json_file(path: str):
    with open(path, "r") as f:
        d = json.load(f)
    return load_from_dict(d)

def total_sessions_required(classes):
    # If classes list already expanded into sessions, just len(classes)
    return len(classes)

def build_per_day_period_map(timeslots):
    # returns ordered list of days and periods per day map
    day_order = []
    periods_per_day = {}
    for label in timeslots:
        if "_" in label:
            day, period = label.split("_",1)
        elif "-" in label:
            day, period = label.split("-",1)
        else:
            parts=label.split(" ",1)
            day = parts[0]
            period = parts[1] if len(parts)>1 else ""
        if day not in day_order:
            day_order.append(day)
        periods_per_day.setdefault(day, []).append(period)
    # convert to counts
    periods_count = {d: len(periods_per_day[d]) for d in day_order}
    return day_order, periods_count
