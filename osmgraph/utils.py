import math
from typing import Iterable, List

EARTH_RADIUS_M = 6_371_000.0

def compute_haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlmb = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) * 2) + (math.cos(phi1) * math.cos(phi2) * (math.sin(dlmb / 2) * 2))
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(a))

def dedupe_consecutive(ids_iter: Iterable[int]) -> List[int]:
    result: List[int] = []
    last: int | None = None
    for nid in ids_iter:
        if nid != last:
            result.append(nid)
            last = nid
    return result 