
from pathlib import Path
from typing import Dict, List, Tuple, Iterable
import xml.etree.ElementTree as ET

DRIVE_HIGHWAYS = {
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential", "living_street", "service",
    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
}
PEDESTRIAN_SET = {"pedestrian", "footway", "steps"}
CYCLE_SET = {"cycleway"}

def parse_nodes(osm_path: Path) -> Dict[int, Tuple[float, float]]:
    nodes: Dict[int, Tuple[float, float]] = {}
    for _, elem in ET.iterparse(str(osm_path), events=("end",)):
        if elem.tag == "node":
            try:
                osmid = int(elem.attrib["id"])
                lat = float(elem.attrib["lat"])
                lon = float(elem.attrib["lon"])
                nodes[osmid] = (lat, lon)
            except (KeyError, ValueError):
                continue
            finally:
                elem.clear()
    return nodes

def iter_ways(osm_path: Path) -> Iterable[Tuple[List[int], Dict[str, str]]]:
    for _, elem in ET.iterparse(str(osm_path), events=("end",)):
        if elem.tag == "way":
            node_ids = [int(nd.attrib["ref"]) for nd in elem.findall("nd")]
            tags = {t.attrib["k"]: t.attrib.get("v", "") for t in elem.findall("tag")}
            yield node_ids, tags
            elem.clear()

def is_relevant_way(tags: Dict[str, str]) -> bool:
    highway = tags.get("highway")
    if highway is None:
        return False
    if tags.get("area") == "yes":
        return False
    if highway == "construction":
        return False
    if tags.get("access") in {"private", "no"}:
        return False
    if tags.get("vehicle") == "no" and tags.get("bicycle") == "no" and tags.get("foot") == "no":
        return False
    return (
        (highway in DRIVE_HIGHWAYS)
        or (highway in PEDESTRIAN_SET)
        or (highway in CYCLE_SET)
        or (highway == "path")
    )