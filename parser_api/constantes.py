# Raio médio da Terra em metros, usado na distância de Haversine.
EARTH_RADIUS_M: float = 6_371_000.0

# Conjunto de valores de 'highway' considerados dirigíveis.
DRIVE_HIGHWAYS = {
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential", "living_street", "service",
    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
}

# Conjuntos específicos para pedestres e ciclovias.
PEDESTRIAN_SET = {"pedestrian", "footway", "steps"}
CYCLE_SET = {"cycleway"}

# Endpoint padrão da Overpass API (pode ser trocado por espelhos).
DEFAULT_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
