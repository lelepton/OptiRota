# Constantes: tipos de via por highway
CAR_TAGS = {
    "primary", "primary_link",
    "secondary", "secondary_link",
    "tertiary", "tertiary_link",
    "residential", "service",
}

BIKE_TAGS = CAR_TAGS | {"living_street", "cycleway", "path"}
# Observação: pedestre (foot) inclui todas as vias, mantemos allow_foot=True.


# Velocidades por tipo de via (m/s) usadas na estimativa de tempo por aresta
SPEED_LIMITS_MPS = {
    "primary":     22.22,  # ~80 km/h
    "secondary":   16.67,  # ~60 km/h
    "tertiary":    11.11,  # ~40 km/h
    "residential":  8.33,  # ~30 km/h
}


# Coeficientes (heurística de "tempo" artificial) por tipo de via base
COEF_BY_TAG = {
    "primary": 0.8,
    "secondary": 0.6,
    "tertiary": 0.4,
    "residential": 0.3,
}
