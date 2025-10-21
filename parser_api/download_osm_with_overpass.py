import urllib.parse
import urllib.request
import logging
import time
from pathlib import Path
from .build_overpass_query import build_overpass_query
from .constantes import DEFAULT_OVERPASS_URL

def download_osm_with_overpass(
    lat: float,
    lon: float,
    radius_m: float,
    dest_osm: Path,
    overpass_url: str = DEFAULT_OVERPASS_URL,
    retries: int = 3,
    backoff_sec: float = 2.5,
) -> None:
    '''
    Executa POST na Overpass API com a consulta gerada e salva o XML em
    dest_osm. Implementa retry com backoff simples para falhas transitórias.
    
    Parâmetros
    ----------
    lat, lon     : float  -> centro do círculo (graus decimais)
    radius_m     : float  -> raio em METROS
    dest_osm     : Path   -> caminho do arquivo .osm a salvar
    overpass_url : str    -> endpoint da Overpass (opcional)
    retries      : int    -> tentativas em caso de falha (padrão: 3)
    backoff_sec  : float  -> fator de espera entre tentativas (padrão: 2.5)
    
    Retorno
    -------
    None
    
    Complexidade
    ------------
    O(Tamanho_da_resposta) para E/S; latência de rede variável.
    
    Observações
    -----------
    - Define User-Agent explícito e Content-Type padrão de formulários.
    - Em falha, registra warnings e levanta exceção após esgotar tentativas.
    '''

    dest_osm.parent.mkdir(parents=True, exist_ok=True)
    query = build_overpass_query(lat, lon, radius_m)
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")

    req = urllib.request.Request(
        overpass_url,
        data=data,
        method="POST",
        headers={
            "User-Agent": "osm-csv-downloader/1.0 (+https://example.org)",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        },
    )

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Overpass HTTP {resp.status}")
                content = resp.read()
                dest_osm.write_bytes(content)
                return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            logging.warning("Falha ao baixar (tentativa %d/%d): %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(backoff_sec * attempt)
            else:
                break

    raise RuntimeError(f"Não foi possível baixar dados da Overpass: {last_err}")
