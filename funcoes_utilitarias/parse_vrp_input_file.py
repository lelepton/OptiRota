import re
import shlex

def parse_vrp_input_file(file_path: str) -> tuple[float, float, list[float], list[list]]:
    '''
    Lê um arquivo texto no formato:
    <depot_lat> <depot_lon> <capacity_tons>
    
    <lat> <lon> "HH:MM-HH:MM, HH:MM-HH:MM" <weight_tons>
    ...
    Retorna tupla com origem, capacidade e lista de entregas cruas.

    Parâmetros
    ----------
    file_path : str (caminho do arquivo)

    Retorno
    -------
    (float, float, float, list[list]) :
    (origin_lat, origin_lon, capacity_tons, deliveries_raw)
    onde deliveries_raw = [[lat, lon, weight_tons, windows_string], ...]

    Observações
    -----------
    - Não há validação/erros; assume-se arquivo bem formatado.
    '''

    # Parser robusto (shlex) para o arquivo VRP:
    #   <lat> <lon> <num_veiculos>
    #   <cap1> <cap2> ... <capN>
    #   <lat> <lon> "A-B, C-D" <peso>
    # - Aceita janelas com ou sem aspas; se sem aspas e com espaços, reconstrói juntando tokens intermediários.
    # - Tolera lat,lon separados por espaço ou vírgula.
    # - Mensagens de erro incluem a linha problemática.

    with open(file_path, "r", encoding="utf-8") as f:
        raw_lines = [ln.rstrip("\n") for ln in f]

    # Remove vazias/puras de comentário
    lines = [ln.strip() for ln in raw_lines if ln.strip() and not ln.strip().startswith("#")]

    if len(lines) < 2:
        raise ValueError("Arquivo VRP precisa de ao menos 2 linhas (origem+veículos e capacidades).")

    # Linha 1: lat lon num_veiculos (permite vírgula depois da lat)
    m0 = re.match(r'^([\-\d.]+)\s*,?\s+([\-\d.]+)\s+(\d+)\s*$', lines[0])
    if not m0:
        raise ValueError(f"Linha 1 inválida: {lines[0]!r}. Esperado: <lat> <lon> <num_veiculos>")
    origin_lat, origin_lon, num_vehicles = float(m0.group(1)), float(m0.group(2)), int(m0.group(3))

    # Linha 2: capacidades (separadas por espaço OU vírgula)
    caps_tokens = re.split(r'[,\s]+', lines[1].strip())
    capacities_tons = [float(tok) for tok in caps_tokens if tok]
    if len(capacities_tons) < num_vehicles:
        # usa quantas existirem; evita index error; quem quiser N exato, forneça N tokens
        num_vehicles = len(capacities_tons)
    capacities_tons = capacities_tons[:num_vehicles]

    deliveries_raw: list[list] = []
    for idx, line in enumerate(lines[2:], start=3):
        try:
            parts = shlex.split(line)
            if len(parts) < 4:
                # Fallback tolerante: lat lon [windows string] weight
                m = re.match(r'^\s*([\-\d.]+)\s*,?\s+([\-\d.]+)\s+(.*?)\s+([\-\d.]+)\s*$', line)
                if not m:
                    raise ValueError("Formato não reconhecido")
                lat = float(m.group(1)); lon = float(m.group(2))
                weight_tons = float(m.group(4))
                windows_string = m.group(3).strip()
            else:
                lat = float(parts[0]); lon = float(parts[1])
                weight_tons = float(parts[-1])
                windows_string = " ".join(parts[2:-1]).strip()

            # Normaliza aspas tipográficas/dobradas
            if windows_string and windows_string[0] == windows_string[-1] and windows_string[0] in ('\"','"', '“', '”'):
                windows_string = windows_string[1:-1].strip()
            deliveries_raw.append([lat, lon, weight_tons, windows_string])
        except Exception as exc:
            raise ValueError(f"Linha {idx} inválida: {line!r} | erro: {exc}") from exc

    return origin_lat, origin_lon, capacities_tons, deliveries_raw

