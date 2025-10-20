from typing import Optional
from classes_de_elementos.road_graph import RoadGraph
from classes_de_elementos.edge import Edge
from classes_de_elementos.delivery import Delivery
from funcoes_utilitarias._split_minutes_seconds_exact import _split_minutes_seconds_exact
from funcoes_utilitarias._time_to_seconds import _time_to_seconds
from funcoes_utilitarias._seconds_to_hhmmss import _seconds_to_hhmmss
from funcoes_utilitarias._fmt_seconds import _fmt_seconds

def VRP_heuristic(
    graph: "RoadGraph",
    origin_lat: float,
    origin_lon: float,
    start_time_HH_MM: str,
    capacities_tons_L: list[float],
    deliveries_raw: list[list],
    highway: str = "car",
) -> list[tuple[int, list["Edge"]]]:
    '''
    Heurística gulosa com janelas diárias e vários veículos.
    Antes de empurrar entrega para “DIA N+1”, tenta alocar no MESMO dia
    para outro veículo (imprimindo “DIA N, VEÍCULO M” quando o veículo
    começa a operar no dia).

    Parâmetros
    ---------
    graph              : RoadGraph
    origin_lat         : float  (latitude do depósito)
    origin_lon         : float  (longitude do depósito)
    start_time_HH_MM   : str    (horário de partida “HH:MM”)
    capacities_tons_L  : list   (capacidade [t] de cada veículo)
    deliveries_raw     : list   ([[lat, lon, weight_tons, windows_str], ...])
    highway            : str    ('car' | 'bike' | 'foot')

    Retorno
    -------
    Retorna uma lista: [(veiculo_id_1based, lista_de_arestas_percorridas), ...]
    - veiculo_id_1based : int (1, 2, 3, ...)
    - lista_de_arestas_percorridas : List[Edge] concatenando, na ordem,
    as arestas dos deslocamentos realizados por aquele veículo.
    List[Tuple[int, List[Edge]]] : (veículo 1-based, arestas)
    '''

    # Regras de escolha
    # -----------------
    # 1) Entre as entregas atendíveis **agora** (sem espera), escolha a que tem
    #    menor horário de atendimento (empate: menor distância).
    # 2) Se não houver atendimento imediato, considere **voltar ao depósito**
    #    (somando o tempo de retorno) e planejar a partir de lá (aqui pode haver
    #    espera até a janela). Escolha a de menor horário de atendimento após o
    #    retorno (empate: menor distância).
    # 3) Se nada for possível HOJE (mesmo considerando o retorno), o caminhão
    #    retorna à base (somando o tempo), e avançamos para o próximo dia (DIA N+1),
    #    reiniciando o relógio em start_time_HH_MM e resetando a capacidade.
    #
    # Observações
    # -----------
    # - A capacidade NÃO entra no ranking; só impede “atendimento imediato” se não
    #   couber. Após retorno ao depósito, a capacidade é resetada.
    # - Assume-se que **não existem entregas que comecem em um dia e terminem em outro**.

    # 1) Construção das entregas
    deliveries: list[Delivery] = [
        Delivery.from_windows_string(idx, lat, lon, windows_string, weight_tons)
        for idx, (lat, lon, weight_tons, windows_string) in enumerate(deliveries_raw)
    ]

    # 2) Estado dos veículos
    num_vehicles = len(capacities_tons_L)
    vehicles = [{
        "lat": float(origin_lat),
        "lon": float(origin_lon),
        "t": float(_time_to_seconds(start_time_HH_MM)),
        "cap": float(capacities_tons_L[i]),
        "cap_max": float(capacities_tons_L[i]),
        "started": False,  # já imprimiu "DIA N, VEÍCULO M" no dia atual
    } for i in range(num_vehicles)]

    # 2.1) Acumulador das arestas percorridas por veículo
    vehicles_paths: list[list[Edge]] = [[] for _ in range(num_vehicles)]

    current_day = 1
    print(f"DIA {current_day}")
    delivered_today = False
    current_block_vi: int | None = None  # veículo cujo bloco está “aberto”

    def best_candidate_for_vehicle(vstate: dict) -> tuple[Optional[dict], Optional[str]]:
        '''
        Função interna: melhor candidato para UM veículo
        Retorna um dict com:
          - delivery
          - distance_meters
          - travel_time_seconds
          - service_time_seconds
          - edges              (se “direct”)
          - edges_return       (se “via depósito”)
          - edges_to_delivery  (se “via depósito”)
        e também o tipo: "direct" | "depot"
        '''

        cur_lat = vstate["lat"]; cur_lon = vstate["lon"]; cur_t = vstate["t"]; cur_cap = vstate["cap"]

        direct_candidates: list[dict] = []
        depot_candidates:  list[dict] = []

        # Caminho de retorno atual->depósito (para calcular tempo de volta quando “via depósito”)
        edges_back, _dist_back, ret_back_s = graph.astar_with_time_between(
            cur_lat, cur_lon, origin_lat, origin_lon, highway
        )

        for delivery in deliveries:
            if delivery.delivery_was_completed:
                continue

            # A) Direto (sem voltar ao depósito)
            if delivery.delivery_weight_tons <= cur_cap:
                edges_a, dist_a, trav_a = graph.astar_with_time_between(
                    cur_lat, cur_lon, delivery.delivery_latitude, delivery.delivery_longitude, highway
                )
                svc_a = delivery.earliest_same_day_service_time(cur_t + trav_a)
                if svc_a is not None:
                    direct_candidates.append({
                        "delivery": delivery,
                        "distance_meters": float(dist_a),
                        "travel_time_seconds": float(trav_a),
                        "service_time_seconds": float(svc_a),
                        "edges": edges_a,
                    })

            # B) Via depósito (reset de capacidade)
            edges_b, dist_b, trav_b = graph.astar_with_time_between(
                origin_lat, origin_lon, delivery.delivery_latitude, delivery.delivery_longitude, highway
            )
            svc_b = delivery.earliest_same_day_service_time(cur_t + ret_back_s + trav_b)
            if svc_b is not None and delivery.delivery_weight_tons <= vstate["cap_max"]:
                depot_candidates.append({
                    "delivery": delivery,
                    "distance_meters": float(dist_b),
                    "travel_time_seconds": float(trav_b),
                    "service_time_seconds": float(svc_b),
                    "edges_return": edges_back,
                    "edges_to_delivery": edges_b,
                })

        def _rank(c: dict) -> tuple:
            return (c["service_time_seconds"], c["distance_meters"], c["delivery"].delivery_identifier)

        if direct_candidates:
            direct_candidates.sort(key=_rank)
            return direct_candidates[0], "direct"
        if depot_candidates:
            depot_candidates.sort(key=_rank)
            return depot_candidates[0], "depot"
        return None, None

    # Loop principal (dias)
    while True:
        pending = [d for d in deliveries if not d.delivery_was_completed]
        if not pending:
            # Fechar bloco ativo (se houver), antes de encerrar
            if current_block_vi is not None:
                print("Carga retorna a distribuidora.")
                current_block_vi = None
            # Fim → retorna estrutura solicitada
            return [(vi + 1, vehicles_paths[vi]) for vi in range(num_vehicles)]

        # Melhor atendimento entre todos os veículos
        best = None; best_vi = None; best_kind = None
        for vi, v in enumerate(vehicles):
            cand, kind = best_candidate_for_vehicle(v)
            if cand is None:
                continue
            key = (cand["service_time_seconds"], cand["distance_meters"], cand["delivery"].delivery_identifier)
            if best is None or key < (best["service_time_seconds"], best["distance_meters"], best["delivery"].delivery_identifier): # pylint: disable=unsubscriptable-object
                best, best_vi, best_kind = cand, vi, kind

        if best is not None:
            # Troca de bloco de veículo: fecha o anterior
            if current_block_vi is None:
                current_block_vi = best_vi
            elif current_block_vi != best_vi:
                print("Carga retorna a distribuidora.")
                current_block_vi = best_vi

            v = vehicles[best_vi]
            d = best["delivery"]

            # Cabeçalho do bloco do veículo no dia
            if not v["started"]:
                print(f"DIA {current_day}, VEÍCULO {best_vi + 1}")
                v["started"] = True

            # Caso A: direto
            if best_kind == "direct":
                mins, rem = _split_minutes_seconds_exact(best["travel_time_seconds"])
                print(
                    f"Carga sai ({v['lat']:.6f}, {v['lon']:.6f}) {_seconds_to_hhmmss(v['t'])} "
                    f"em direção a ({d.delivery_latitude:.6f}, {d.delivery_longitude:.6f}) "
                    f"e chegará em {mins} minutos {_fmt_seconds(rem)} segundos "
                    f"às {_seconds_to_hhmmss(best['service_time_seconds'])} horas."
                )
                # Acumula arestas do deslocamento direto
                vehicles_paths[best_vi].extend(best.get("edges", []))

                v["t"] = float(best["service_time_seconds"])
                v["lat"], v["lon"] = d.delivery_latitude, d.delivery_longitude
                v["cap"] -= d.delivery_weight_tons
                d.delivery_was_completed = True
                delivered_today = True
                continue

            # Caso B: via depósito (reset capacidade)
            # Acumula arestas: retorno e ida ao cliente
            vehicles_paths[best_vi].extend(best.get("edges_return", []))
            mins, rem = _split_minutes_seconds_exact(best["travel_time_seconds"])
            print(
                f"Carga sai da distribuidora {_seconds_to_hhmmss(v['t'])} "
                f"em direção a ({d.delivery_latitude:.6f}, {d.delivery_longitude:.6f}) "
                f"e chegará em {mins} minutos {_fmt_seconds(rem)} segundos "
                f"às {_seconds_to_hhmmss(best['service_time_seconds'])} horas."
            )
            vehicles_paths[best_vi].extend(best.get("edges_to_delivery", []))

            v["t"] = float(best["service_time_seconds"])
            v["lat"], v["lon"] = d.delivery_latitude, d.delivery_longitude
            v["cap"] -= d.delivery_weight_tons
            # reset de cap já é implícito ao voltar no cálculo; aqui seguimos a lógica original:
            # (se desejar “zerar” explicitamente na volta, mover esse reset para o ponto de retorno)
            d.delivery_was_completed = True
            delivered_today = True
            continue

        # Nenhum veículo atende algo hoje → checa inviabilidade e vira o dia
        if not delivered_today:
            possible_next_day = False
            day_start_seconds = float(_time_to_seconds(start_time_HH_MM))
            max_caps = [vv["cap_max"] for vv in vehicles] or [0.0]
            for dcheck in [d for d in deliveries if not d.delivery_was_completed]:
                if dcheck.delivery_weight_tons > max(max_caps):
                    print(f"Entrega {dcheck.delivery_identifier} inviável: peso maior que a capacidade de qualquer veículo.")
                    dcheck.delivery_was_completed = True
                    continue
                _eX, _dX, travX = graph.astar_with_time_between(
                    origin_lat, origin_lon, dcheck.delivery_latitude, dcheck.delivery_longitude, highway
                )
                svcX = dcheck.earliest_same_day_service_time(day_start_seconds + travX)
                if svcX is not None:
                    possible_next_day = True
                    break
            if not possible_next_day:
                if current_block_vi is not None:
                    print("Carga retorna a distribuidora.")
                    current_block_vi = None
                print("Nenhuma entrega restante é atendível em dias futuros (por capacidade/janelas/alcance). Encerrando.")
                return [(vi + 1, vehicles_paths[vi]) for vi in range(num_vehicles)]

        # Fechar bloco ativo antes de virar o dia
        if current_block_vi is not None:
            print("Carga retorna a distribuidora.")
            current_block_vi = None

        # Novo dia: reseta posição/tempo/capacidade de todos
        current_day += 1
        print(f"DIA {current_day}")
        delivered_today = False
        for v in vehicles:
            v["lat"], v["lon"] = float(origin_lat), float(origin_lon)
            v["t"] = float(_time_to_seconds(start_time_HH_MM))
            v["cap"] = float(v["cap_max"])
            v["started"] = False
