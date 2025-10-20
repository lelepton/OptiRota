from dataclasses import dataclass

@dataclass
class Delivery:
    '''Estrutura que representa uma entrega com janelas de tempo diárias e peso.'''
    delivery_identifier: int
    delivery_latitude: float
    delivery_longitude: float
    delivery_weight_tons: float
    delivery_time_windows_seconds: list[tuple[int, int]]
    delivery_was_completed: bool = False

    @staticmethod
    def _time_to_seconds(time_hhmm: str) -> int:
        '''
        Converte uma string de horário "HH:MM" para o número de segundos desde 00:00.

        Parâmetros
        ----------
        time_hhmm : str

        Retorno
        -------
        int : segundos no dia (0–86399)
        '''
        hh, mm = [int(x) for x in time_hhmm.strip().split(":")]
        return hh * 3600 + mm * 60

    @classmethod
    def from_windows_string(cls, idx: int, lat: float, lon: float,
                            windows_string: str, weight_tons: float) -> "Delivery":
        '''
        Constrói um objeto Delivery a partir de uma string de janelas no formato:
        "HH:MM-HH:MM, HH:MM-HH:MM, ..." e de um peso em toneladas.
        As janelas são convertidas para segundos e janelas sobrepostas são mescladas.

        Parâmetros
        ----------
        idx            : int             (identificador)
        lat, lon       : float           (coordenadas do cliente)
        windows_string : str             (janelas em string)
        weight_tons    : float           (peso da entrega)

        Retorno
        -------
        Delivery : instância preenchida
        '''

        pairs: list[tuple[int, int]] = []
        for seg in [s.strip() for s in windows_string.split(",") if s.strip()]:
            a, b = [p.strip() for p in seg.split("-")]
            s, e = cls._time_to_seconds(a), cls._time_to_seconds(b)
            if e < s:
                s, e = e, s
            pairs.append((s, e))
        pairs.sort()
        # mesclar sobreposições
        merged: list[tuple[int, int]] = []
        for s, e in pairs:
            if not merged or s > merged[-1][1]:
                merged.append((s, e))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        return cls(idx, float(lat), float(lon), float(weight_tons), merged)

    def earliest_same_day_service_time(self, arrival_seconds_of_day: float) -> int | None:
        '''
        Dado um horário de chegada (em segundos do dia), retorna o primeiro instante
        **no mesmo dia** em que a entrega pode ser atendida.

        Parâmetros
        ----------
        arrival_seconds_of_day : float

        Retorno
        -------
        int | None : segundos do dia do atendimento; None se todas as janelas de hoje
                     já passaram para esse horário de chegada.
        '''
        best: int | None = None
        for s, e in self.delivery_time_windows_seconds:
            if arrival_seconds_of_day > e:
                continue
            candidate = int(arrival_seconds_of_day) if arrival_seconds_of_day >= s else s
            if best is None or candidate < best:
                best = candidate
        return best
