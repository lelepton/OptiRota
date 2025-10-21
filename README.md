## Definição

O OptiRota é um sofisticado sistema de otimização de rotas. Sua implementação foi feita em multicamadas, começando pela análise de dados geoespaciais do mundo real, e se estendendo para a construção de um grafo navegável, a implementação e análise de algoritmos de busca fundamentais (Dijikstra e A*) e, finalmente, a aplicação da heurística para resolver uma versão com restrições do Problema de Roteamento de Veículos (VRP).


Este repositório reúne quatro utilitários CLI do OptiRota para **gerar grafos a partir do OSM**, **baixar recortes via Overpass**, **fazer rotas/VRP** e **criar casos de teste**.

- `parser.py` — converte **.osm** local → `nodes`/`edges`
- `parser_api.py` — baixa via **Overpass API** um círculo (centro+raio) → `nodes`/`edges`
- `optirota.py` — utilitário de rotas (Dijkstra/A*/tempo) e **VRP multiveículos**
- `vrp_casegen.py` — gerador de arquivo de entrada para o VRP

---

## Dependências

- **Python 3.9+** (recomendado).  
- Sem dependências externas — scripts usam **stdlib** (csv, math, argparse, urllib/xml etc.).  
- **Windows**: se usar CMD/PowerShell, prefira **`--center=<lat,lon>`** (com `=`) para evitar quebra de argumentos.

> Os nomes de saída `nodes`/`edges` **não precisam de extensão**. O conteúdo é CSV com cabeçalho.

---

## Fluxo sugerido (fim‑a‑fim)

1. **Baixar recorte OSM (círculo) → grafos**  
   ```bash
   python parser_api.py --center=-9.648123,-35.717456 --radius=2000      --nodes out/nodes --edges out/edges
   ```
2. **Testar o grafo no roteador** (ex.: A* com tempo)  
   ```bash
   python optirota.py astar_time out/nodes out/edges      -9.648123 -35.717456  -9.640000 -35.710000  car
   ```
3. **Gerar um caso de VRP** (depósito + N veículos + M destinos)  
   ```bash
   python vrp_casegen.py --center=-9.648123,-35.717456 --radius=2000      --vehicles 3 --destinations 12 --out casos/vrp_demo.txt --seed 7
   ```
4. **Executar o VRP** no grafo gerado (com o arquivo do passo 3)  
   ```bash
   python optirota.py vrp out/nodes out/edges casos/vrp_demo.txt 08:30 car
   ```

> Dica: use `--save-osm` no `parser_api.py` para guardar o .osm e repetir testes sem novo download.

---

## 1) `parser.py` — OSM (.osm) → `nodes`/`edges`

**Sintaxe**
```bash
python parser.py --in <arquivo.osm> --nodes <nodes_out> --edges <edges_out> [--log-level NIVEL]
```

**Parâmetros**
- `--in` (**obrigatório**): arquivo **.osm** de entrada.  
- `--nodes` (**obrigatório**): caminho do arquivo de **nós** (saida).  
- `--edges` (**obrigatório**): caminho do arquivo de **arestas** (saida).  
- `--log-level` (opcional): `INFO` (padrão), `DEBUG`, `WARNING`, `ERROR`.

**Saídas**
- **nodes**: `osmid,y,x` (id do nó, lat, lon)  
- **edges**: `u,v,d,name,highway`  
  - `u,v`: nós origem/destino  
  - `d`: distância Haversine (**metros**)  
  - `name`: nome da via (pode ser vazio)  
  - `highway`: categoria OSM

**Regras**
- **Direcionalidade**: `oneway=yes/true/1` → `u→v`; `-1` → `v→u`; senão **ambas**.  
- **Filtros**: exclui `area=yes`, `highway=construction`, `access=private`.  
- **Limpeza**: remove nós consecutivos repetidos (evita arestas de d=0).

**Exemplos**
```bash
python parser.py --in data/maceio.osm --nodes out/nodes --edges out/edges
python parser.py --in ~/mapas/cidade.osm --nodes nodes --edges edges --log-level DEBUG
```

---

## 2) `parser_api.py` — Overpass → `nodes`/`edges` (círculo)

**Sintaxe**
```bash
python parser_api.py --center <lat,lon> --radius <m> --nodes <nodes_out> --edges <edges_out>   [--overpass-url URL] [--save-osm arq.osm] [--log-level NIVEL]
```

**Parâmetros**
- `--center` (**obrigatório**): **lat,lon**. No Windows prefira `--center=<lat,lon>`.  
- `--radius` (**obrigatório**): raio em **metros**.  
- `--nodes` / `--edges` (**obrigatórios**): saídas.  
- `--overpass-url` (opcional): endpoint Overpass (padrão conhecido).  
- `--save-osm` (opcional): salva o `.osm` baixado.  
- `--log-level` (opcional): `INFO` (padrão), `DEBUG`, `WARNING`, `ERROR`.

**Saídas / Regras**: idênticas ao `parser.py` (estrutura e filtros).

**Exemplos**
```bash
# 2 km ao redor do centro
python parser_api.py --center=-9.648123,-35.717456 --radius=2000   --nodes out/nodes --edges out/edges

# Guardar .osm + DEBUG
python parser_api.py --center=-9.648123,-35.717456 --radius=3000   --nodes nodes --edges edges --save-osm data/recorte.osm --log-level DEBUG
```

---

## 3) `optirota.py` — rotas e VRP

Modos aceitos: `stats`, `nearest`, `dijkstra`, `astar`, `astar_time`, `vrp`, `node_to_xy_dist`.  
Perfis: `car`, `bike`, `foot`.

### `stats`
```bash
python optirota.py stats <nodes> <edges>
```

### `nearest`
```bash
python optirota.py nearest <nodes> <edges> <lat> <lon> <car|bike|foot>
# Ex.: python optirota.py nearest out/nodes out/edges -9.648123 -35.717456 car
```

### `dijkstra` (distância mínima)
```bash
python optirota.py dijkstra <nodes> <edges> <lat1> <lon1> <lat2> <lon2> <perfil>
# Ex.: python optirota.py dijkstra out/nodes out/edges -9.648123 -35.717456 -9.640000 -35.710000 car
```

### `astar` (distância mínima com A*)
```bash
python optirota.py astar <nodes> <edges> <lat1> <lon1> <lat2> <lon2> <perfil>
```

### `astar_time` (tempo real acumulado; custo heurístico ajustado)
```bash
python optirota.py astar_time <nodes> <edges> <lat1> <lon1> <lat2> <lon2> <perfil>
```

### `vrp` (multiveículos, capacidade, janelas)
```bash
python optirota.py vrp <nodes> <edges> <input_txt> <start_HH:MM> <perfil>
# Ex.: python optirota.py vrp out/nodes out/edges casos/vrp_demo.txt 08:30 car
```

**Formato do `<input_txt>`**
```
<depot_lat> <depot_lon> <num_veiculos>
<cap1> <cap2> ... <capN>
<lat> <lon> "HH:MM-HH:MM[, HH:MM-HH:MM]" <peso_tons>
<lat> <lon> HH:MM-HH:MM                    <peso_tons>
<lat>, <lon> "HH:MM-HH:MM"                 <peso_tons>
```
- Parser robusto: aceita **aspas ou não**, `lat, lon` com **vírgula** e **múltiplas janelas** separadas por vírgula.  
- Saída do VRP: blocos **“DIA N, VEÍCULO M”**; ao trocar de veículo/virar o dia/encerrar um bloco imprime **“Carga retorna a distribuidora.”**.  
- A função interna do VRP (heurística) também **retorna** uma lista: `[(veículo 1‑based, lista de Edge)]` com as arestas percorridas por veículo.

### `node_to_xy_dist`
Distância (m) de um **nó** do grafo (id) até um **ponto** (lon=x, lat=y).
```bash
python optirota.py node_to_xy_dist <nodes> <edges> <x_lon> <y_lat> <node_id>
```

---

## 4) `vrp_casegen.py` — gerador de casos VRP

Cria um arquivo compatível com o `vrp` do `optirota.py` a partir de **centro+raio** e **N veículos / M destinos**.

**Sintaxe**
```bash
python vrp_casegen.py --center <lat,lon> --radius <m>   --vehicles <N> --destinations <M> --out <arquivo.txt> [opções]
```

**Obrigatórios**
- `--center` (lat,lon), `--radius` (m), `--vehicles` (N), `--destinations` (M), `--out`

**Opcionais (principais)**
- `--seed` (reprodutibilidade)  
- `--caps "3,4.5,5"` (capacidades explícitas) **ou** faixas:
  - `--cap-min` / `--cap-max` (t) — padrão `3.0 / 5.0`
- Pesos por entrega:
  - `--w-min` / `--w-max` (t) — padrão `0.3 / 1.5` (limitados pela maior capacidade)
- Janelas:
  - `--day-start` / `--day-end` (ex.: `08:00` / `18:00`)
  - `--win-min` / `--win-max` (min) — padrão `45 / 120`
  - `--multi-prob` (0..1) para gerar **duas** janelas (padrão `0.25`)

**Exemplos**
```bash
python vrp_casegen.py --center=-9.648123,-35.717456 --radius=2000   --vehicles 3 --destinations 12 --out casos/vrp_demo.txt --seed 7

python vrp_casegen.py --center=-9.65,-35.71 --radius 1500   --vehicles 2 --destinations 8 --out casos/vrp_caps.txt   --caps "3,4.5" --day-start 07:30 --day-end 17:00 --win-min 30 --win-max 60
```

---

## Dicas & solução de problemas

- **Windows/CMD**: use `--center=<lat,lon>` (com `=`) para evitar “expected one argument”.  
- **Arquivos sem extensão**: permitido; conteúdo é **CSV** com cabeçalho.  
- **Área muito densa (Overpass)**: reduza o `--radius` ou use `--save-osm` para reusar o .osm.  
- **VRP inviável**: o algoritmo detecta entregas com peso maior que qualquer capacidade e encerra limpo quando nada é atendível por janelas/alcance. Ajuste capacidades e janelas.  
- **Repetibilidade**: use `--seed` no `vrp_casegen.py`.

---

## Códigos de saída (padrão)

- **0**: sucesso  
- **1**: entrada/argumentos inválidos (arquivo inexistente, raio ≤ 0, etc.)  
- **2**: falha de execução (download, parsing, IO, exceções)

---

## Licença e créditos

- Dados de mapa: **OpenStreetMap** (OSM). Respeite a licença ODbL.  
- Estes scripts são exemplos educacionais/didáticos e podem precisar de ajustes para produção.
