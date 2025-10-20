#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VRP Route Visualizer with Tkinter
Integra parser.py, optirota.py para visualização de rotas com mapa
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import sys
import threading
import math
from typing import List, Tuple, Optional

# Importar módulos (ajustar caminhos conforme necessário)
# Assumindo que parser.py e optirota.py estão no mesmo diretório


class VRPVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("OptiRota Visualizer")
        self.root.geometry("1200x950")
        
        self.osm_file = None
        self.output_nodes_csv = Path.cwd() / "out_nodes.csv"
        self.output_edges_csv = Path.cwd() / "out_edges.csv"
        self.input_vrp_txt = None
        self.graph = None
        self.routes = []
        
        self.setup_ui()
    
    def setup_ui(self):
        """Configura a interface do usuário"""
        # Frame de entrada
        input_frame = ttk.LabelFrame(self.root, text="Configuração de Entrada e Saída", padding=10)
        input_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Seleção de arquivo OSM
        ttk.Label(input_frame, text="Arquivo OSM:").grid(row=0, column=0, sticky=tk.W)
        self.osm_label = ttk.Label(input_frame, text="Nenhum arquivo selecionado", 
                                    foreground="gray")
        self.osm_label.grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Button(input_frame, text="Selecionar OSM", 
                  command=self.select_osm_file).grid(row=0, column=2, padx=5)
        
        # Arquivo VRP de entrada
        ttk.Label(input_frame, text="Arquivo VRP (entrada):").grid(row=1, column=0, sticky=tk.W)
        self.vrp_input_label = ttk.Label(input_frame, text="Nenhum arquivo selecionado", 
                                         foreground="gray")
        self.vrp_input_label.grid(row=1, column=1, sticky=tk.W, padx=5)
        ttk.Button(input_frame, text="Selecionar VRP", 
                  command=self.select_vrp_input_file).grid(row=1, column=2, padx=5)
        
        # Diretório de saída
        ttk.Label(input_frame, text="Diretório de Saída (CSVs):").grid(row=2, column=0, sticky=tk.W)
        self.output_label = ttk.Label(input_frame, text=str(Path.cwd()), foreground="blue")
        self.output_label.grid(row=2, column=1, sticky=tk.W, padx=5)
        ttk.Button(input_frame, text="Mudar", 
                  command=self.select_output_dir).grid(row=2, column=2, padx=5)
        
        # Frame de parâmetros globais
        params_frame = ttk.LabelFrame(self.root, text="Parâmetros Globais", padding=10)
        params_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(params_frame, text="Modo de Transporte:").grid(row=0, column=0, sticky=tk.W)
        self.transport_var = tk.StringVar(value="car")
        self.transport_combobox = ttk.Combobox(params_frame, textvariable=self.transport_var, 
                    values=["car", "bike", "foot"], state="readonly", width=10)
        self.transport_combobox.grid(row=0, column=1, sticky=tk.W, padx=5)

        # --- NOVA SEÇÃO: PATHFINDING (PONTO A PONTO) ---
        path_frame = ttk.LabelFrame(self.root, text="Pathfinding (Ponto a Ponto)", padding=10)
        path_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(path_frame, text="Partida (lat,lon):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.start_coord_var = tk.StringVar(value="-9.6481,-35.7174")
        ttk.Entry(path_frame, textvariable=self.start_coord_var, width=25).grid(row=0, column=1, sticky=tk.W)

        ttk.Label(path_frame, text="Chegada (lat,lon):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.end_coord_var = tk.StringVar(value="-9.6653,-35.7354")
        ttk.Entry(path_frame, textvariable=self.end_coord_var, width=25).grid(row=1, column=1, sticky=tk.W)

        ttk.Label(path_frame, text="Algoritmo:").grid(row=0, column=2, sticky=tk.W, padx=(10,0))
        self.algo_var = tk.StringVar(value="A* (Tempo)")
        self.algo_combobox = ttk.Combobox(path_frame, textvariable=self.algo_var,
                                          values=["Dijkstra", "A* (Distância)", "A* (Tempo)"],
                                          state="readonly", width=15)
        self.algo_combobox.grid(row=0, column=3, sticky=tk.W)

        ttk.Button(path_frame, text="Calcular e Visualizar Rota", 
                   command=self.run_pathfinding).grid(row=1, column=2, columnspan=2, padx=10, sticky=tk.E)


        # --- SEÇÃO VRP ---
        vrp_params_frame = ttk.LabelFrame(self.root, text="Parâmetros VRP", padding=10)
        vrp_params_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(vrp_params_frame, text="Hora de Início (HH:MM):").grid(row=0, column=0, sticky=tk.W)
        self.start_time_var = tk.StringVar(value="08:00")
        ttk.Entry(vrp_params_frame, textvariable=self.start_time_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5)
        
        # Frame de controle
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(control_frame, text="1. Processar OSM → CSV", 
                  command=self.process_osm).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="2. Executar VRP", 
                  command=self.run_vrp).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="3. Visualizar Rotas VRP", 
                  command=self.visualize_vrp_routes).pack(side=tk.LEFT, padx=5)
        
        self.progress_var = tk.DoubleVar()
        progress = ttk.Progressbar(control_frame, variable=self.progress_var, maximum=100)
        progress.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        # Frame de log
        log_frame = ttk.LabelFrame(self.root, text="Log de Execução", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        scrollbar = ttk.Scrollbar(log_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_text = tk.Text(log_frame, height=15, yscrollcommand=scrollbar.set)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.log_text.yview)
    
    def log_message(self, message):
        """Adiciona mensagem ao log"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def select_osm_file(self):
        """Seleciona arquivo OSM"""
        file = filedialog.askopenfilename(filetypes=[("OSM files", "*.osm"), ("All files", "*.*")])
        if file:
            self.osm_file = Path(file)
            self.osm_label.config(text=self.osm_file.name, foreground="black")
            self.log_message(f"✓ OSM selecionado: {self.osm_file}")
    
    def select_vrp_input_file(self):
        """Seleciona arquivo VRP de entrada"""
        file = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file:
            self.input_vrp_txt = Path(file)
            self.vrp_input_label.config(text=self.input_vrp_txt.name, foreground="black")
            self.log_message(f"✓ VRP de entrada selecionado: {self.input_vrp_txt}")
    
    def select_output_dir(self):
        """Seleciona diretório de saída"""
        dir_path = filedialog.askdirectory()
        if dir_path:
            output_dir = Path(dir_path)
            self.output_nodes_csv = output_dir / "out_nodes.csv"
            self.output_edges_csv = output_dir / "out_edges.csv"
            self.output_label.config(text=str(output_dir), foreground="blue")
            self.log_message(f"✓ Diretório de saída: {output_dir}")
    
    def process_osm(self):
        """Processa arquivo OSM para gerar CSVs"""
        if not self.osm_file or not self.osm_file.exists():
            messagebox.showerror("Erro", "Selecione um arquivo OSM válido")
            return
        
        self.log_message("\nProcessando OSM para CSV...")
        
        def process():
            try:
                import subprocess
                # Tenta localizar o script parser.py
                parser_script = Path(__file__).parent / "parser.py"
                if not parser_script.exists():
                    self.log_message(f"✗ Erro: 'parser.py' não encontrado no diretório do script.")
                    return

                # Monta os argumentos para chamar o script 'parser_api.py' ou 'parser.py'
                # Neste contexto, vamos assumir que o usuário selecionou um .osm local
                cmd = [
                    sys.executable, str(parser_script),
                    "--in", str(self.osm_file),
                    "--nodes", str(self.output_nodes_csv),
                    "--edges", str(self.output_edges_csv)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    self.log_message("✓ OSM processado com sucesso!")
                    self.log_message(f"  Nodes: {self.output_nodes_csv}")
                    self.log_message(f"  Edges: {self.output_edges_csv}")
                    self.progress_var.set(25)
                else:
                    self.log_message(f"✗ Erro ao processar OSM: {result.stderr}")
                    self.progress_var.set(0)
            except Exception as e:
                self.log_message(f"✗ Exceção ao processar OSM: {str(e)}")
                self.progress_var.set(0)
        
        thread = threading.Thread(target=process, daemon=True)
        thread.start()
    
    def run_vrp(self):
        """Executa o algoritmo VRP"""
        if not self.input_vrp_txt or not self.input_vrp_txt.exists():
            messagebox.showerror("Erro", "Selecione um arquivo VRP válido")
            return
        
        if not self.output_nodes_csv.exists() or not self.output_edges_csv.exists():
            messagebox.showerror("Erro", "Arquivos CSV de nós/arestas não encontrados. Processe o OSM primeiro.")
            return
        
        start_time = self.start_time_var.get()
        transport = self.transport_var.get()
        
        self.log_message(f"\nExecutando VRP...")
        self.log_message(f"  Hora de Início: {start_time}")
        self.log_message(f"  Transporte: {transport}")
        
        def execute():
            try:
                import subprocess
                optirota_script = Path(__file__).parent / "optirota.py"
                if not optirota_script.exists():
                     self.log_message(f"✗ Erro: 'optirota.py' não encontrado.")
                     return

                cmd = [
                    sys.executable, str(optirota_script), "vrp",
                    str(self.output_nodes_csv),
                    str(self.output_edges_csv),
                    str(self.input_vrp_txt),
                    start_time,
                    transport
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                
                if result.returncode == 0:
                    self.log_message("✓ VRP executado com sucesso!")
                    if result.stdout:
                        for line in result.stdout.split("\n"):
                            if line.strip():
                                self.log_message(f"  {line}")
                    self.progress_var.set(100)
                else:
                    self.log_message(f"✗ Erro na execução do VRP: {result.stderr}")
                    if result.stdout: self.log_message(f"  Saída: {result.stdout}")
                    self.progress_var.set(0)
            except Exception as e:
                self.log_message(f"✗ Exceção ao executar VRP: {str(e)}")
                self.progress_var.set(0)
        
        thread = threading.Thread(target=execute, daemon=True)
        thread.start()
    
    def run_pathfinding(self):
        """Executa um algoritmo de pathfinding (Dijkstra/A*) e visualiza."""
        if not self.output_nodes_csv.exists() or not self.output_edges_csv.exists():
            messagebox.showerror("Erro", "Arquivos CSV de nós/arestas não encontrados.\nExecute '1. Processar OSM → CSV' primeiro.")
            return

        try:
            lat1, lon1 = map(float, self.start_coord_var.get().split(','))
            lat2, lon2 = map(float, self.end_coord_var.get().split(','))
        except (ValueError, IndexError):
            messagebox.showerror("Erro de Formato", "Coordenadas devem estar no formato 'latitude,longitude'.")
            return
            
        algo_map = {
            "Dijkstra": "dijkstra",
            "A* (Distância)": "astar",
            "A* (Tempo)": "astar_time"
        }
        algorithm = algo_map[self.algo_var.get()]
        transport = self.transport_var.get()

        self.log_message(f"\nCalculando rota com {self.algo_var.get()}...")
        self.log_message(f"  De: ({lat1}, {lon1}) Para: ({lat2}, {lon2})")
        self.log_message(f"  Modo: {transport}")

        def execute():
            try:
                import subprocess
                optirota_script = Path(__file__).parent / "optirota.py"
                if not optirota_script.exists():
                    self.log_message(f"✗ Erro: 'optirota.py' não encontrado.")
                    return

                # Modificar optirota.py para salvar a rota em JSON
                # e chamar a visualização aqui.
                cmd = [
                    sys.executable, str(optirota_script), algorithm,
                    str(self.output_nodes_csv), str(self.output_edges_csv),
                    str(lat1), str(lon1), str(lat2), str(lon2), transport
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

                if result.returncode == 0:
                    self.log_message("✓ Rota calculada com sucesso!")
                    if result.stdout:
                         for line in result.stdout.split("\n"):
                            if line.strip(): self.log_message(f"  {line}")
                    
                    # Chamar visualização do path
                    self.visualize_single_path((lat1, lon1), (lat2, lon2))
                else:
                    self.log_message(f"✗ Erro no cálculo da rota: {result.stderr}")
                    if result.stdout: self.log_message(f"  Saída: {result.stdout}")

            except Exception as e:
                self.log_message(f"✗ Exceção no cálculo da rota: {str(e)}")

        thread = threading.Thread(target=execute, daemon=True)
        thread.start()
        
    def visualize_single_path(self, start_coords, end_coords):
        """Abre uma janela de visualização para uma única rota."""
        # O arquivo JSON é gerado pelo script optirota.py modificado
        output_dir = self.output_nodes_csv.parent
        path_json_file = output_dir / "path_route.json"

        if not path_json_file.exists():
            messagebox.showerror("Erro", f"Arquivo de rota não encontrado: {path_json_file}\nVerifique o log para erros no cálculo.")
            return
            
        self.log_message("\nGerando mapa com rota ponto a ponto...")

        def generate_map():
            try:
                import folium
                import json
                import tempfile
                import webbrowser
                import os
                
                with open(path_json_file, 'r', encoding='utf-8') as f:
                    path_data = json.load(f)

                if not path_data.get("routes"):
                    self.log_message("✗ Rota vazia ou não encontrada no arquivo JSON.")
                    messagebox.showinfo("Informação", "Não foi possível encontrar um caminho entre os pontos.")
                    return

                route_info = path_data["routes"][0]
                segments = route_info.get("segments", [])
                
                if not segments:
                    self.log_message("✗ Rota sem segmentos.")
                    messagebox.showinfo("Informação", "Não foi possível encontrar um caminho entre os pontos.")
                    return

                # Centralizar mapa
                map_center = start_coords
                m = folium.Map(location=map_center, zoom_start=15, tiles='OpenStreetMap')

                # Marcadores
                folium.Marker(location=start_coords, popup="<b>Partida</b>", icon=folium.Icon(color='green', icon='play', prefix='fa')).add_to(m)
                folium.Marker(location=end_coords, popup="<b>Chegada</b>", icon=folium.Icon(color='red', icon='stop', prefix='fa')).add_to(m)

                # Desenhar rota
                coordinates = [segments[0]["from"]] + [s["to"] for s in segments]
                folium.PolyLine(
                    locations=coordinates, color='#0000FF', weight=4, opacity=0.8,
                    tooltip=f"Distância: {route_info['total_distance_m']:.1f}m"
                ).add_to(m)

                # Salvar e abrir
                temp_dir = Path(tempfile.gettempdir())
                map_file = temp_dir / "path_visualization.html"
                m.save(str(map_file))
                
                webbrowser.open('file://' + os.path.realpath(str(map_file)))
                
                self.log_message(f"✓ Mapa da rota gerado!")
                self.log_message(f"  Arquivo: {map_file}")
                self.log_message(f"  Distância Total: {route_info['total_distance_m']:.1f}m")

            except Exception as e:
                self.log_message(f"✗ Erro ao gerar mapa da rota: {str(e)}")
                import traceback
                self.log_message(traceback.format_exc())
                messagebox.showerror("Erro", f"Erro ao gerar mapa: {str(e)}")

        # A geração do mapa pode ser rápida, mas mantemos em thread por consistência
        thread = threading.Thread(target=generate_map, daemon=True)
        thread.start()

    def visualize_vrp_routes(self):
        """Abre janela de visualização das rotas VRP com Folium"""
        if not self.input_vrp_txt or not self.input_vrp_txt.exists():
            messagebox.showerror("Erro", "Arquivo VRP não selecionado")
            return
        
        routes_json = self.input_vrp_txt.parent / "vrp_routes.json"
        if not routes_json.exists():
            messagebox.showerror("Erro", 
                "Arquivo de rotas VRP não encontrado. Execute '2. Executar VRP' primeiro.\n"
                f"Esperado: {routes_json}")
            return
        
        self.log_message("\nGerando mapa com rotas VRP...")
        
        def generate_map():
            try:
                import folium
                import json
                import tempfile
                import webbrowser
                import os
                
                with open(self.input_vrp_txt, 'r', encoding='utf-8') as f:
                    vrp_lines = [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]
                
                depot_parts = vrp_lines[0].replace(',', ' ').split()
                depot_lat, depot_lon = float(depot_parts[0]), float(depot_parts[1])
                
                with open(routes_json, 'r', encoding='utf-8') as f:
                    routes_data = json.load(f)
                
                cores = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000']
                
                m = folium.Map(location=[depot_lat, depot_lon], zoom_start=14, tiles='OpenStreetMap')
                
                folium.Marker(
                    location=[depot_lat, depot_lon],
                    popup="<b>DEPÓSITO</b>",
                    tooltip="Depósito/Distribuição",
                    icon=folium.Icon(color='black', icon='warehouse', prefix='fa')
                ).add_to(m)
                
                all_points = set()

                for route in routes_data.get("routes", []):
                    vehicle_id = route["vehicle_id"]
                    cor = cores[(vehicle_id - 1) % len(cores)]
                    segments = route.get("segments", [])
                    
                    if segments:
                        coordinates = [segments[0]["from"]] + [s["to"] for s in segments]
                        
                        folium.PolyLine(
                            locations=coordinates, color=cor, weight=3, opacity=0.8,
                            tooltip=f"Rota Veículo {vehicle_id}"
                        ).add_to(m)

                        # Adicionar marcadores de entrega (destinos únicos)
                        for seg in segments:
                             point_to = tuple(seg["to"])
                             if point_to not in all_points:
                                 folium.CircleMarker(
                                     location=point_to, radius=5, color=cor, fill=True,
                                     fillColor=cor, fillOpacity=0.7,
                                     tooltip=f"Entrega Veículo {vehicle_id}"
                                 ).add_to(m)
                                 all_points.add(point_to)

                # Salvar e abrir
                temp_dir = Path(tempfile.gettempdir())
                map_file = temp_dir / "vrp_rotas_reais.html"
                m.save(str(map_file))
                
                webbrowser.open('file://' + os.path.realpath(str(map_file)))
                
                self.log_message(f"✓ Mapa com rotas VRP gerado!")
                self.log_message(f"  Arquivo: {map_file}")
                
            except Exception as e:
                self.log_message(f"✗ Erro ao gerar mapa VRP: {str(e)}")
                import traceback
                self.log_message(traceback.format_exc())
                messagebox.showerror("Erro", f"Erro ao gerar mapa: {str(e)}")
        
        thread = threading.Thread(target=generate_map, daemon=True)
        thread.start()


def main():
    root = tk.Tk()
    app = VRPVisualizer(root)
    root.mainloop()


if __name__ == "__main__":
    main()