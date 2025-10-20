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
        self.root.title("VRP Route Visualizer")
        self.root.geometry("1200x800")
        
        self.osm_file = None
        self.output_nodes_csv = None
        self.output_edges_csv = None
        self.input_vrp_txt = None
        self.graph = None
        self.routes = []
        
        self.setup_ui()
    
    def setup_ui(self):
        """Configura a interface do usuário"""
        # Frame de entrada
        input_frame = ttk.LabelFrame(self.root, text="Configuração de Entrada", padding=10)
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
        ttk.Label(input_frame, text="Diretório de Saída:").grid(row=2, column=0, sticky=tk.W)
        self.output_label = ttk.Label(input_frame, text=str(Path.cwd()), foreground="blue")
        self.output_label.grid(row=2, column=1, sticky=tk.W, padx=5)
        ttk.Button(input_frame, text="Mudar", 
                  command=self.select_output_dir).grid(row=2, column=2, padx=5)
        
        # Frame de parâmetros
        params_frame = ttk.LabelFrame(self.root, text="Parâmetros VRP", padding=10)
        params_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Horário de início
        ttk.Label(params_frame, text="Hora de Início (HH:MM):").grid(row=0, column=0, sticky=tk.W)
        self.start_time_var = tk.StringVar(value="08:00")
        ttk.Entry(params_frame, textvariable=self.start_time_var, width=10).grid(row=0, column=1, 
                                                                                  sticky=tk.W, padx=5)
        
        # Modo de transporte
        ttk.Label(params_frame, text="Modo de Transporte:").grid(row=0, column=2, sticky=tk.W)
        self.transport_var = tk.StringVar(value="car")
        ttk.Combobox(params_frame, textvariable=self.transport_var, 
                    values=["car", "bike", "foot"], state="readonly", width=8).grid(row=0, column=3, 
                                                                                      sticky=tk.W, padx=5)
        
        # Número de veículos
        ttk.Label(params_frame, text="Número de Veículos:").grid(row=1, column=0, sticky=tk.W)
        self.num_vehicles_var = tk.StringVar(value="1")
        ttk.Spinbox(params_frame, from_=1, to=10, textvariable=self.num_vehicles_var, 
                   width=5).grid(row=1, column=1, sticky=tk.W, padx=5)
        
        # Frame de controle
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(control_frame, text="Processar OSM → CSV", 
                  command=self.process_osm).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Executar VRP", 
                  command=self.run_vrp).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Visualizar Rotas", 
                  command=self.visualize_routes).pack(side=tk.LEFT, padx=5)
        
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
        
        if not self.output_nodes_csv:
            output_dir = Path.cwd()
            self.output_nodes_csv = output_dir / "out_nodes.csv"
            self.output_edges_csv = output_dir / "out_edges.csv"
        
        self.log_message("Processando OSM...")
        
        def process():
            try:
                import subprocess
                # Chamar parser.py
                result = subprocess.run([
                    sys.executable, "parser.py",
                    "--in", str(self.osm_file),
                    "--nodes", str(self.output_nodes_csv),
                    "--edges", str(self.output_edges_csv)
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    self.log_message("✓ OSM processado com sucesso!")
                    self.log_message(f"  Nodes: {self.output_nodes_csv}")
                    self.log_message(f"  Edges: {self.output_edges_csv}")
                    self.progress_var.set(50)
                else:
                    self.log_message(f"✗ Erro ao processar: {result.stderr}")
                    self.progress_var.set(0)
            except Exception as e:
                self.log_message(f"✗ Exceção: {str(e)}")
                self.progress_var.set(0)
        
        thread = threading.Thread(target=process, daemon=True)
        thread.start()
    
    def run_vrp(self):
        """Executa o algoritmo VRP"""
        if not self.input_vrp_txt or not self.input_vrp_txt.exists():
            messagebox.showerror("Erro", "Selecione um arquivo VRP válido")
            return
        
        if not self.output_nodes_csv or not self.output_edges_csv:
            messagebox.showerror("Erro", "Processe o OSM primeiro")
            return
        
        start_time = self.start_time_var.get()
        transport = self.transport_var.get()
        
        self.log_message(f"\nExecutando VRP...")
        self.log_message(f"  Hora: {start_time}")
        self.log_message(f"  Transporte: {transport}")
        
        def execute():
            try:
                import subprocess
                result = subprocess.run([
                    sys.executable, "optirota.py", "vrp",
                    str(self.output_nodes_csv),
                    str(self.output_edges_csv),
                    str(self.input_vrp_txt),
                    start_time,
                    transport
                ], capture_output=True, text=True, timeout=600)
                
                if result.returncode == 0:
                    self.log_message("✓ VRP executado com sucesso!")
                    if result.stdout:
                        for line in result.stdout.split("\n"):
                            if line.strip():
                                self.log_message(f"  {line}")
                    self.progress_var.set(100)
                else:
                    self.log_message(f"✗ Erro: {result.stderr}")
                    self.progress_var.set(0)
            except Exception as e:
                self.log_message(f"✗ Exceção: {str(e)}")
                self.progress_var.set(0)
        
        thread = threading.Thread(target=execute, daemon=True)
        thread.start()
    
    def visualize_routes(self):
        """Abre janela de visualização das rotas REAIS com Folium"""
        if not self.input_vrp_txt or not self.input_vrp_txt.exists():
            messagebox.showerror("Erro", "Arquivo VRP não selecionado")
            return
        
        # Procura pelo arquivo de rotas JSON
        routes_json = self.input_vrp_txt.parent / "vrp_routes.json"
        if not routes_json.exists():
            messagebox.showerror("Erro", 
                "Arquivo de rotas não encontrado. Execute 'Executar VRP' primeiro.\n"
                f"Esperado: {routes_json}")
            return
        
        self.log_message("\nGerando mapa com rotas reais...")
        
        def generate_map():
            try:
                import folium
                import json
                import tempfile
                import webbrowser
                import os
                
                # Parse arquivo VRP
                with open(self.input_vrp_txt, 'r', encoding='utf-8') as f:
                    vrp_lines = [l.strip() for l in f if l.strip()]
                
                depot_parts = vrp_lines[0].replace(',', ' ').split()
                depot_lat, depot_lon = float(depot_parts[0]), float(depot_parts[1])
                
                # Carregar rotas
                with open(routes_json, 'r', encoding='utf-8') as f:
                    routes_data = json.load(f)
                
                # Cores para cada veículo
                cores = [
                    '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', 
                    '#00FFFF', '#FFA500', '#800080', '#FFC0CB', '#A52A2A',
                    '#008000', '#000080', '#008080', '#800080', '#FFD700'
                ]
                
                # Criar mapa
                m = folium.Map(
                    location=[depot_lat, depot_lon],
                    zoom_start=14,
                    tiles='OpenStreetMap'
                )
                
                # Marcador do depósito
                folium.Marker(
                    location=[depot_lat, depot_lon],
                    popup=f"<b>DEPÓSITO</b><br>Lat: {depot_lat:.6f}<br>Lon: {depot_lon:.6f}",
                    tooltip="Depósito/Distribuição",
                    icon=folium.Icon(color='red', icon='warehouse', prefix='fa')
                ).add_to(m)
                
                # Desenhar rotas
                for route in routes_data.get("routes", []):
                    vehicle_id = route["vehicle_id"]
                    cor = cores[(vehicle_id - 1) % len(cores)]
                    segments = route.get("segments", [])
                    
                    # Construir polilinha da rota
                    if segments:
                        coordinates = []
                        # Adicionar ponto de partida (depósito)
                        coordinates.append([depot_lat, depot_lon])
                        
                        # Adicionar todos os pontos intermediários
                        for segment in segments:
                            from_point = segment["from"]
                            to_point = segment["to"]
                            coordinates.append(from_point)
                            coordinates.append(to_point)
                        
                        # Desenhar a rota do veículo
                        folium.PolyLine(
                            locations=coordinates,
                            color=cor,
                            weight=3,
                            opacity=0.8,
                            popup=f"Veículo {vehicle_id}",
                            tooltip=f"Rota Veículo {vehicle_id} - {route['total_distance_m']:.1f}m"
                        ).add_to(m)
                        
                        # Marcador de início/fim de rota
                        if coordinates:
                            folium.CircleMarker(
                                location=coordinates[-1],
                                radius=5,
                                popup=f"Fim rota veículo {vehicle_id}",
                                color=cor,
                                fill=True,
                                fillColor=cor,
                                fillOpacity=0.7,
                                weight=2
                            ).add_to(m)
                
                # Adicionar legenda
                legend_html = f'''
                <div style="position: fixed; 
                     bottom: 50px; right: 50px; width: 250px; height: auto;
                     background-color: white; border:2px solid grey; z-index:9999; 
                     font-size:12px; padding: 10px; border-radius: 5px;
                     max-height: 400px; overflow-y: auto;">
                    <p style="margin: 0; font-weight: bold; margin-bottom: 10px;">Legenda</p>
                    <p><i class="fa fa-warehouse" style="color:red; margin-right: 5px;"></i> Depósito</p>
                '''
                
                for route in routes_data.get("routes", []):
                    vehicle_id = route["vehicle_id"]
                    cor = cores[(vehicle_id - 1) % len(cores)]
                    dist = route.get("total_distance_m", 0)
                    legend_html += f'<p style="margin: 5px 0;"><span style="display:inline-block; width:20px; height:3px; background-color:{cor}; margin-right: 5px;"></span> Veículo {vehicle_id} ({dist:.0f}m)</p>'
                
                legend_html += '</div>'
                m.get_root().html.add_child(folium.Element(legend_html))
                
                # Salvar e abrir
                temp_dir = Path(tempfile.gettempdir())
                map_file = temp_dir / "vrp_rotas_reais.html"
                m.save(str(map_file))
                
                webbrowser.open('file://' + os.path.realpath(str(map_file)))
                
                self.log_message(f"✓ Mapa com rotas reais gerado!")
                self.log_message(f"  Arquivo: {map_file}")
                total_rotas = len(routes_data.get("routes", []))
                self.log_message(f"  Total de rotas: {total_rotas}")
                for route in routes_data.get("routes", []):
                    self.log_message(f"    Veículo {route['vehicle_id']}: {route['total_distance_m']:.1f}m")
                
            except Exception as e:
                self.log_message(f"✗ Erro ao gerar mapa: {str(e)}")
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