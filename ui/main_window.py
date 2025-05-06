"""
Módulo principal para la interfaz de usuario del visor NIfTI.
Contiene la clase NiftiViewer que maneja la ventana principal y todas las interacciones con el usuario.
"""
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, colorchooser
import os
import sys
import json
import numpy as np

# Importar módulos de visualización y dibujo
from ui.visualization import VisualizationManager
from ui.drawing import DrawingManager

# Importar módulos de procesamiento
from processing.segmentation import SegmentationProcessor
from processing.filtering import FilterProcessor

# Importar utilidades
from utils.io_utils import IOUtils
from utils.image_utils import ImageUtils

class NiftiViewer:
    """
    Clase principal que maneja la interfaz de usuario del visor NIfTI.
    """
    def __init__(self, root):
        """Inicializa la aplicación y configura la interfaz de usuario."""
        self.root = root
        self.root.title("NIfTI Viewer con Visualización 3D y Herramientas de Dibujo")
        self.root.geometry("800x700")
        
        # Centrar la ventana en la pantalla
        self._center_window()
        
        # Variables
        self.image_data = None
        self.nii_image = None
        self.width, self.height, self.depth = 0, 0, 0
        self.corte_actual = "Axial"
        self.indice_corte = 0
        self.file_path = None
        
        # Variables de dibujo
        self.drawing_mode = False  # Renombrado para evitar confusión
        self.last_x = 0
        self.last_y = 0
        self.draw_radius = 3
        self.draw_color = (255, 0, 0)  # Rojo por defecto
        self.draw_points = []  # Lista para almacenar puntos dibujados
        self.overlay_data = None  # Array 3D para superponer dibujos
        self.current_display_img = None  # Almacena la imagen actual mostrada
        self.seed_selection_mode = False
        
        # Inicializar administradores
        self.init_managers()
        
        # Crear elementos de la interfaz
        self.create_ui()
        
        # Establecer icono de la ventana
        self.set_window_icon()

    def _center_window(self):
        """Centra la ventana en la pantalla"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def init_managers(self):
        """Inicializa los administradores para diferentes funcionalidades."""
        self.io_utils = IOUtils()
        self.image_utils = ImageUtils()
        self.visualization = VisualizationManager(self)
        self.drawing_manager = DrawingManager(self)  # Nombre corregido para evitar confusión
        self.segmentation = SegmentationProcessor(self)
        self.filtering = FilterProcessor(self)

    def set_window_icon(self):
        """Establece el icono de la ventana principal."""
        icon_path = self.resource_path("brain_icon.png")
        if os.path.exists(icon_path):
            try:
                icon = tk.PhotoImage(file=icon_path)
                self.root.iconphoto(False, icon)
            except Exception as e:
                print(f"Error al cargar el icono: {e}")

    def resource_path(self, relative_path):
        """
        Obtiene la ruta absoluta a un recurso, funciona para desarrollo y PyInstaller.
        """
        try:
            # PyInstaller crea una carpeta temporal y almacena la ruta en _MEIPASS
            base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
            # Ajuste para la nueva estructura modular
            if '__file__' in locals() and os.path.dirname(os.path.abspath(__file__)) in base_path:
                # Estamos en modo de desarrollo, ajustar ruta
                base_path = os.path.join(os.path.dirname(base_path), 'resources')
            return os.path.join(base_path, relative_path)
        except Exception:
            return relative_path
    
    def create_ui(self):
        """Crea los elementos de la interfaz de usuario."""
        # Crear menú
        self.create_menu()
        
        # Crear layout principal con panel lateral
        self.create_main_layout()
        
    def create_main_layout(self):
        """Crea el layout principal con panel lateral para mejor visualización"""
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True)
        
        # Panel lateral (izquierdo)
        sidebar_frame = ttk.LabelFrame(main_frame, text="Controles")
        sidebar_frame.pack(side="left", fill="y", padx=10, pady=5)
        
        # Panel de contenido (derecho)
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        
        # Controles en el panel lateral
        self.create_sidebar_controls(sidebar_frame)
        
        # Contenido principal en el panel de contenido
        self.create_content_area(content_frame)
        
        # Barra de estado
        self.create_status_bar()

    def create_sidebar_controls(self, parent):
        """Crea los controles en el panel lateral"""
        # Sección de archivo
        file_frame = ttk.LabelFrame(parent, text="Archivo")
        file_frame.pack(fill="x", padx=5, pady=5, ipady=5)
        
        self.btn_load = ttk.Button(file_frame, text="Abrir archivo NIfTI", command=self.load_image)
        self.btn_load.pack(fill="x", padx=5, pady=5)
        
        # Información del archivo
        info_frame = ttk.LabelFrame(parent, text="Información")
        info_frame.pack(fill="x", padx=5, pady=5)
        
        self.label_info = ttk.Label(info_frame, text="No se ha cargado ningún archivo")
        self.label_info.pack(padx=5, pady=5)
        
        # Sección de vista
        view_frame = ttk.LabelFrame(parent, text="Vista")
        view_frame.pack(fill="x", padx=5, pady=5)
        
        self.btn_axial = ttk.Button(view_frame, text="Axial", 
                                  command=lambda: self.change_slice_type("Axial"), state="disabled")
        self.btn_axial.pack(fill="x", padx=5, pady=2)
        
        self.btn_sagittal = ttk.Button(view_frame, text="Sagital", 
                                     command=lambda: self.change_slice_type("Sagittal"), state="disabled")
        self.btn_sagittal.pack(fill="x", padx=5, pady=2)
        
        self.btn_coronal = ttk.Button(view_frame, text="Coronal", 
                                    command=lambda: self.change_slice_type("Coronal"), state="disabled")
        self.btn_coronal.pack(fill="x", padx=5, pady=2)
        
        self.btn_3d = ttk.Button(view_frame, text="Vista 3D", 
                               command=self.visualize_3d, state="disabled")
        self.btn_3d.pack(fill="x", padx=5, pady=2)
        
        # Controles de dibujo
        draw_frame = ttk.LabelFrame(parent, text="Herramientas de Dibujo")
        draw_frame.pack(fill="x", padx=5, pady=5)
        
        self.btn_color = ttk.Button(draw_frame, text="Color", 
                                  command=self.choose_color, state="disabled")
        self.btn_color.pack(fill="x", padx=5, pady=2)
        
        # Botón de exportar al lado del selector de color
        self.btn_save_drawings = ttk.Button(draw_frame, text="Exportar Dibujo", 
                                         command=self.save_drawings, state="disabled")
        self.btn_save_drawings.pack(fill="x", padx=5, pady=2)
        
        self.btn_clear = ttk.Button(draw_frame, text="Borrar", 
                                  command=self.clear_drawings, state="disabled")
        self.btn_clear.pack(fill="x", padx=5, pady=2)
        
        # Frame para color y modo de dibujo
        color_mode_frame = ttk.Frame(draw_frame)
        color_mode_frame.pack(fill="x", padx=5, pady=2)
        
        self.color_preview = tk.Canvas(color_mode_frame, width=20, height=20, bg="red")
        self.color_preview.pack(side="left", padx=5)
        
        self.draw_mode_var = tk.BooleanVar()
        self.draw_mode_var.set(False)
        self.chk_draw = ttk.Checkbutton(color_mode_frame, text="Modo Dibujo", 
                                      variable=self.draw_mode_var, state="disabled")
        self.chk_draw.pack(side="left", padx=5)
        
        # Tamaño del pincel
        brush_frame = ttk.Frame(draw_frame)
        brush_frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(brush_frame, text="Tamaño:").pack(side="left")
        
        self.brush_size_var = tk.IntVar(value=3)
        brushes = [("S", 1), ("M", 3), ("L", 5)]
        
        for text, value in brushes:
            # Crear el radiobutton con un valor específico capturado en el lambda
            rb = ttk.Radiobutton(brush_frame, text=text, value=value, 
                      variable=self.brush_size_var,
                      command=lambda v=value: self.set_brush_size(v))
            rb.pack(side="left", padx=5)
            # Aplicar estado disabled después de crear el widget
            rb.state(["disabled"])
        
        # Navegación de cortes en el panel lateral
        slice_frame = ttk.LabelFrame(parent, text="Navegación de Cortes")
        slice_frame.pack(fill="x", padx=5, pady=5)
        
        self.slice_slider = ttk.Scale(slice_frame, from_=0, to=100, orient="horizontal", 
                                    command=self.update_slice)
        self.slice_slider.pack(fill="x", padx=5, pady=5)
        self.slice_slider.state(["disabled"])
        
        self.slice_label = ttk.Label(slice_frame, text="Corte: 0/0")
        self.slice_label.pack(padx=5, pady=2)

    def create_content_area(self, parent):
        """Crea el área de contenido principal con la visualización de imagen"""
        # Frame para visualización de imagen
        display_frame = ttk.LabelFrame(parent, text="Visualización de Imagen")
        display_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Canvas con barras de desplazamiento
        canvas_frame = ttk.Frame(display_frame)
        canvas_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Barras de desplazamiento
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient="horizontal")
        h_scrollbar.pack(side="bottom", fill="x")
        
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical")
        v_scrollbar.pack(side="right", fill="y")
        
        # Canvas para mostrar la imagen
        self.canvas = tk.Canvas(canvas_frame, 
                             xscrollcommand=h_scrollbar.set,
                             yscrollcommand=v_scrollbar.set)
        self.canvas.pack(fill="both", expand=True)
        
        # Configurar barras de desplazamiento
        h_scrollbar.config(command=self.canvas.xview)
        v_scrollbar.config(command=self.canvas.yview)
        
        # Vincular eventos del mouse para dibujo
        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)
    
    def create_menu(self):
        """Crea la barra de menú principal."""
        menubar = tk.Menu(self.root)
        
        # Menú Archivo
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Abrir archivo NIfTI", command=self.load_image)
        filemenu.add_separator()
        filemenu.add_command(label="Guardar dibujos", command=self.save_drawings, state="disabled")
        filemenu.add_command(label="Cargar dibujos", command=self.load_drawings, state="disabled")
        filemenu.add_separator()
        filemenu.add_command(label="Salir", command=self.root.quit)
        menubar.add_cascade(label="Archivo", menu=filemenu)
        
        # Menú Vista
        viewmenu = tk.Menu(menubar, tearoff=0)
        viewmenu.add_command(label="Vista Axial", command=lambda: self.change_slice_type("Axial"), state="disabled")
        viewmenu.add_command(label="Vista Sagital", command=lambda: self.change_slice_type("Sagittal"), state="disabled")
        viewmenu.add_command(label="Vista Coronal", command=lambda: self.change_slice_type("Coronal"), state="disabled")
        viewmenu.add_separator()
        viewmenu.add_command(label="Visualización 3D", command=self.visualize_3d, state="disabled")
        menubar.add_cascade(label="Vista", menu=viewmenu)
        
        # Menú Dibujo
        drawmenu = tk.Menu(menubar, tearoff=0)
        drawmenu.add_command(label="Cambiar color de dibujo", command=self.choose_color, state="disabled")
        drawmenu.add_command(label="Borrar dibujos", command=self.clear_drawings, state="disabled")
        drawmenu.add_separator()
        
        # Submenú para tamaño del pincel
        sizemenu = tk.Menu(drawmenu, tearoff=0)
        sizemenu.add_command(label="Pequeño (1px)", command=lambda: self.set_brush_size(1), state="disabled")
        sizemenu.add_command(label="Medio (3px)", command=lambda: self.set_brush_size(3), state="disabled")
        sizemenu.add_command(label="Grande (5px)", command=lambda: self.set_brush_size(5), state="disabled")
        drawmenu.add_cascade(label="Tamaño del pincel", menu=sizemenu)
        
        menubar.add_cascade(label="Dibujo", menu=drawmenu)
        
        # Menú Segmentación
        segmenu = tk.Menu(menubar, tearoff=0)
        segmenu.add_command(label="Umbralización", 
                          command=lambda: self.show_segmentation_options("Umbralización"), state="disabled")
        segmenu.add_command(label="Umbralización Iso-data", 
                          command=lambda: self.show_segmentation_options("Iso-data"), state="disabled")
        segmenu.add_command(label="Crecimiento de Regiones", 
                          command=lambda: self.show_segmentation_options("Crecimiento"), state="disabled")
        segmenu.add_command(label="K-Means", 
                          command=lambda: self.show_segmentation_options("K-Means"), state="disabled")
        menubar.add_cascade(label="Segmentación", menu=segmenu)
        
        # Menú Preprocesamiento
        prepmenu = tk.Menu(menubar, tearoff=0)
        prepmenu.add_command(label="Filtro Media", 
                           command=lambda: self.show_preprocessing_options("Media"), state="disabled")
        prepmenu.add_command(label="Filtro Mediana", 
                           command=lambda: self.show_preprocessing_options("Mediana"), state="disabled")
        prepmenu.add_command(label="Filtro Bilateral (preserva bordes)", 
                           command=lambda: self.show_preprocessing_options("Bilateral"), state="disabled")
        prepmenu.add_command(label="Filtro Anisotrópico (preserva bordes)", 
                           command=lambda: self.show_preprocessing_options("Anisotropico"), state="disabled")
        prepmenu.add_command(label="Filtro Non-Local Means", 
                           command=lambda: self.show_preprocessing_options("NLM"), state="disabled")
        menubar.add_cascade(label="Preprocesamiento", menu=prepmenu)
        
        # Menú Ayuda
        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="Acerca de", command=self.show_about)
        menubar.add_cascade(label="Ayuda", menu=helpmenu)
        
        self.root.config(menu=menubar)
        
        # Guardar referencias a los menús
        self.filemenu = filemenu
        self.viewmenu = viewmenu
        self.drawmenu = drawmenu
        self.sizemenu = sizemenu
        self.segmenu = segmenu
        self.prepmenu = prepmenu
        
    def create_status_bar(self):
        """Crea la barra de estado en la parte inferior."""
        self.status_var = tk.StringVar()
        self.status_var.set("Listo")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                                  relief="sunken", anchor="w")
        self.status_bar.pack(side="bottom", fill="x")
        
        # Visualización de coordenadas
        self.coord_var = tk.StringVar()
        self.coord_var.set("Coordenadas: -")
        self.coord_label = ttk.Label(self.root, textvariable=self.coord_var, 
                                   relief="sunken", anchor="e")
        self.coord_label.pack(side="bottom", fill="x")
    
    def load_image(self):
        """Carga un archivo de imagen NIfTI."""
        self.io_utils.load_image(self)
    
    def change_slice_type(self, slice_type):
        """Cambia la orientación del corte."""
        self.visualization.change_slice_type(slice_type)
    
    def update_slice(self, *args):
        """Actualiza el corte mostrado."""
        self.visualization.update_slice(*args)
    
    def choose_color(self):
        """Abre un diálogo para elegir el color de dibujo."""
        self.drawing_manager.choose_color()
    
    def set_brush_size(self, size):
        """Establece el tamaño del pincel de dibujo."""
        self.draw_radius = size
        self.drawing_manager.set_brush_size(size)
    
    def start_draw(self, event):
        """Inicia el dibujo al presionar el mouse."""
        self.drawing_manager.start_draw(event)
    
    def draw(self, event):
        """Dibuja mientras el mouse se mueve."""
        self.drawing_manager.draw(event)
    
    def stop_draw(self, event):
        """Detiene el dibujo al soltar el mouse."""
        self.drawing_manager.stop_draw(event)
    
    def clear_drawings(self):
        """Borra todos los dibujos."""
        self.drawing_manager.clear_drawings()
    
    def save_drawings(self):
        """Guarda los puntos de dibujo en un archivo JSON."""
        self.io_utils.save_drawings(self)
    
    def load_drawings(self):
        """Carga los puntos de dibujo desde un archivo JSON."""
        self.io_utils.load_drawings(self)
    
    def show_about(self):
        """Muestra el diálogo Acerca de."""
        messagebox.showinfo(
            "Acerca de",
            "NIfTI Viewer con Visualización 3D y Herramientas de Dibujo\n\n"
            "Un visor simple para datos de neuroimagen en formato NIfTI.\n"
            "Características:\n"
            "- Visualización de cortes 2D (Axial, Sagital, Coronal)\n"
            "- Herramientas de dibujo para marcar regiones de interés\n"
            "- Seguimiento de coordenadas 3D para marcadores\n"
            "- Guardar/cargar coordenadas de dibujo\n"
            "- Visualización con mapa de colores tipo-CT\n"
            "- Algoritmos de segmentación y filtrado"
        )
    
    def visualize_3d(self):
        """Crea una visualización 3D de los datos NIfTI usando VTK."""
        self.visualization.visualize_3d()
    
    def show_segmentation_options(self, algorithm):
        """Muestra opciones de configuración para el algoritmo de segmentación seleccionado."""
        self.segmentation.show_options(algorithm)
    
    def show_preprocessing_options(self, filter_type):
        """Muestra opciones de configuración para el filtro seleccionado."""
        self.filtering.show_options(filter_type)