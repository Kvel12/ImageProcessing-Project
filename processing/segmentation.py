"""
Módulo para algoritmos de segmentación de imágenes médicas.
Implementa diferentes técnicas de segmentación para imágenes NIfTI.
"""
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import nibabel as nib
from PIL import Image, ImageTk
from sklearn.cluster import KMeans
import os

class SegmentationProcessor:
    """
    Clase que implementa algoritmos de segmentación para imágenes médicas.
    """
    def __init__(self, parent):
        """
        Inicializa el procesador de segmentación.
        
        Args:
            parent: La instancia de NiftiViewer que contiene este procesador.
        """
        self.parent = parent
    
    def show_options(self, algorithm):
        """
        Muestra opciones de configuración para el algoritmo de segmentación seleccionado.
        
        Args:
            algorithm: El algoritmo de segmentación seleccionado.
        """
        self.parent.seg_window = tk.Toplevel(self.parent.root)
        self.parent.seg_window.title(f"Opciones de {algorithm}")
        self.parent.seg_window.geometry("400x300")
        self.parent.seg_window.transient(self.parent.root)
        self.parent.seg_window.grab_set()
        
        # Centrar ventana
        self.parent.seg_window.update_idletasks()
        width = self.parent.seg_window.winfo_width()
        height = self.parent.seg_window.winfo_height()
        x = (self.parent.seg_window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.parent.seg_window.winfo_screenheight() // 2) - (height // 2)
        self.parent.seg_window.geometry(f'{width}x{height}+{x}+{y}')
    
        frame = ttk.Frame(self.parent.seg_window, padding="10")
        frame.pack(fill="both", expand=True)
    
        # Opciones específicas para cada algoritmo
        if algorithm == "Umbralización":
            ttk.Label(frame, text="Umbral mínimo:").grid(row=0, column=0, sticky="w", pady=5)
            self.thresh_min_var = tk.DoubleVar(value=0.3)
            ttk.Scale(frame, from_=0, to=1, variable=self.thresh_min_var, 
                    orient="horizontal").grid(row=0, column=1, sticky="ew", pady=5)
            ttk.Label(frame, textvariable=self.thresh_min_var).grid(row=0, column=2, padx=5)
        
            ttk.Label(frame, text="Umbral máximo:").grid(row=1, column=0, sticky="w", pady=5)
            self.thresh_max_var = tk.DoubleVar(value=0.7)
            ttk.Scale(frame, from_=0, to=1, variable=self.thresh_max_var, 
                    orient="horizontal").grid(row=1, column=1, sticky="ew", pady=5)
            ttk.Label(frame, textvariable=self.thresh_max_var).grid(row=1, column=2, padx=5)
        
        elif algorithm == "Iso-data":
            ttk.Label(frame, text="Iteraciones máximas:").grid(row=0, column=0, sticky="w", pady=5)
            self.isodata_max_iter_var = tk.IntVar(value=100)
            ttk.Spinbox(frame, from_=10, to=500, textvariable=self.isodata_max_iter_var, 
                      width=5).grid(row=0, column=1, sticky="w", pady=5)
            
            ttk.Label(frame, text="Tolerancia:").grid(row=1, column=0, sticky="w", pady=5)
            self.isodata_tolerance_var = tk.DoubleVar(value=0.001)
            ttk.Spinbox(frame, from_=0.0001, to=0.01, increment=0.0001, 
                      textvariable=self.isodata_tolerance_var, width=5).grid(row=1, column=1, sticky="w", pady=5)
        
        elif algorithm == "Crecimiento":
            ttk.Label(frame, text="Tolerancia:").grid(row=0, column=0, sticky="w", pady=5)
            self.tolerance_var = tk.DoubleVar(value=0.1)
            ttk.Scale(frame, from_=0.01, to=0.5, variable=self.tolerance_var, 
                    orient="horizontal").grid(row=0, column=1, sticky="ew", pady=5)
            ttk.Label(frame, textvariable=self.tolerance_var).grid(row=0, column=2, padx=5)
        
            ttk.Label(frame, text="Para seleccionar un punto semilla:").grid(row=1, column=0, columnspan=3, sticky="w", pady=5)
            ttk.Label(frame, text="1. Haga clic en 'Seleccionar semilla'").grid(row=2, column=0, columnspan=3, sticky="w")
            ttk.Label(frame, text="2. Luego haga clic sobre la imagen").grid(row=3, column=0, columnspan=3, sticky="w")
        
            self.parent.seed_point = None
            ttk.Button(frame, text="Seleccionar semilla", 
                    command=self.parent.drawing_manager.enable_seed_selection).grid(row=4, column=0, columnspan=3, pady=10)
            
            # Nuevo botón para selección automática
            ttk.Button(frame, text="Semilla automática (centro)", 
                    command=self.parent.drawing_manager.select_automatic_seed).grid(row=5, column=0, columnspan=3, pady=5)
        
        elif algorithm == "K-Means":
            ttk.Label(frame, text="Número de clusters (K):").grid(row=0, column=0, sticky="w", pady=5)
            self.k_var = tk.IntVar(value=3)
            ttk.Spinbox(frame, from_=2, to=10, textvariable=self.k_var, width=5).grid(row=0, column=1, sticky="w", pady=5)
        
            ttk.Label(frame, text="Máximo de iteraciones:").grid(row=1, column=0, sticky="w", pady=5)
            self.max_iter_var = tk.IntVar(value=100)
            ttk.Spinbox(frame, from_=10, to=500, textvariable=self.max_iter_var, width=5).grid(row=1, column=1, sticky="w", pady=5)
    
        # Botones comunes
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=10, column=0, columnspan=3, pady=20)
    
        ttk.Button(button_frame, text="Ejecutar", 
                command=lambda: self.run_segmentation(algorithm)).pack(side="left", padx=10)
        ttk.Button(button_frame, text="Cancelar", 
                command=self.parent.seg_window.destroy).pack(side="left", padx=10)
    
        # Ajustar el grid
        frame.columnconfigure(1, weight=1)
    
    def run_segmentation(self, algorithm):
        """
        Ejecuta el algoritmo de segmentación seleccionado.
        
        Args:
            algorithm: El algoritmo de segmentación a ejecutar.
        """
        if self.parent.image_data is None:
            messagebox.showerror("Error", "No hay imagen cargada")
            return
    
        try:
            # Primero cerrar la ventana de opciones
            self.parent.seg_window.destroy()
            
            # Mostrar diálogo de progreso
            progress_window = tk.Toplevel(self.parent.root)
            progress_window.title(f"Procesando {algorithm}")
            progress_window.geometry("300x100")
            progress_window.transient(self.parent.root)
            progress_window.grab_set()
            
            # Centrar ventana
            progress_window.update_idletasks()
            x = (progress_window.winfo_screenwidth() // 2) - (progress_window.winfo_width() // 2)
            y = (progress_window.winfo_screenheight() // 2) - (progress_window.winfo_height() // 2)
            progress_window.geometry(f'+{x}+{y}')
            
            # Añadir controles de progreso
            ttk.Label(progress_window, text=f"Ejecutando segmentación con {algorithm}...").pack(pady=10)
            progress = ttk.Progressbar(progress_window, orient="horizontal", length=250, mode="indeterminate")
            progress.pack(pady=10)
            progress.start()
            
            self.parent.status_var.set(f"Ejecutando segmentación con {algorithm}...")
            self.parent.root.update_idletasks()
        
            # Crear una copia de los datos para no modificar los originales
            segmentation_result = np.zeros_like(self.parent.image_data)
        
            # Ejecutar el algoritmo apropiado
            if algorithm == "Umbralización":
                min_val = np.min(self.parent.image_data)
                max_val = np.max(self.parent.image_data)
                min_threshold = min_val + self.thresh_min_var.get() * (max_val - min_val)
                max_threshold = min_val + self.thresh_max_var.get() * (max_val - min_val)
            
                segmentation_result = self.threshold_segmentation(min_threshold, max_threshold)
                
            elif algorithm == "Iso-data":
                segmentation_result = self.isodata_segmentation(
                    self.isodata_max_iter_var.get(), 
                    self.isodata_tolerance_var.get()
                )
            
            elif algorithm == "Crecimiento":
                if self.parent.seed_point is None:
                    progress_window.destroy()
                    messagebox.showerror("Error", "Debe seleccionar un punto semilla")
                    return
                
                segmentation_result = self.region_growing(
                    self.parent.seed_point, 
                    self.tolerance_var.get()
                )
            
            elif algorithm == "K-Means":
                segmentation_result = self.kmeans_segmentation(
                    self.k_var.get(), 
                    self.max_iter_var.get()
                )
            
            # Cerrar ventana de progreso
            progress_window.destroy()
            
            # Mostrar resultado
            self.show_segmentation_result(segmentation_result, algorithm)
        
        except Exception as e:
            if 'progress_window' in locals() and progress_window.winfo_exists():
                progress_window.destroy()
                
            messagebox.showerror("Error", f"Error al ejecutar la segmentación: {str(e)}")
            self.parent.status_var.set("Error en la segmentación")
    
    def threshold_segmentation(self, min_threshold, max_threshold):
        """
        Implementación de segmentación por umbralización.
        
        Args:
            min_threshold: Umbral mínimo.
            max_threshold: Umbral máximo.
            
        Returns:
            Array NumPy con el resultado de la segmentación.
        """
        result = np.zeros_like(self.parent.image_data)
        mask = (self.parent.image_data >= min_threshold) & (self.parent.image_data <= max_threshold)
        result[mask] = 1
        return result
    
    def isodata_segmentation(self, max_iterations=100, tolerance=0.001):
        """
        Implementación de segmentación por umbralización Iso-data (algoritmo de Ridler-Calvard).
        Este algoritmo encuentra automáticamente un umbral óptimo.
        
        Args:
            max_iterations: Número máximo de iteraciones.
            tolerance: Tolerancia para convergencia.
            
        Returns:
            Array NumPy con el resultado de la segmentación.
        """
        # Implementación mejorada basada en el código proporcionado por el usuario
        result = np.zeros_like(self.parent.image_data)
        
        # Procesar cada slice
        for z in range(self.parent.depth):
            # Actualizar mensaje de estado cada 10 slices
            if z % 10 == 0:
                self.parent.status_var.set(f"Iso-data: procesando slice {z+1}/{self.parent.depth}")
                self.parent.root.update_idletasks()
                
            # Obtener slice
            slice_data = self.parent.image_data[:, :, z].copy()
            
            # Aplanar la imagen y filtrar valores no válidos
            flat_data = slice_data.flatten()
            flat_data = flat_data[~np.isnan(flat_data)]  # Eliminar NaN
            
            # Inicializar umbral como la media de los datos
            threshold = np.mean(flat_data)
            
            # Iterar hasta convergencia o máximo de iteraciones
            for i in range(max_iterations):
                # Dividir datos en dos grupos según umbral actual
                group1 = flat_data[flat_data < threshold]
                group2 = flat_data[flat_data >= threshold]
                
                # Asegurarse de que ambos grupos tienen al menos un elemento
                if len(group1) == 0 or len(group2) == 0:
                    # Si un grupo está vacío, usar otro método para inicializar
                    low, high = np.min(flat_data), np.max(flat_data)
                    threshold = (low + high) / 2
                    continue
                
                # Calcular medias de cada grupo
                mean1 = np.mean(group1)
                mean2 = np.mean(group2)
                
                # Calcular nuevo umbral como promedio de las medias
                new_threshold = (mean1 + mean2) / 2
                
                # Verificar convergencia
                if abs(new_threshold - threshold) < tolerance:
                    break
                    
                threshold = new_threshold
            
            # Aplicar umbral para generar segmentación binaria
            result[:, :, z] = (slice_data >= threshold).astype(np.float64)
        
        return result
    
    def region_growing(self, seed_point, tolerance):
        """
        Implementación de segmentación por crecimiento de regiones.
        Se ha optimizado para rendimiento y reducción de memoria.
        
        Args:
            seed_point: Punto semilla (x, y, z).
            tolerance: Tolerancia para incluir píxeles en la región.
            
        Returns:
            Array NumPy con el resultado de la segmentación.
        """
        # Obtener coordenadas del punto semilla
        x, y, z = seed_point
    
        # Obtener el valor del punto semilla
        seed_value = self.parent.image_data[x, y, z]
    
        # Calcular rango de tolerancia
        min_val = np.min(self.parent.image_data)
        max_val = np.max(self.parent.image_data)
        tolerance_range = tolerance * (max_val - min_val)
    
        # Crear máscara para el resultado
        result = np.zeros_like(self.parent.image_data)
    
        # Crear array para controlar los puntos visitados
        processed = np.zeros_like(self.parent.image_data, dtype=bool)
    
        # Lista de puntos a procesar (comienza con la semilla)
        points_queue = [seed_point]
    
        # Dirección de vecinos (6-conectividad: arriba, abajo, izquierda, derecha, adelante, atrás)
        neighbors = [
            (-1, 0, 0), (1, 0, 0),
            (0, -1, 0), (0, 1, 0),
            (0, 0, -1), (0, 0, 1)
        ]
        
        # Mostrar progreso
        total_processed = 0
        update_interval = 5000  # Actualizar cada 5000 puntos procesados
        
        # Procesamiento por lotes para mejorar rendimiento
        batch_size = 1000
        current_batch = []
    
        # Mientras haya puntos por procesar
        while points_queue:
            # Procesar puntos en lotes para mejorar rendimiento
            current_batch = points_queue[:batch_size]
            points_queue = points_queue[batch_size:]
            
            for point in current_batch:
                current_x, current_y, current_z = point
            
                # Si está fuera de los límites o ya fue procesado, continuar
                if (current_x < 0 or current_x >= self.parent.width or
                    current_y < 0 or current_y >= self.parent.height or
                    current_z < 0 or current_z >= self.parent.depth or
                    processed[current_x, current_y, current_z]):
                    continue
            
                # Marcar como procesado
                processed[current_x, current_y, current_z] = True
                total_processed += 1
            
                if total_processed % update_interval == 0:
                    self.parent.status_var.set(f"Crecimiento de regiones: {total_processed} puntos procesados...")
                    self.parent.root.update_idletasks()
            
                # Obtener valor del punto actual
                current_value = self.parent.image_data[current_x, current_y, current_z]
            
                # Si está dentro del rango de tolerancia, agregar a la región
                if abs(current_value - seed_value) <= tolerance_range:
                    result[current_x, current_y, current_z] = 1
                
                    # Agregar vecinos a la cola
                    for dx, dy, dz in neighbors:
                        new_x, new_y, new_z = current_x + dx, current_y + dy, current_z + dz
                        if (0 <= new_x < self.parent.width and 
                            0 <= new_y < self.parent.height and 
                            0 <= new_z < self.parent.depth and 
                            not processed[new_x, new_y, new_z]):
                            points_queue.append((new_x, new_y, new_z))
            
            # Si no hay más puntos en la cola y no quedan más en el lote, salir
            if not points_queue and not current_batch:
                break
                
        self.parent.status_var.set(f"Crecimiento de regiones completado: {total_processed} puntos procesados")
    
        return result
    
    def kmeans_segmentation(self, k, max_iterations=100):
        """
        Implementación optimizada de segmentación por K-Means.
        Procesa cada slice individualmente para reducir uso de memoria.
        
        Args:
            k: Número de clusters.
            max_iterations: Máximo de iteraciones.
            
        Returns:
            Array NumPy con el resultado de la segmentación.
        """
        # Inicializar el resultado
        result = np.zeros_like(self.parent.image_data)
        
        # Procesar cada slice por separado para reducir memoria
        for z in range(self.parent.depth):
            # Actualizar estado
            self.parent.status_var.set(f"K-Means: procesando slice {z+1}/{self.parent.depth}")
            self.parent.root.update_idletasks()
            
            # Obtener slice
            slice_data = self.parent.image_data[:, :, z]
            
            # Aplanar y normalizar datos
            flattened_data = slice_data.flatten().reshape(-1, 1)
            
            # Usar KMeans de scikit-learn
            kmeans = KMeans(n_clusters=k, max_iter=max_iterations, random_state=42, n_init=10)
            labels = kmeans.fit_predict(flattened_data)
            
            # Reshape de vuelta a la forma original
            result[:, :, z] = labels.reshape(slice_data.shape)
        
        # Normalizar resultado a valores entre 0 y 1 para visualización
        result = result / (k - 1)
        
        return result
    
    def show_segmentation_result(self, result, algorithm):
        """
        Muestra el resultado de la segmentación en una nueva ventana.
        
        Args:
            result: Resultado de la segmentación.
            algorithm: Algoritmo utilizado.
        """
        # Crear una nueva ventana
        result_window = tk.Toplevel(self.parent.root)
        result_window.title(f"Resultado de Segmentación: {algorithm}")
        result_window.geometry("800x700")
        
        # Centrar ventana
        result_window.update_idletasks()
        width = result_window.winfo_width()
        height = result_window.winfo_height()
        x = (result_window.winfo_screenwidth() // 2) - (width // 2)
        y = (result_window.winfo_screenheight() // 2) - (height // 2)
        result_window.geometry(f'{width}x{height}+{x}+{y}')
    
        # Variables para la ventana de resultados
        self.result_data = result
        self.result_slice_type = "Axial"
        self.result_slice_index = self.parent.depth // 2 if self.result_slice_type == "Axial" else (
            self.parent.width // 2 if self.result_slice_type == "Sagittal" else self.parent.height // 2)
    
        # Frame para controles
        control_frame = ttk.Frame(result_window)
        control_frame.pack(fill="x", padx=10, pady=5)
    
        # Botones para cambiar vista
        ttk.Button(control_frame, text="Axial", 
                 command=lambda: self.change_result_view("Axial", result_window)).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Sagittal", 
                 command=lambda: self.change_result_view("Sagittal", result_window)).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Coronal", 
                 command=lambda: self.change_result_view("Coronal", result_window)).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Vista 3D", 
                 command=lambda: self.parent.visualization.visualize_segmentation_3d(result)).pack(side="left", padx=5)
    
        # Checkbox para comparación
        self.show_comparison_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Mostrar comparación", 
                       variable=self.show_comparison_var, 
                       command=lambda: self.update_result_slice(self.result_slice_index, result_window)).pack(side="left", padx=10)
    
        # Slider para navegación
        slice_frame = ttk.Frame(result_window)
        slice_frame.pack(fill="x", padx=10, pady=5)
    
        self.result_slider = ttk.Scale(slice_frame, from_=0, to=100, orient="horizontal", 
                                      command=lambda v: self.update_result_slice(v, result_window))
        self.result_slider.pack(fill="x", padx=10, pady=5)
    
        self.result_slice_label = ttk.Label(slice_frame, text="Slice: 0/0")
        self.result_slice_label.pack(pady=2)
    
        # Frame con barras de desplazamiento para el canvas
        canvas_frame = ttk.Frame(result_window)
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Barras de desplazamiento
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient="horizontal")
        h_scrollbar.pack(side="bottom", fill="x")
        
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical")
        v_scrollbar.pack(side="right", fill="y")
    
        # Canvas para mostrar la imagen
        self.result_canvas = tk.Canvas(canvas_frame, 
                                    xscrollcommand=h_scrollbar.set,
                                    yscrollcommand=v_scrollbar.set)
        self.result_canvas.pack(fill="both", expand=True)
        
        # Configurar barras de desplazamiento
        h_scrollbar.config(command=self.result_canvas.xview)
        v_scrollbar.config(command=self.result_canvas.yview)
    
        # Botones adicionales
        btn_frame = ttk.Frame(result_window)
        btn_frame.pack(fill="x", padx=10, pady=5)
    
        ttk.Button(btn_frame, text="Aplicar como marcado", 
                 command=lambda: self.apply_segmentation_as_overlay(result)).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Exportar a NIfTI", 
                 command=lambda: self.export_segmentation(result)).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Cerrar", 
                 command=result_window.destroy).pack(side="right", padx=5)
    
        # Configurar slider según la orientación
        self.setup_result_slider()
    
        # Mostrar la primera imagen
        self.update_result_slice(self.result_slice_index, result_window)
    
    def change_result_view(self, view_type, window):
        """
        Cambia la orientación de visualización del resultado.
        
        Args:
            view_type: Tipo de vista (Axial, Sagittal, Coronal).
            window: Ventana de resultados.
        """
        self.result_slice_type = view_type
    
        # Resetear el índice de corte según la orientación
        if view_type == "Axial":
            self.result_slice_index = self.parent.depth // 2
        elif view_type == "Sagittal":
            self.result_slice_index = self.parent.width // 2
        else:  # Coronal
            self.result_slice_index = self.parent.height // 2
    
        # Actualizar slider y vista
        self.setup_result_slider()
        self.update_result_slice(self.result_slice_index, window)
    
    def setup_result_slider(self):
        """Configura el slider según la orientación actual."""
        if self.result_slice_type == "Axial":
            max_slice = self.parent.depth - 1
        elif self.result_slice_type == "Sagittal":
            max_slice = self.parent.width - 1
        else:  # Coronal
            max_slice = self.parent.height - 1
    
        self.result_slider.config(from_=0, to=max_slice)
        self.result_slider.set(self.result_slice_index)
    
    def update_result_slice(self, value, window):
        """
        Actualiza la visualización del corte del resultado.
        
        Args:
            value: Valor del slider.
            window: Ventana de resultados.
        """
        if isinstance(value, str):
            value = float(value)
        self.result_slice_index = int(value)
    
        # Obtener el corte según la orientación
        if self.result_slice_type == "Axial":
            slice_data = self.result_data[:, :, self.result_slice_index]
            original_slice = self.parent.image_data[:, :, self.result_slice_index]
            max_slice = self.parent.depth - 1
        elif self.result_slice_type == "Sagittal":
            slice_data = self.result_data[self.result_slice_index, :, :]
            original_slice = self.parent.image_data[self.result_slice_index, :, :]
            max_slice = self.parent.width - 1
        else:  # Coronal
            slice_data = self.result_data[:, self.result_slice_index, :]
            original_slice = self.parent.image_data[:, self.result_slice_index, :]
            max_slice = self.parent.height - 1
    
        # Actualizar etiqueta
        self.result_slice_label.config(text=f"Slice: {self.result_slice_index}/{max_slice}")
    
        # Procesar imagen original
        norm_original = self.parent.image_utils.normalize_image(original_slice)
        color_original = self.parent.image_utils.apply_colormap(norm_original)
    
        # Crear overlay coloreado para la segmentación
        unique_labels = np.unique(slice_data)
        num_labels = len(unique_labels)
    
        # Crear mapa de colores para la segmentación
        colormap = {}
        for i, label in enumerate(unique_labels):
            if label == 0:  # Fondo es transparente
                colormap[label] = (0, 0, 0, 0)
            else:
                # Crear colores distintos para cada etiqueta
                hue = (i-1) / max(1, num_labels-1) * 180  # Valores HSV de 0 a 180 para OpenCV
                sat = 255
                val = 255
                bgr = cv2.cvtColor(np.uint8([[[hue, sat, val]]]), cv2.COLOR_HSV2BGR)[0][0]
                colormap[label] = (*bgr, 150)  # BGR + alpha
    
        # Crear imagen de overlay
        overlay = np.zeros((*slice_data.shape, 4), dtype=np.uint8)
        for label, color in colormap.items():
            mask = (slice_data == label)
            for c in range(3):  # RGB channels
                overlay[..., c][mask] = color[c]
            overlay[..., 3][mask] = color[3]  # Alpha channel
    
        # Resize ambas imágenes
        display_size = (512, 512)
        color_original_resized = cv2.resize(color_original, display_size)
        overlay_resized = cv2.resize(overlay, display_size)
    
        # Mezclar original con overlay
        result_img = color_original_resized.copy()
        for y in range(display_size[1]):
            for x in range(display_size[0]):
                if overlay_resized[y, x, 3] > 0:  # Si hay algo en el overlay (alpha > 0)
                    alpha = overlay_resized[y, x, 3] / 255.0
                    for c in range(3):
                        result_img[y, x, c] = int(result_img[y, x, c] * (1 - alpha) + overlay_resized[y, x, c] * alpha)
    
        # Si se seleccionó la comparación, mostrar lado a lado
        if hasattr(self, 'show_comparison_var') and self.show_comparison_var.get():
            # Crear imagen compuesta: original | segmentada
            combined_img = np.zeros((display_size[1], display_size[0] * 2, 3), dtype=np.uint8)
            combined_img[:, :display_size[0], :] = color_original_resized
            combined_img[:, display_size[0]:, :] = result_img
            
            # Añadir texto
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(combined_img, "Original", (10, 30), font, 1, (255, 255, 255), 2)
            cv2.putText(combined_img, "Segmentada", (display_size[0] + 10, 30), font, 1, (255, 255, 255), 2)
            
            # Dibujar línea divisoria
            cv2.line(combined_img, (display_size[0], 0), (display_size[0], display_size[1]), (255, 255, 255), 2)
            
            # Ajustar tamaño del canvas
            self.result_canvas.config(width=display_size[0] * 2, height=display_size[1])
        else:
            # Mostrar solo imagen segmentada
            combined_img = result_img
            self.result_canvas.config(width=display_size[0], height=display_size[1])
    
        # Mostrar en el canvas
        img_pil = Image.fromarray(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))
        img_tk = ImageTk.PhotoImage(img_pil)
    
        self.result_canvas.config(width=img_pil.width, height=img_pil.height)
        if hasattr(self, 'result_img_on_canvas'):
            self.result_canvas.delete(self.result_img_on_canvas)
        self.result_img_on_canvas = self.result_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.result_canvas.image = img_tk  # Mantener referencia
    
    def apply_segmentation_as_overlay(self, segmentation):
        """
        Aplica el resultado de la segmentación como una capa de dibujo.
        
        Args:
            segmentation: Resultado de la segmentación.
        """
        if messagebox.askyesno("Aplicar Segmentación", 
                            "¿Desea aplicar la segmentación como marcado en la imagen original?"):
            # Identificar voxels segmentados
            segmented_indices = np.where(segmentation > 0)
        
            # Convertir a puntos de dibujo
            for i in range(len(segmented_indices[0])):
                x, y, z = segmented_indices[0][i], segmented_indices[1][i], segmented_indices[2][i]
            
                # Crear un punto con el color actual
                self.parent.overlay_data[x, y, z] = 1
                self.parent.draw_points.append({
                    'x': int(x),
                    'y': int(y),
                    'z': int(z),
                    'color': self.parent.draw_color
                })
        
            # Actualizar visualización
            self.parent.visualization.update_slice()
            self.parent.status_var.set("Segmentación aplicada como marcado")
    
    def export_segmentation(self, segmentation):
        """
        Exporta el resultado de la segmentación como archivo NIfTI.
        
        Args:
            segmentation: Resultado de la segmentación.
        """
        try:
            file_path = tk.filedialog.asksaveasfilename(
                defaultextension=".nii.gz",
                filetypes=[("NIfTI Files", "*.nii *.nii.gz")],
                title="Guardar Segmentación como NIfTI"
            )
        
            if not file_path:
                return
            
            # Crear un nuevo objeto NIfTI con los datos de segmentación
            segmentation_nii = nib.Nifti1Image(segmentation.astype(np.int16), self.parent.nii_image.affine)
        
            # Guardar el archivo
            nib.save(segmentation_nii, file_path)
        
            self.parent.status_var.set(f"Segmentación guardada en {os.path.basename(file_path)}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Error al exportar segmentación: {str(e)}")