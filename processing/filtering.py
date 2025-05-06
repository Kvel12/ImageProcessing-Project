"""
Módulo para implementar filtros de preprocesamiento de imágenes médicas.
Incluye filtros para reducción de ruido y mejora de imagen.
"""
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import nibabel as nib
from PIL import Image, ImageTk
import os
from scipy import ndimage

class FilterProcessor:
    """
    Clase que implementa filtros de preprocesamiento para imágenes médicas.
    """
    def __init__(self, parent):
        """
        Inicializa el procesador de filtros.
        
        Args:
            parent: La instancia de NiftiViewer que contiene este procesador.
        """
        self.parent = parent
    
    def show_options(self, filter_type):
        """
        Muestra opciones de configuración para el filtro seleccionado.
        
        Args:
            filter_type: El tipo de filtro seleccionado.
        """
        self.parent.prep_window = tk.Toplevel(self.parent.root)
        self.parent.prep_window.title(f"Opciones de Filtro {filter_type}")
        self.parent.prep_window.geometry("400x300")
        self.parent.prep_window.transient(self.parent.root)
        self.parent.prep_window.grab_set()
        
        # Centrar ventana
        self.parent.prep_window.update_idletasks()
        width = self.parent.prep_window.winfo_width()
        height = self.parent.prep_window.winfo_height()
        x = (self.parent.prep_window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.parent.prep_window.winfo_screenheight() // 2) - (height // 2)
        self.parent.prep_window.geometry(f'{width}x{height}+{x}+{y}')
    
        frame = ttk.Frame(self.parent.prep_window, padding="10")
        frame.pack(fill="both", expand=True)
    
        # Opciones específicas para cada filtro
        if filter_type == "Media":
            ttk.Label(frame, text="Tamaño de kernel:").grid(row=0, column=0, sticky="w", pady=5)
            self.kernel_size_var = tk.IntVar(value=3)
            sizes = [3, 5, 7, 9]
            kernel_combobox = ttk.Combobox(frame, textvariable=self.kernel_size_var, values=sizes, state="readonly", width=5)
            kernel_combobox.grid(row=0, column=1, sticky="w", pady=5)
            kernel_combobox.current(0)
        
        elif filter_type == "Mediana":
            ttk.Label(frame, text="Tamaño de kernel:").grid(row=0, column=0, sticky="w", pady=5)
            self.kernel_size_var = tk.IntVar(value=3)
            sizes = [3, 5, 7, 9]
            kernel_combobox = ttk.Combobox(frame, textvariable=self.kernel_size_var, values=sizes, state="readonly", width=5)
            kernel_combobox.grid(row=0, column=1, sticky="w", pady=5)
            kernel_combobox.current(0)
        
        elif filter_type == "Bilateral":
            ttk.Label(frame, text="Tamaño de ventana:").grid(row=0, column=0, sticky="w", pady=5)
            self.window_size_var = tk.IntVar(value=9)
            ttk.Spinbox(frame, from_=3, to=15, textvariable=self.window_size_var, width=5).grid(row=0, column=1, sticky="w", pady=5)
        
            ttk.Label(frame, text="Sigma espacial:").grid(row=1, column=0, sticky="w", pady=5)
            self.sigma_space_var = tk.DoubleVar(value=1.5)
            ttk.Spinbox(frame, from_=0.1, to=5.0, increment=0.1, textvariable=self.sigma_space_var, width=5).grid(row=1, column=1, sticky="w", pady=5)
        
            ttk.Label(frame, text="Sigma rango:").grid(row=2, column=0, sticky="w", pady=5)
            self.sigma_range_var = tk.DoubleVar(value=50.0)
            ttk.Spinbox(frame, from_=10.0, to=150.0, increment=5.0, textvariable=self.sigma_range_var, width=5).grid(row=2, column=1, sticky="w", pady=5)
        
        elif filter_type == "Anisotropico":
            ttk.Label(frame, text="Iteraciones:").grid(row=0, column=0, sticky="w", pady=5)
            self.iterations_var = tk.IntVar(value=10)
            ttk.Spinbox(frame, from_=1, to=50, textvariable=self.iterations_var, width=5).grid(row=0, column=1, sticky="w", pady=5)
        
            ttk.Label(frame, text="Kappa (conductancia):").grid(row=1, column=0, sticky="w", pady=5)
            self.kappa_var = tk.DoubleVar(value=50.0)
            ttk.Spinbox(frame, from_=1.0, to=100.0, increment=1.0, textvariable=self.kappa_var, width=5).grid(row=1, column=1, sticky="w", pady=5)
        
            ttk.Label(frame, text="Lambda (paso tiempo):").grid(row=2, column=0, sticky="w", pady=5)
            self.lambda_var = tk.DoubleVar(value=0.25)
            ttk.Spinbox(frame, from_=0.05, to=0.25, increment=0.05, textvariable=self.lambda_var, width=5).grid(row=2, column=1, sticky="w", pady=5)
            
        elif filter_type == "NLM":
            ttk.Label(frame, text="Tamaño de parche:").grid(row=0, column=0, sticky="w", pady=5)
            self.nlm_patch_size_var = tk.IntVar(value=3)
            ttk.Combobox(frame, textvariable=self.nlm_patch_size_var, 
                        values=[3, 5, 7], width=5, state="readonly").grid(row=0, column=1, sticky="w", pady=5)
        
            ttk.Label(frame, text="Radio de búsqueda:").grid(row=1, column=0, sticky="w", pady=5)
            self.nlm_search_var = tk.IntVar(value=5)
            ttk.Combobox(frame, textvariable=self.nlm_search_var, 
                        values=[5, 7, 9, 11], width=5, state="readonly").grid(row=1, column=1, sticky="w", pady=5)
        
            ttk.Label(frame, text="Parámetro h (fuerza):").grid(row=2, column=0, sticky="w", pady=5)
            self.nlm_h_var = tk.DoubleVar(value=0.1)
            ttk.Scale(frame, from_=0.01, to=0.5, variable=self.nlm_h_var, 
                    orient="horizontal").grid(row=2, column=1, sticky="ew", pady=5)
            ttk.Label(frame, textvariable=self.nlm_h_var).grid(row=2, column=2, padx=5)
    
        # Botones comunes
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=10, column=0, columnspan=3, pady=20)
    
        ttk.Button(button_frame, text="Aplicar", 
                 command=lambda: self.run_preprocessing(filter_type)).pack(side="left", padx=10)
        ttk.Button(button_frame, text="Cancelar", 
                 command=self.parent.prep_window.destroy).pack(side="left", padx=10)
    
        # Ajustar el grid
        frame.columnconfigure(1, weight=1)
    
    def run_preprocessing(self, filter_type):
        """
        Ejecuta el filtro de preprocesamiento seleccionado.
        
        Args:
            filter_type: El tipo de filtro a aplicar.
        """
        if self.parent.image_data is None:
            messagebox.showerror("Error", "No hay imagen cargada")
            return
    
        try:
            self.parent.status_var.set(f"Aplicando filtro {filter_type}...")
            self.parent.root.update_idletasks()
        
            # Crear una copia de los datos para no modificar los originales
            preprocessed_data = self.parent.image_data.copy()
        
            # Aplicar el filtro apropiado
            if filter_type == "Media":
                preprocessed_data = self.mean_filter(preprocessed_data, self.kernel_size_var.get())
            elif filter_type == "Mediana":
                preprocessed_data = self.median_filter(preprocessed_data, self.kernel_size_var.get())
            elif filter_type == "Bilateral":
                preprocessed_data = self.bilateral_filter(
                    preprocessed_data, 
                    self.window_size_var.get(),
                    self.sigma_space_var.get(),
                    self.sigma_range_var.get()
                )
            elif filter_type == "Anisotropico":
                preprocessed_data = self.anisotropic_diffusion(
                    preprocessed_data,
                    self.iterations_var.get(),
                    self.kappa_var.get(),
                    self.lambda_var.get()
                )
            elif filter_type == "NLM":
                preprocessed_data = self.non_local_means(
                    preprocessed_data,
                    self.nlm_patch_size_var.get(),
                    self.nlm_search_var.get(),
                    self.nlm_h_var.get()
                )
            
            # Mostrar resultado
            self.show_preprocessing_result(preprocessed_data, filter_type)
        
        except Exception as e:
            messagebox.showerror("Error", f"Error al aplicar el filtro: {str(e)}")
            self.parent.status_var.set("Error en el preprocesamiento")
    
        # Cerrar ventana de opciones
        self.parent.prep_window.destroy()
    
    def mean_filter(self, data, kernel_size):
        """
        Implementa un filtro de media optimizado utilizando scipy.ndimage.
        
        Args:
            data: Los datos de imagen a filtrar.
            kernel_size: Tamaño del kernel.
            
        Returns:
            Array NumPy con la imagen filtrada.
        """
        # Usar uniform_filter de scipy para un filtro de media eficiente
        return ndimage.uniform_filter(data, size=kernel_size)
    
    def median_filter(self, data, kernel_size):
        """
        Implementa un filtro de mediana eficiente utilizando scipy.ndimage.
        
        Args:
            data: Los datos de imagen a filtrar.
            kernel_size: Tamaño del kernel.
            
        Returns:
            Array NumPy con la imagen filtrada.
        """
        # Usar median_filter de scipy para un filtro de mediana eficiente
        return ndimage.median_filter(data, size=kernel_size)
    
    def bilateral_filter(self, data, window_size, sigma_space, sigma_range):
        """
        Implementa un filtro bilateral que preserva bordes.
        Procesa cada slice individualmente para reducir uso de memoria.
        
        Args:
            data: Los datos de imagen a filtrar.
            window_size: Tamaño de la ventana.
            sigma_space: Sigma espacial.
            sigma_range: Sigma de rango.
            
        Returns:
            Array NumPy con la imagen filtrada.
        """
        result = np.zeros_like(data)
        
        # Procesar cada slice por separado para reducir uso de memoria
        for z in range(self.parent.depth):
            # Actualizar estado
            self.parent.status_var.set(f"Aplicando filtro bilateral: slice {z+1}/{self.parent.depth}")
            self.parent.root.update_idletasks()
            
            # Obtener slice
            slice_data = data[:, :, z]
            
            # Normalizar para OpenCV (0-1 a 0-255)
            min_val = np.min(slice_data)
            max_val = np.max(slice_data)
            normalized = ((slice_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            
            # Aplicar filtro bilateral
            filtered = cv2.bilateralFilter(normalized, window_size, sigma_range, sigma_space)
            
            # Desnormalizar y guardar resultado
            result[:, :, z] = (filtered / 255.0) * (max_val - min_val) + min_val
        
        return result
    
    def anisotropic_diffusion(self, data, iterations, kappa, lambda_val):
        """
        Implementación optimizada del filtro de difusión anisotrópica (Perona-Malik).
        Procesa cada slice individualmente para mejorar rendimiento.
        
        Args:
            data: Los datos de imagen a filtrar.
            iterations: Número de iteraciones.
            kappa: Parámetro de conductancia.
            lambda_val: Paso de tiempo.
            
        Returns:
            Array NumPy con la imagen filtrada.
        """
        # Crear copia de datos
        result = np.zeros_like(data)
    
        # Función de conducción
        def g(gradient, k):
            """Función de conducción de Perona-Malik (preserva bordes de alto contraste)"""
            return np.exp(-(gradient/k)**2)
    
        # Procesar cada slice por separado
        for z in range(self.parent.depth):
            # Actualizar estado
            self.parent.status_var.set(f"Difusión anisotrópica: slice {z+1}/{self.parent.depth}")
            self.parent.root.update_idletasks()
            
            # Obtener slice y copiar para procesamiento
            slice_data = data[:, :, z].copy()
            
            # Iterar por el número especificado de iteraciones
            for i in range(iterations):
                # Crear una copia temporal para la actualización
                updated = slice_data.copy()
            
                # Iterar por cada pixel en el slice (excepto bordes)
                for y in range(1, self.parent.height-1):
                    for x in range(1, self.parent.width-1):
                        # Calcular gradientes en las 4 direcciones
                        nabla_n = slice_data[x, y-1] - slice_data[x, y]
                        nabla_s = slice_data[x, y+1] - slice_data[x, y]
                        nabla_e = slice_data[x+1, y] - slice_data[x, y]
                        nabla_w = slice_data[x-1, y] - slice_data[x, y]
                    
                        # Calcular coeficientes de difusión
                        cn = g(nabla_n, kappa)
                        cs = g(nabla_s, kappa)
                        ce = g(nabla_e, kappa)
                        cw = g(nabla_w, kappa)
                    
                        # Actualizar valor actual según ecuación de difusión
                        updated[x, y] = slice_data[x, y] + lambda_val * (
                            cn * nabla_n + cs * nabla_s + 
                            ce * nabla_e + cw * nabla_w
                        )
            
                # Actualizar slice para la siguiente iteración
                slice_data = updated
            
            # Guardar resultado
            result[:, :, z] = slice_data
    
        return result
    
    def non_local_means(self, data, patch_size, search_window, h_param):
        """
        Implementa filtro Non-Local Means optimizado.
        Procesa slice por slice para reducir uso de memoria.
        
        Args:
            data: Los datos de imagen a filtrar.
            patch_size: Tamaño del parche.
            search_window: Radio de búsqueda.
            h_param: Parámetro de filtrado.
            
        Returns:
            Array NumPy con la imagen filtrada.
        """
        result = np.zeros_like(data)
        
        # Procesar cada slice por separado
        for z in range(self.parent.depth):
            # Actualizar estado
            self.parent.status_var.set(f"Aplicando NLM: slice {z+1}/{self.parent.depth}")
            self.parent.root.update_idletasks()
            
            # Obtener slice
            slice_data = data[:, :, z]
            
            # Normalizar para OpenCV (0-1 a 0-255)
            min_val = np.min(slice_data)
            max_val = np.max(slice_data)
            normalized = ((slice_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            
            # Aplicar filtro Non-Local Means
            # h controla la fuerza del filtrado, templateWindowSize es tamaño de parche,
            # searchWindowSize es el tamaño de la ventana de búsqueda
            filtered = cv2.fastNlMeansDenoising(
                normalized,
                None,
                h=h_param*100,  # Multiplicamos por 100 para escalar al rango que espera OpenCV
                templateWindowSize=patch_size,
                searchWindowSize=search_window
            )
            
            # Desnormalizar y guardar resultado
            result[:, :, z] = (filtered / 255.0) * (max_val - min_val) + min_val
        
        return result
    
    def show_preprocessing_result(self, result, filter_type):
        """
        Muestra el resultado del preprocesamiento en una nueva ventana.
        
        Args:
            result: Resultado del preprocesamiento.
            filter_type: Tipo de filtro aplicado.
        """
        # Crear una nueva ventana
        result_window = tk.Toplevel(self.parent.root)
        result_window.title(f"Resultado de Preprocesamiento: {filter_type}")
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
        self.result_slice_index = self.parent.depth // 2
    
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
    
        # Comparación lado a lado
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
    
        self.result_slice_label = ttk.Label(slice_frame, text="Corte: 0/0")
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
    
        ttk.Button(btn_frame, text="Aplicar a imagen", 
                 command=lambda: self.apply_preprocessing(result)).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Exportar a NIfTI", 
                 command=lambda: self.export_preprocessing(result)).pack(side="left", padx=5)
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
        Actualiza la visualización del corte del resultado con comparación opcional.
        
        Args:
            value: Valor del slider.
            window: Ventana de resultados.
        """
        if isinstance(value, str):
            value = float(value)
        self.result_slice_index = int(value)
    
        # Obtener el corte según la orientación
        if self.result_slice_type == "Axial":
            filtered_slice = self.result_data[:, :, self.result_slice_index]
            original_slice = self.parent.image_data[:, :, self.result_slice_index]
            max_slice = self.parent.depth - 1
        elif self.result_slice_type == "Sagittal":
            filtered_slice = self.result_data[self.result_slice_index, :, :]
            original_slice = self.parent.image_data[self.result_slice_index, :, :]
            max_slice = self.parent.width - 1
        else:  # Coronal
            filtered_slice = self.result_data[:, self.result_slice_index, :]
            original_slice = self.parent.image_data[:, self.result_slice_index, :]
            max_slice = self.parent.height - 1
    
        # Actualizar etiqueta
        self.result_slice_label.config(text=f"Corte: {self.result_slice_index}/{max_slice}")
    
        # Normalizar y aplicar colormap a ambas imágenes
        norm_filtered = self.parent.image_utils.normalize_image(filtered_slice)
        norm_original = self.parent.image_utils.normalize_image(original_slice)
        
        color_filtered = self.parent.image_utils.apply_colormap(norm_filtered)
        color_original = self.parent.image_utils.apply_colormap(norm_original)
    
        # Resize ambas imágenes
        display_size = (512, 512)
        color_filtered_resized = cv2.resize(color_filtered, display_size)
        color_original_resized = cv2.resize(color_original, display_size)
    
        # Si se seleccionó la comparación, mostrar lado a lado
        if self.show_comparison_var.get():
            # Crear imagen compuesta: original | filtrada
            combined_img = np.zeros((display_size[1], display_size[0] * 2, 3), dtype=np.uint8)
            combined_img[:, :display_size[0], :] = color_original_resized
            combined_img[:, display_size[0]:, :] = color_filtered_resized
            
            # Añadir texto
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(combined_img, "Original", (10, 30), font, 1, (255, 255, 255), 2)
            cv2.putText(combined_img, "Filtrada", (display_size[0] + 10, 30), font, 1, (255, 255, 255), 2)
            
            # Dibujar línea divisoria
            cv2.line(combined_img, (display_size[0], 0), (display_size[0], display_size[1]), (255, 255, 255), 2)
            
            # Ajustar tamaño del canvas
            self.result_canvas.config(width=display_size[0] * 2, height=display_size[1])
        else:
            # Mostrar solo imagen filtrada
            combined_img = color_filtered_resized
            self.result_canvas.config(width=display_size[0], height=display_size[1])
    
        # Mostrar en el canvas
        img_pil = Image.fromarray(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))
        img_tk = ImageTk.PhotoImage(img_pil)
    
        if hasattr(self, 'result_img_on_canvas'):
            self.result_canvas.delete(self.result_img_on_canvas)
        self.result_img_on_canvas = self.result_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.result_canvas.image = img_tk  # Mantener referencia
    
    def apply_preprocessing(self, processed_data):
        """
        Aplica el resultado del preprocesamiento como nueva imagen principal.
        
        Args:
            processed_data: Datos procesados a aplicar.
        """
        if messagebox.askyesno("Aplicar Preprocesamiento", 
                            "¿Desea aplicar el resultado como imagen principal?\n" +
                            "Esto reemplazará los datos actuales."):
            # Actualizar datos de la imagen
            self.parent.image_data = processed_data.copy()
        
            # Limpiar dibujos previos
            self.parent.overlay_data = np.zeros_like(self.parent.image_data)
            self.parent.draw_points = []
        
            # Actualizar visualización
            self.parent.visualization.update_slice()
            self.parent.status_var.set("Preprocesamiento aplicado como imagen principal")
    
    def export_preprocessing(self, result):
        """
        Exporta el resultado del preprocesamiento como archivo NIfTI.
        
        Args:
            result: Resultado del preprocesamiento.
        """
        try:
            file_path = tk.filedialog.asksaveasfilename(
                defaultextension=".nii.gz",
                filetypes=[("NIfTI Files", "*.nii *.nii.gz")],
                title="Guardar Resultado como NIfTI"
            )
        
            if not file_path:
                return
            
            # Crear un nuevo objeto NIfTI con los datos procesados
            processed_nii = nib.Nifti1Image(result, self.parent.nii_image.affine)
        
            # Guardar el archivo
            nib.save(processed_nii, file_path)
        
            self.parent.status_var.set(f"Resultado guardado en {os.path.basename(file_path)}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Error al exportar resultado: {str(e)}")