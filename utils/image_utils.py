"""
Módulo de utilidades para procesamiento de imágenes en la aplicación NIfTI Viewer.
Proporciona funciones para normalización, filtrado y manipulación de imágenes.
"""
import numpy as np
import cv2

class ImageUtils:
    """
    Clase para gestionar operaciones comunes de procesamiento de imágenes.
    """
    def __init__(self):
        """Inicializa el objeto de utilidades de imagen."""
        pass
    
    def normalize_image(self, img):
        """
        Normaliza una imagen al rango 0-255.
        
        Args:
            img: Array NumPy con datos de imagen.
            
        Returns:
            Array NumPy con la imagen normalizada al rango 0-255.
        """
        min_val = np.min(img)
        max_val = np.max(img)
        if max_val == min_val:
            return np.zeros_like(img, dtype=np.uint8)
        return ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    
    def apply_colormap(self, img):
        """
        Aplica un mapa de colores (tipo CT) a la imagen.
        
        Args:
            img: Array NumPy con datos de imagen normalizada.
            
        Returns:
            Array NumPy con la imagen coloreada.
        """
        return cv2.applyColorMap(img, cv2.COLORMAP_BONE)
    
    def resize_image(self, img, target_size=(512, 512)):
        """
        Redimensiona una imagen al tamaño objetivo.
        
        Args:
            img: Array NumPy con datos de imagen.
            target_size: Tamaño objetivo como tupla (ancho, alto).
            
        Returns:
            Array NumPy con la imagen redimensionada.
        """
        # Verificar que la imagen no esté vacía
        if img is None or img.size == 0:
            return np.zeros((*target_size, 3) if len(img.shape) == 3 else target_size, dtype=np.uint8)
            
        # Verificar el tipo de datos
        if not isinstance(img, np.ndarray):
            img = np.array(img)
            
        # Convertir a uint8 si es necesario
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
            
        # Ajustar las dimensiones si es necesario
        if len(img.shape) == 2:
            return cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        elif len(img.shape) == 3:
            return cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        else:
            raise ValueError(f"Formato de imagen no soportado: {img.shape}")
    
    def normalize_0_1(self, data):
        """
        Normaliza datos al rango [0-1].
        
        Args:
            data: Datos a normalizar.
            
        Returns:
            Array NumPy con datos normalizados.
        """
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val == min_val:
            return np.zeros_like(data, dtype=np.float32)
        return ((data - min_val) / (max_val - min_val)).astype(np.float32)
    
    def create_overlay(self, base_img, overlay_img, alpha=0.7):
        """
        Superpone una imagen sobre otra con transparencia.
        
        Args:
            base_img: Imagen base.
            overlay_img: Imagen a superponer.
            alpha: Nivel de opacidad (0-1).
            
        Returns:
            Array NumPy con la imagen combinada.
        """
        if len(base_img.shape) == 2:
            # Convertir imagen base a RGB si es en escala de grises
            base_rgb = cv2.cvtColor(base_img, cv2.COLOR_GRAY2RGB)
        else:
            base_rgb = base_img.copy()
            
        if len(overlay_img.shape) == 2:
            # Convertir overlay a RGB si es en escala de grises
            overlay_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_GRAY2RGB)
        else:
            overlay_rgb = overlay_img.copy()
            
        # Asegurar que ambas imágenes tengan el mismo tamaño
        if base_rgb.shape[:2] != overlay_rgb.shape[:2]:
            overlay_rgb = cv2.resize(overlay_rgb, (base_rgb.shape[1], base_rgb.shape[0]))
            
        # Crear máscara para áreas con contenido en el overlay
        mask = (overlay_rgb > 0).any(axis=2)
        mask_3d = np.stack([mask, mask, mask], axis=2)
        
        # Combinar imágenes
        result = np.where(mask_3d, cv2.addWeighted(base_rgb, 1-alpha, overlay_rgb, alpha, 0), base_rgb)
        
        return result
    
    def apply_window_level(self, img, window_width, window_center):
        """
        Aplica ajuste de ventana/nivel (window/level) a una imagen médica.
        
        Args:
            img: Array NumPy con datos de imagen.
            window_width: Ancho de la ventana.
            window_center: Centro de la ventana.
            
        Returns:
            Array NumPy con la imagen ajustada.
        """
        # Calcular límites de la ventana
        low = window_center - window_width / 2
        high = window_center + window_width / 2
        
        # Aplicar ventana
        result = np.clip(img, low, high)
        
        # Normalizar al rango 0-255
        result = ((result - low) / (high - low) * 255).astype(np.uint8)
        
        return result
    
    def create_montage(self, slices, grid_size=None, padding=2):
        """
        Crea un montaje de múltiples cortes en una sola imagen.
        
        Args:
            slices: Lista de arrays NumPy con imágenes.
            grid_size: Tamaño de la cuadrícula como tupla (filas, columnas).
            padding: Espacio entre imágenes.
            
        Returns:
            Array NumPy con el montaje de imágenes.
        """
        if not slices:
            return np.zeros((100, 100), dtype=np.uint8)
            
        # Determinar el tamaño de la cuadrícula si no se proporciona
        if grid_size is None:
            n = len(slices)
            grid_size = (int(np.ceil(np.sqrt(n))), int(np.ceil(np.sqrt(n))))
        
        rows, cols = grid_size
        
        # Asegurar que todas las imágenes tengan el mismo tamaño
        slice_height, slice_width = slices[0].shape[:2]
        
        # Crear imagen para el montaje
        montage_height = rows * slice_height + (rows - 1) * padding
        montage_width = cols * slice_width + (cols - 1) * padding
        
        # Determinar el número de canales
        if len(slices[0].shape) == 3:
            channels = slices[0].shape[2]
            montage = np.zeros((montage_height, montage_width, channels), dtype=np.uint8)
        else:
            montage = np.zeros((montage_height, montage_width), dtype=np.uint8)
        
        # Colocar cada slice en el montaje
        slice_idx = 0
        for row in range(rows):
            for col in range(cols):
                if slice_idx >= len(slices):
                    break
                    
                y_start = row * (slice_height + padding)
                y_end = y_start + slice_height
                x_start = col * (slice_width + padding)
                x_end = x_start + slice_width
                
                montage[y_start:y_end, x_start:x_end] = slices[slice_idx]
                slice_idx += 1
        
        return montage
    
    def add_text_to_image(self, img, text, position, font_scale=1.0, color=(255, 255, 255), thickness=2):
        """
        Añade texto a una imagen.
        
        Args:
            img: Array NumPy con la imagen.
            text: Texto a añadir.
            position: Posición como tupla (x, y).
            font_scale: Escala de la fuente.
            color: Color del texto como tupla (B, G, R).
            thickness: Grosor del texto.
            
        Returns:
            Array NumPy con la imagen con el texto añadido.
        """
        # Crear una copia de la imagen
        result = img.copy()
        
        # Añadir texto
        cv2.putText(result, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale, color, thickness, cv2.LINE_AA)
        
        return result
    
    def draw_rectangle(self, img, top_left, bottom_right, color=(255, 0, 0), thickness=2):
        """
        Dibuja un rectángulo en una imagen.
        
        Args:
            img: Array NumPy con la imagen.
            top_left: Esquina superior izquierda como tupla (x, y).
            bottom_right: Esquina inferior derecha como tupla (x, y).
            color: Color del rectángulo como tupla (B, G, R).
            thickness: Grosor de la línea.
            
        Returns:
            Array NumPy con la imagen con el rectángulo dibujado.
        """
        # Crear una copia de la imagen
        result = img.copy()
        
        # Dibujar rectángulo
        cv2.rectangle(result, top_left, bottom_right, color, thickness)
        
        return result
    
    def draw_circle(self, img, center, radius, color=(255, 0, 0), thickness=2):
        """
        Dibuja un círculo en una imagen.
        
        Args:
            img: Array NumPy con la imagen.
            center: Centro del círculo como tupla (x, y).
            radius: Radio del círculo.
            color: Color del círculo como tupla (B, G, R).
            thickness: Grosor de la línea (-1 para círculo relleno).
            
        Returns:
            Array NumPy con la imagen con el círculo dibujado.
        """
        # Crear una copia de la imagen
        result = img.copy()
        
        # Dibujar círculo
        cv2.circle(result, center, radius, color, thickness)
        
        return result
    
    def extract_roi(self, img, top_left, bottom_right):
        """
        Extrae una región de interés (ROI) de una imagen.
        
        Args:
            img: Array NumPy con la imagen.
            top_left: Esquina superior izquierda como tupla (x, y).
            bottom_right: Esquina inferior derecha como tupla (x, y).
            
        Returns:
            Array NumPy con la región extraída.
        """
        x1, y1 = top_left
        x2, y2 = bottom_right
        
        # Asegurar que las coordenadas están dentro de los límites
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1], x2)
        y2 = min(img.shape[0], y2)
        
        return img[y1:y2, x1:x2].copy()