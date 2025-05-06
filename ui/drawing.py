"""
Módulo para gestionar las funcionalidades de dibujo en el visor NIfTI.
"""
import numpy as np
import cv2
from tkinter import colorchooser, messagebox
import tkinter as tk
from PIL import Image, ImageTk

class DrawingManager:
    """
    Clase que gestiona las funcionalidades de dibujo y anotación.
    """
    def __init__(self, parent):
        """
        Inicializa el administrador de dibujo.
        
        Args:
            parent: La instancia de NiftiViewer que contiene este administrador.
        """
        self.parent = parent
    
    def choose_color(self):
        """Abre un selector de color para elegir el color de dibujo."""
        color = colorchooser.askcolor(title="Elegir Color de Dibujo", initialcolor=self.parent.draw_color)
        if color[1]:  # color es una tupla ((r,g,b), hexstring)
            self.parent.draw_color = tuple(int(c) for c in color[0])
            self.parent.color_preview.config(bg=color[1])
            self.parent.status_var.set(f"Color de dibujo establecido a {color[1]}")
    
    def set_brush_size(self, size):
        """
        Establece el tamaño del pincel de dibujo.
        
        Args:
            size: Tamaño del pincel en píxeles.
        """
        self.parent.draw_radius = size
        self.parent.status_var.set(f"Tamaño del pincel establecido a {size}px")
        
        # Actualizar la variable del brushsize para sincronizar los radiobuttons
        self.parent.brush_size_var.set(size)
    
    def start_draw(self, event):
        """
        Inicia el dibujo al presionar el mouse.
        
        Args:
            event: Evento del mouse.
        """
        # Verificar si estamos en modo de selección de semilla primero
        if hasattr(self.parent, 'seed_selection_mode') and self.parent.seed_selection_mode:
            self.select_seed_point(event)
            return
            
        if not self.parent.draw_mode_var.get() or self.parent.image_data is None:
            return
        
        self.parent.drawing_mode = True
        self.parent.last_x, self.parent.last_y = event.x, event.y
        
        # Dibujar un solo punto
        self.draw(event)
    
    def draw(self, event):
        """
        Dibuja en la imagen mientras se mueve el mouse.
        
        Args:
            event: Evento del mouse.
        """
        if not self.parent.drawing_mode or self.parent.current_display_img is None:
            return
    
        # Obtener posición actual del mouse en coordenadas del canvas
        canvas_x, canvas_y = event.x, event.y
    
        # Dibujar línea entre la última posición y la posición actual en la imagen mostrada
        cv2.line(self.parent.current_display_img, 
                 (self.parent.last_x, self.parent.last_y), 
                 (canvas_x, canvas_y),
                 self.parent.draw_color, 
                 self.parent.draw_radius * 2)
    
        # Actualizar visualización
        img = Image.fromarray(self.parent.current_display_img)
        img_tk = ImageTk.PhotoImage(img)
        self.parent.canvas.itemconfig(self.parent.img_on_canvas, image=img_tk)
        self.parent.canvas.image = img_tk  # Mantener referencia
    
        # Obtener las dimensiones del corte actual
        if self.parent.corte_actual == "Axial":
            slice_width, slice_height = self.parent.width, self.parent.height
        elif self.parent.corte_actual == "Sagittal":
            slice_width, slice_height = self.parent.depth, self.parent.height
        else:  # Coronal
            slice_width, slice_height = self.parent.width, self.parent.depth
    
        # Convertir coordenadas del canvas a coordenadas del corte
        display_width, display_height = 512, 512  # Sus dimensiones de redimensionamiento
        slice_x = int(canvas_x * slice_width / display_width)
        slice_y = int(canvas_y * slice_height / display_height)
    
        # Mapear a coordenadas 3D basadas en la vista actual
        if self.parent.corte_actual == "Axial":
            x_3d, y_3d, z_3d = slice_x, slice_y, self.parent.indice_corte
        elif self.parent.corte_actual == "Sagittal":
            x_3d, y_3d, z_3d = self.parent.indice_corte, slice_y, slice_x
        else:  # Coronal
            x_3d, y_3d, z_3d = slice_x, self.parent.indice_corte, slice_y
    
        # Asegurar que las coordenadas estén dentro de los límites
        x_3d = max(0, min(x_3d, self.parent.width - 1))
        y_3d = max(0, min(y_3d, self.parent.height - 1))
        z_3d = max(0, min(z_3d, self.parent.depth - 1))
    
        # Actualizar datos de overlay
        self.parent.overlay_data[x_3d, y_3d, z_3d] = 1
    
        # Almacenar punto dibujado
        self.parent.draw_points.append({
            'x': int(x_3d),
            'y': int(y_3d),
            'z': int(z_3d),
            'color': self.parent.draw_color
        })
    
        # Actualizar visualización de coordenadas
        self.parent.coord_var.set(f"Dibujado en: x={x_3d}, y={y_3d}, z={z_3d} (Vista: {self.parent.corte_actual})")
    
        # Recordar la última posición
        self.parent.last_x, self.parent.last_y = canvas_x, canvas_y

    def stop_draw(self, event):
        """
        Detiene el dibujo al soltar el mouse.
        
        Args:
            event: Evento del mouse.
        """
        self.parent.drawing_mode = False
    
    def clear_drawings(self):
        """Borra todos los dibujos."""
        if self.parent.image_data is None:
            return
            
        if messagebox.askyesno("Borrar Dibujos", "¿Está seguro de que desea borrar todos los dibujos?"):
            self.parent.overlay_data = np.zeros_like(self.parent.image_data)
            self.parent.draw_points = []
            self.parent.visualization.update_slice()
            self.parent.status_var.set("Dibujos borrados")
            
    def enable_seed_selection(self):
        """Habilita el modo de selección de punto semilla para el crecimiento de regiones."""
        # Guardar la ventana antes de ocultarla
        if hasattr(self.parent, 'seg_window'):
            try:
                # Modificar el título de la ventana para mayor claridad
                self.parent.seg_window.title("ESPERANDO SELECCIÓN - Haga clic en la imagen principal")
                # Cambiar el color de fondo para destacar que está en modo especial
                self.parent.seg_window.configure(background='yellow')
                
                # Verificar si la ventana existe antes de ocultarla
                if self.parent.seg_window.winfo_exists():
                    self.parent.seg_window.withdraw()  # Oculta la ventana de opciones temporalmente
            except Exception as e:
                print(f"Error al preparar ventana para selección de semilla: {e}")
        
        # Activar el modo de selección de semilla
        self.parent.seed_selection_mode = True
        
        # Cambiar el cursor para indicar que estamos en modo de selección
        self.parent.canvas.config(cursor="crosshair")
        
        # Mensaje claro en la barra de estado
        self.parent.status_var.set("¡MODO SELECCIÓN DE SEMILLA ACTIVO! - Haga clic en un punto de la imagen para seleccionar la semilla")
        
        # Actualizar la interfaz inmediatamente
        self.parent.root.update_idletasks()
        
        # Opcionalmente, mostrar un mensaje de información
        messagebox.showinfo("Selección de Semilla", 
                          "Ahora haga clic en la imagen principal para seleccionar un punto semilla.\n\n"
                          "El punto que seleccione será el punto inicial para el crecimiento de región.")
    
    def select_seed_point(self, event):
        """
        Captura el punto semilla seleccionado para el crecimiento de regiones.
        
        Args:
            event: Evento del mouse.
        """
        try:
            # Convertir coordenadas del canvas a coordenadas del volumen
            canvas_x, canvas_y = event.x, event.y
        
            # Obtener dimensiones de la visualización actual
            if self.parent.corte_actual == "Axial":
                slice_width, slice_height = self.parent.width, self.parent.height
            elif self.parent.corte_actual == "Sagittal":
                slice_width, slice_height = self.parent.depth, self.parent.height
            else:  # Coronal
                slice_width, slice_height = self.parent.width, self.parent.depth
        
            # Convertir a coordenadas de la imagen
            display_width, display_height = 512, 512  # Dimensiones tras el resize
            slice_x = int(canvas_x * slice_width / display_width)
            slice_y = int(canvas_y * slice_height / display_height)
        
            # Convertir a coordenadas 3D
            if self.parent.corte_actual == "Axial":
                x_3d, y_3d, z_3d = slice_x, slice_y, self.parent.indice_corte
            elif self.parent.corte_actual == "Sagittal":
                x_3d, y_3d, z_3d = self.parent.indice_corte, slice_y, slice_x
            else:  # Coronal
                x_3d, y_3d, z_3d = slice_x, self.parent.indice_corte, slice_y
        
            # Asegurar que las coordenadas estén dentro de los límites
            x_3d = max(0, min(x_3d, self.parent.width - 1))
            y_3d = max(0, min(y_3d, self.parent.height - 1))
            z_3d = max(0, min(z_3d, self.parent.depth - 1))
            
            self.parent.seed_point = (x_3d, y_3d, z_3d)
            
            # Mostrar marcador
            self.draw_seed_marker(x_3d, y_3d, z_3d)
        
            # Desactivar el modo de selección de semilla
            self.parent.seed_selection_mode = False
            
            # Restaurar el cursor normal
            self.parent.canvas.config(cursor="")
            
            # Mensaje claro de confirmación
            mensaje = f"Punto semilla seleccionado en: ({x_3d}, {y_3d}, {z_3d})"
            self.parent.status_var.set(mensaje)
        
            # Mostrar la ventana de opciones nuevamente
            if hasattr(self.parent, 'seg_window'):
                try:
                    # Restaurar el título y color de fondo
                    self.parent.seg_window.title("Opciones de Crecimiento")
                    self.parent.seg_window.configure(background='')
                    
                    # Mostrar la ventana
                    if self.parent.seg_window.winfo_exists():
                        self.parent.seg_window.deiconify()
                        
                        # Actualizar la interfaz para mostrar el punto seleccionado
                        for widget in self.parent.seg_window.winfo_children():
                            if isinstance(widget, tk.Frame):
                                # Buscar y eliminar label de semilla si existe
                                for child in widget.winfo_children():
                                    if isinstance(child, tk.Label) and "Semilla:" in str(child.cget("text")):
                                        child.destroy()
                                
                                # Agregar nuevo label con información de la semilla
                                seed_label = tk.Label(widget, 
                                                    text=f"Semilla: ({x_3d}, {y_3d}, {z_3d})",
                                                    fg="green", font=("Arial", 10, "bold"))
                                seed_label.grid(row=5, column=0, columnspan=3, sticky="w", pady=5)
                                break
                except Exception as e:
                    print(f"Error al restaurar ventana de opciones: {e}")
                    # Si falla, crear una nueva ventana de opciones
                    messagebox.showinfo("Semilla Seleccionada", mensaje)
        
            self.parent.root.update_idletasks()
            
        except Exception as e:
            self.parent.seed_selection_mode = False
            self.parent.canvas.config(cursor="")
            messagebox.showerror("Error", f"Error al seleccionar punto semilla: {str(e)}")
        
    def select_automatic_seed(self):
        """Selecciona automáticamente un punto semilla en el centro del volumen."""
        if self.parent.image_data is None:
            messagebox.showinfo("Sin datos", "Por favor cargue una imagen primero.")
            return
            
        # Seleccionar el centro del volumen como semilla
        x = self.parent.width // 2
        y = self.parent.height // 2
        z = self.parent.depth // 2
        
        self.parent.seed_point = (x, y, z)
        
        # Actualizar la vista para mostrar el corte donde está la semilla
        self.parent.corte_actual = "Axial"
        self.parent.indice_corte = z
        self.parent.slice_slider.set(z)
        self.parent.visualization.update_slice()
        
        # Mostrar marcador
        self.draw_seed_marker(x, y, z)
        
        # Actualizar ventana de opciones
        if hasattr(self.parent, 'seg_window'):
            for widget in self.parent.seg_window.winfo_children():
                if isinstance(widget, tk.Frame):
                    # Buscar y eliminar label de semilla si existe
                    for child in widget.winfo_children():
                        if isinstance(child, tk.Label) and "Semilla:" in str(child.cget("text")):
                            child.destroy()
                    
                    # Agregar nuevo label con información de la semilla
                    seed_label = tk.Label(widget, 
                                        text=f"Semilla: ({x}, {y}, {z}) - Centro",
                                        fg="green", font=("Arial", 10, "bold"))
                    seed_label.grid(row=5, column=0, columnspan=3, sticky="w", pady=5)
                    break
        
        self.parent.status_var.set(f"Semilla automática seleccionada en el centro: ({x}, {y}, {z})")
        
    def draw_seed_marker(self, x, y, z):
        """
        Dibuja un marcador en la posición seleccionada para el punto semilla.
        
        Args:
            x: Coordenada x del punto semilla.
            y: Coordenada y del punto semilla.
            z: Coordenada z del punto semilla.
        """
        # Convertir coordenadas 3D a coordenadas de pantalla según la vista actual
        if self.parent.corte_actual == "Axial" and z == self.parent.indice_corte:
            display_x = int(x * 512 / self.parent.width)
            display_y = int(y * 512 / self.parent.height)
        elif self.parent.corte_actual == "Sagittal" and x == self.parent.indice_corte:
            display_x = int(z * 512 / self.parent.depth)
            display_y = int(y * 512 / self.parent.height)
        elif self.parent.corte_actual == "Coronal" and y == self.parent.indice_corte:
            display_x = int(x * 512 / self.parent.width)
            display_y = int(z * 512 / self.parent.depth)
        else:
            # Si la semilla no está en el corte actual, mostrar un mensaje y cambiar al corte adecuado
            if self.parent.corte_actual == "Axial":
                self.parent.indice_corte = z
            elif self.parent.corte_actual == "Sagittal":
                self.parent.indice_corte = x
            else:  # Coronal
                self.parent.indice_corte = y
                
            self.parent.slice_slider.set(self.parent.indice_corte)
            self.parent.visualization.update_slice()
            
            # Recalcular coordenadas de pantalla
            if self.parent.corte_actual == "Axial":
                display_x = int(x * 512 / self.parent.width)
                display_y = int(y * 512 / self.parent.height)
            elif self.parent.corte_actual == "Sagittal":
                display_x = int(z * 512 / self.parent.depth)
                display_y = int(y * 512 / self.parent.height)
            else:  # Coronal
                display_x = int(x * 512 / self.parent.width)
                display_y = int(z * 512 / self.parent.depth)

        # Dibujar un marcador más visible y distintivo
        # Círculo exterior
        outer_circle_id = self.parent.canvas.create_oval(
            display_x-10, display_y-10,
            display_x+10, display_y+10,
            outline="white", width=2
        )
        
        # Círculo interior
        inner_circle_id = self.parent.canvas.create_oval(
            display_x-5, display_y-5,
            display_x+5, display_y+5,
            outline="red", width=2, fill="red"
        )
        
        # Cruz para marcar el centro
        line1_id = self.parent.canvas.create_line(
            display_x-7, display_y, display_x+7, display_y,
            fill="white", width=2
        )
        
        line2_id = self.parent.canvas.create_line(
            display_x, display_y-7, display_x, display_y+7,
            fill="white", width=2
        )
        
        # Texto que indique "SEMILLA"
        text_id = self.parent.canvas.create_text(
            display_x, display_y-20,
            text="SEMILLA",
            fill="yellow",
            font=("Arial", 10, "bold")
        )
        
        # Eliminar los marcadores después de 5 segundos para no estorbar
        marker_ids = [outer_circle_id, inner_circle_id, line1_id, line2_id, text_id]
        self.parent.root.after(5000, lambda: [self.parent.canvas.delete(id) for id in marker_ids])