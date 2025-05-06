"""
Módulo de utilidades para operaciones de entrada/salida en la aplicación NIfTI Viewer.
Maneja la carga y guardado de archivos NIfTI y datos de dibujo.
"""
import os
import json
import nibabel as nib
import numpy as np
from tkinter import filedialog, messagebox, ttk

class IOUtils:
    """
    Clase para gestionar operaciones de entrada/salida.
    """
    def __init__(self):
        """Inicializa el objeto de utilidades IO."""
        pass
    
    def load_image(self, parent):
        """
        Carga un archivo de imagen NIfTI.
        
        Args:
            parent: Instancia de NiftiViewer.
        """
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("NIfTI Files", "*.nii *.nii.gz")],
                title="Seleccionar Archivo de Imagen NIfTI"
            )
            
            if not file_path:
                return
            
            parent.status_var.set("Cargando imagen...")
            parent.root.update_idletasks()
            
            parent.file_path = file_path
            parent.nii_image = nib.load(file_path)
            parent.image_data = parent.nii_image.get_fdata()
            
            # Obtener dimensiones
            parent.width, parent.height, parent.depth = parent.image_data.shape
            
            # Inicializar datos de overlay
            parent.overlay_data = np.zeros_like(parent.image_data)
            
            # Limpiar puntos de dibujo almacenados
            parent.draw_points = []
            
            # Actualizar UI
            filename = os.path.basename(file_path)
            parent.label_info.config(text=f"Archivo: {filename}\nDimensiones: {parent.width}×{parent.height}×{parent.depth}")
            
            # Habilitar controles
            self._enable_controls(parent)
            
            # Establecer vista predeterminada
            parent.visualization.change_slice_type("Axial")
            parent.status_var.set(f"Cargado: {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar imagen: {str(e)}")
            parent.status_var.set("Error al cargar imagen")
    
    def _enable_controls(self, parent):
        """
        Habilita los controles de la interfaz después de cargar una imagen.
        
        Args:
            parent: Instancia de NiftiViewer.
        """
        # Habilitar botones
        parent.btn_axial.state(["!disabled"])
        parent.btn_sagittal.state(["!disabled"])
        parent.btn_coronal.state(["!disabled"])
        parent.btn_3d.state(["!disabled"])
        parent.slice_slider.state(["!disabled"])
        parent.btn_color.state(["!disabled"])
        parent.btn_clear.state(["!disabled"])
        parent.chk_draw.state(["!disabled"])
        parent.btn_save_drawings.state(["!disabled"])
        
        # Habilitar radiobuttons de tamaño de pincel si existen
        for widget in parent.root.winfo_children():
            if isinstance(widget, ttk.Radiobutton):
                widget.state(["!disabled"])
        
        # Habilitar ítems de menú
        parent.viewmenu.entryconfig("Vista Axial", state="normal")
        parent.viewmenu.entryconfig("Vista Sagital", state="normal")
        parent.viewmenu.entryconfig("Vista Coronal", state="normal")
        parent.viewmenu.entryconfig("Visualización 3D", state="normal")
        
        # Habilitar menús de segmentación y preprocesamiento
        for i in range(parent.segmenu.index("end") + 1):
            parent.segmenu.entryconfig(i, state="normal")
            
        for i in range(parent.prepmenu.index("end") + 1):
            parent.prepmenu.entryconfig(i, state="normal")
        
        # Habilitar ítems de menú de dibujo
        parent.drawmenu.entryconfig("Cambiar color de dibujo", state="normal")
        parent.drawmenu.entryconfig("Borrar dibujos", state="normal")
        parent.sizemenu.entryconfig("Pequeño (1px)", state="normal")
        parent.sizemenu.entryconfig("Medio (3px)", state="normal")
        parent.sizemenu.entryconfig("Grande (5px)", state="normal")
        parent.filemenu.entryconfig("Guardar dibujos", state="normal")
        parent.filemenu.entryconfig("Cargar dibujos", state="normal")
    
    def save_drawings(self, parent):
        """
        Guarda los puntos de dibujo en un archivo JSON.
        
        Args:
            parent: Instancia de NiftiViewer.
        """
        if not parent.draw_points:
            messagebox.showinfo("Sin Dibujos", "No hay dibujos para guardar.")
            return
            
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON Files", "*.json")],
                title="Guardar Puntos de Dibujo"
            )
            
            if not file_path:
                return
                
            with open(file_path, 'w') as f:
                json.dump({
                    'original_image': os.path.basename(parent.file_path),
                    'dimensions': [parent.width, parent.height, parent.depth],
                    'points': parent.draw_points
                }, f, indent=2)
                
            parent.status_var.set(f"Dibujos guardados en {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al guardar dibujos: {str(e)}")
    
    def load_drawings(self, parent):
        """
        Carga puntos de dibujo desde un archivo JSON.
        
        Args:
            parent: Instancia de NiftiViewer.
        """
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("JSON Files", "*.json")],
                title="Cargar Puntos de Dibujo"
            )
            
            if not file_path:
                return
                
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Verificar que las dimensiones coinciden
            if data['dimensions'] != [parent.width, parent.height, parent.depth]:
                messagebox.showwarning(
                    "Dimensiones No Coinciden",
                    "Las dimensiones de los dibujos guardados no coinciden con la imagen actual."
                )
                return
                
            # Cargar puntos
            parent.draw_points = data['points']
            
            # Recrear datos de overlay
            parent.overlay_data = np.zeros_like(parent.image_data)
            parent.overlay_colors = {}

            for point in parent.draw_points:
                x, y, z = point['x'], point['y'], point['z']
                color = point.get('color', parent.draw_color)
                if 0 <= x < parent.width and 0 <= y < parent.height and 0 <= z < parent.depth:
                    parent.overlay_data[x, y, z] = 1
                    parent.overlay_colors[(x,y,z)] = color
                    
            # Actualizar visualización
            parent.visualization.update_slice()
            parent.status_var.set(f"Dibujos cargados desde {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar dibujos: {str(e)}")
    
    def export_volume_as_nifti(self, parent, data, affine=None, filename=None):
        """
        Exporta un volumen como archivo NIfTI.
        
        Args:
            parent: Instancia de NiftiViewer.
            data: Datos a exportar.
            affine: Matriz de transformación. Si es None, se usa la de la imagen original.
            filename: Nombre de archivo. Si es None, se solicita al usuario.
        
        Returns:
            bool: True si se exportó correctamente, False en caso contrario.
        """
        try:
            # Si no se proporciona nombre de archivo, solicitar al usuario
            if filename is None:
                filename = filedialog.asksaveasfilename(
                    defaultextension=".nii.gz",
                    filetypes=[("NIfTI Files", "*.nii *.nii.gz")],
                    title="Guardar como NIfTI"
                )
                
                if not filename:
                    return False
            
            # Si no se proporciona matriz de transformación, usar la de la imagen original
            if affine is None and parent.nii_image is not None:
                affine = parent.nii_image.affine
            elif affine is None:
                # Crear una matriz de identidad si no hay imagen original
                affine = np.eye(4)
            
            # Crear objeto NIfTI
            nii_img = nib.Nifti1Image(data, affine)
            
            # Guardar archivo
            nib.save(nii_img, filename)
            
            parent.status_var.set(f"Archivo guardado como {os.path.basename(filename)}")
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al exportar archivo: {str(e)}")
            return False