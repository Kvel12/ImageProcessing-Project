"""
Módulo para gestionar la visualización 2D y 3D de imágenes NIfTI.
"""
import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk
from PIL import Image, ImageTk
import cv2
import tkinter as tk
from tkinter import messagebox

class VisualizationManager:
    """
    Clase que gestiona la visualización de imágenes NIfTI en 2D y 3D.
    """
    def __init__(self, parent):
        """
        Inicializa el administrador de visualización.
        
        Args:
            parent: La instancia de NiftiViewer que contiene este administrador.
        """
        self.parent = parent
    
    def change_slice_type(self, slice_type):
        """
        Cambia el tipo de corte (Axial, Sagital, Coronal).
        
        Args:
            slice_type: El tipo de corte a mostrar.
        """
        if self.parent.image_data is None:
            return
        
        self.parent.corte_actual = slice_type
        
        # Actualizar el rango del slider basado en la vista
        if slice_type == "Axial":
            max_slice = self.parent.depth - 1
            self.parent.indice_corte = self.parent.depth // 2
        elif slice_type == "Sagittal":
            max_slice = self.parent.width - 1
            self.parent.indice_corte = self.parent.width // 2
        else:  # Coronal
            max_slice = self.parent.height - 1
            self.parent.indice_corte = self.parent.height // 2
        
        self.parent.slice_slider.config(from_=0, to=max_slice)
        self.parent.slice_slider.set(self.parent.indice_corte)
        self.update_slice()
    
    def update_slice(self, *args):
        """
        Actualiza el corte mostrado.
        
        Args:
            *args: Argumentos opcionales pasados desde el slider.
        """
        if self.parent.image_data is None:
            return
        
        try:
            self.parent.indice_corte = int(self.parent.slice_slider.get())
            
            # Obtener el corte correcto según la orientación
            if self.parent.corte_actual == "Axial":
                slice_data = self.parent.image_data[:, :, self.parent.indice_corte]
                overlay_slice = self.parent.overlay_data[:, :, self.parent.indice_corte]
                max_slice = self.parent.depth - 1
            elif self.parent.corte_actual == "Sagittal":
                slice_data = self.parent.image_data[self.parent.indice_corte, :, :]
                overlay_slice = self.parent.overlay_data[self.parent.indice_corte, :, :]
                max_slice = self.parent.width - 1
            else:  # Coronal
                slice_data = self.parent.image_data[:, self.parent.indice_corte, :]
                overlay_slice = self.parent.overlay_data[:, self.parent.indice_corte, :]
                max_slice = self.parent.height - 1
            
            # Actualizar etiqueta del corte
            self.parent.slice_label.config(text=f"Corte: {self.parent.indice_corte}/{max_slice}")
            
            # Procesar la imagen
            normalized = self.parent.image_utils.normalize_image(slice_data)
            colormap = self.parent.image_utils.apply_colormap(normalized)
            
            # Mezclar con la capa de dibujo (overlay)
            overlay_normalized = (overlay_slice > 0).astype(np.uint8) * 255
            overlay_rgb = np.zeros((*overlay_normalized.shape, 3), dtype=np.uint8)
            
            # Dibujar los puntos con sus colores
            for point in self.parent.draw_points:
                x, y, z = point['x'], point['y'], point['z']
                color = point.get('color', self.parent.draw_color)  # Obtener el color del punto
                if self.parent.corte_actual == "Axial":
                    if z == self.parent.indice_corte:
                        overlay_rgb[y, x] = color  # Vista Axial usa x, y
                elif self.parent.corte_actual == "Sagittal":
                    if x == self.parent.indice_corte:
                        overlay_rgb[y, z] = color  # Vista Sagital usa y, z
                else:  # Coronal
                    if y == self.parent.indice_corte:
                        overlay_rgb[x, z] = color  # Vista Coronal usa x, z
            
            # Redimensionar ambas
            resized = self.parent.image_utils.resize_image(colormap)
            overlay_resized = self.parent.image_utils.resize_image(overlay_rgb)
            
            # Mezclar imágenes
            alpha = 0.7
            mask = (overlay_resized > 0).any(axis=2)
            mask_3d = np.stack([mask, mask, mask], axis=2)
            combined = np.where(mask_3d, cv2.addWeighted(resized, 1-alpha, overlay_resized, alpha, 0), resized)
            
            # Guardar imagen actual para dibujo
            self.parent.current_display_img = combined.copy()
            
            # Convertir a PIL Image y mostrar
            img = Image.fromarray(combined)
            img_tk = ImageTk.PhotoImage(img)
            
            # Actualizar canvas
            self.parent.canvas.config(width=img.width, height=img.height)
            if hasattr(self.parent, 'img_on_canvas'):
                self.parent.canvas.delete(self.parent.img_on_canvas)
            self.parent.img_on_canvas = self.parent.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.parent.canvas.image = img_tk  # Mantener referencia
            
        except Exception as e:
            self.parent.status_var.set(f"Error al actualizar corte: {str(e)}")
    
    def visualize_3d(self):
        """
        Crea una visualización 3D de los datos NIfTI usando VTK.
        """
        if self.parent.image_data is None:
            messagebox.showinfo("Sin datos", "Por favor cargue una imagen NIfTI primero.")
            return
    
        try:
            self.parent.status_var.set("Creando visualización 3D...")
            self.parent.root.update_idletasks()
        
            # Crear una copia de los datos para procesamiento
            volume_data = self.parent.image_data.copy()
        
            # Normalizar datos al rango 0-255
            volume_min = np.min(volume_data)
            volume_max = np.max(volume_data)
            volume_data = ((volume_data - volume_min) / (volume_max - volume_min) * 255).astype(np.uint8)
        
            # Crear VTK image data
            volume = vtk.vtkImageData()
            volume.SetDimensions(self.parent.width, self.parent.height, self.parent.depth)
            volume.SetSpacing(1.0, 1.0, 1.0)
            volume.SetOrigin(0.0, 0.0, 0.0)
            volume.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        
            # Llenar la imagen VTK con datos
            vtk_data = numpy_to_vtk(volume_data.flatten(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
            volume.GetPointData().GetScalars().DeepCopy(vtk_data)
        
            # Añadir datos de overlay si están disponibles (regiones dibujadas)
            if np.any(self.parent.overlay_data > 0):
                # Crear una máscara de las regiones dibujadas
                overlay_mask = (self.parent.overlay_data > 0).astype(np.uint8) * 255
                # Dilatar ligeramente para que sea más visible en 3D
                kernel = np.ones((3, 3, 3), np.uint8)
                overlay_mask = np.array([cv2.dilate(overlay_mask[:, :, i], kernel[:, :, 1], iterations=1) 
                                        for i in range(overlay_mask.shape[2])]).transpose(1, 2, 0)
            
                # Mezclar overlay con datos de volumen
                r, g, b = self.parent.draw_color
                color_factor = 0.8  # Intensidad del color
                for i in range(overlay_mask.shape[0]):
                    for j in range(overlay_mask.shape[1]):
                        for k in range(overlay_mask.shape[2]):
                            if overlay_mask[i, j, k] > 0:
                                # Mezclar color con intensidad original
                                orig_val = volume_data[i, j, k]
                                volume_data[i, j, k] = int(orig_val * (1 - color_factor) + 
                                                        (r + g + b) / 3 * color_factor)
            
                # Actualizar datos del volumen con overlay
                vtk_data = numpy_to_vtk(volume_data.flatten(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
                volume.GetPointData().GetScalars().DeepCopy(vtk_data)
        
            # Crear mapeador y propiedad de volumen
            volume_mapper = vtk.vtkSmartVolumeMapper()
            volume_mapper.SetInputData(volume)
        
            volume_property = vtk.vtkVolumeProperty()
            volume_property.ShadeOn()
            volume_property.SetInterpolationTypeToLinear()
        
            # Crear y establecer funciones de transferencia para apariencia tipo CT
            color_function = vtk.vtkColorTransferFunction()
            opacity_function = vtk.vtkPiecewiseFunction()
        
            # Configurar función de transferencia de color (escala de grises tipo CT)
            color_function.AddRGBPoint(0, 0.0, 0.0, 0.0)      # Negro para aire/fondo
            color_function.AddRGBPoint(50, 0.3, 0.3, 0.3)     # Gris oscuro para tejido blando
            color_function.AddRGBPoint(150, 0.8, 0.8, 0.8)    # Gris claro para hueso
            color_function.AddRGBPoint(255, 1.0, 1.0, 1.0)    # Blanco para hueso denso
        
            # Añadir color para overlay (si hay regiones dibujadas)
            if np.any(self.parent.overlay_data > 0):
                # Añadir pista de color para las regiones dibujadas
                r, g, b = self.parent.draw_color
                color_function.AddRGBPoint(200, r/255, g/255, b/255)
        
            # Configurar función de transferencia de opacidad
            opacity_function.AddPoint(0, 0.0)     # Completamente transparente para fondo
            opacity_function.AddPoint(40, 0.0)    # Todavía transparente para aire
            opacity_function.AddPoint(80, 0.2)    # Ligeramente visible para tejido blando
            opacity_function.AddPoint(150, 0.4)   # Más opaco para hueso
            opacity_function.AddPoint(255, 0.8)   # Más opaco para hueso denso
        
            # Si hay regiones dibujadas, hacerlas más visibles
            if np.any(self.parent.overlay_data > 0):
                opacity_function.AddPoint(200, 0.9)  # Hacer que las regiones dibujadas sean muy visibles
        
            # Establecer las funciones de color y opacidad
            volume_property.SetColor(color_function)
            volume_property.SetScalarOpacity(opacity_function)
        
            # Establecer la opacidad del gradiente para mejora de bordes
            gradient_opacity = vtk.vtkPiecewiseFunction()
            gradient_opacity.AddPoint(0, 0.0)
            gradient_opacity.AddPoint(90, 0.5)
            gradient_opacity.AddPoint(255, 1.0)
            volume_property.SetGradientOpacity(gradient_opacity)
        
            # Crear el volumen
            actor_volume = vtk.vtkVolume()
            actor_volume.SetMapper(volume_mapper)
            actor_volume.SetProperty(volume_property)
        
            # Crear ventana de renderizado y renderizador
            renderer = vtk.vtkRenderer()
            renderer.SetBackground(0.1, 0.1, 0.1)  # Fondo oscuro
        
            render_window = vtk.vtkRenderWindow()
            render_window.AddRenderer(renderer)
            render_window.SetSize(800, 600)
            render_window.SetWindowName(f"Visualización 3D: {self.parent.file_path}")
        
            # Añadir el volumen al renderizador
            renderer.AddVolume(actor_volume)
        
            # Configurar cámara para una buena vista inicial
            camera = renderer.GetActiveCamera()
            camera.SetPosition(0, -400, 0)
            camera.SetFocalPoint(0, 0, 0)
            camera.SetViewUp(0, 0, 1)
            renderer.ResetCamera()
        
            # Crear interacción
            interactor = vtk.vtkRenderWindowInteractor()
            interactor.SetRenderWindow(render_window)
        
            # Añadir marcador de orientación (ejes)
            axes = vtk.vtkAxesActor()
            axes.SetTotalLength(50, 50, 50)
            axes.SetXAxisLabelText("X")
            axes.SetYAxisLabelText("Y")
            axes.SetZAxisLabelText("Z")
            axes.GetXAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
            axes.GetYAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
            axes.GetZAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        
            axes_widget = vtk.vtkOrientationMarkerWidget()
            axes_widget.SetOrientationMarker(axes)
            axes_widget.SetInteractor(interactor)
            axes_widget.SetViewport(0.0, 0.0, 0.2, 0.2)
            axes_widget.SetEnabled(1)
            axes_widget.InteractiveOff()
        
            # Configurar estilo de interacción
            style = vtk.vtkInteractorStyleTrackballCamera()
            interactor.SetInteractorStyle(style)
        
            # Añadir visualización de texto para información
            text_actor = vtk.vtkTextActor()
            text_actor.SetInput(f"Archivo: {self.parent.file_path}\n"
                              f"Dimensiones: {self.parent.width}x{self.parent.height}x{self.parent.depth}\n"
                              f"Use el ratón para rotar, Ctrl+ratón para mover, Scroll para zoom")
            text_actor.GetTextProperty().SetFontSize(12)
            text_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
            text_actor.SetPosition(10, 10)
            renderer.AddActor2D(text_actor)
        
            # Inicializar y comenzar el interactor
            interactor.Initialize()
            render_window.Render()
        
            self.parent.status_var.set("Visualización 3D lista")
            interactor.Start()
        
        except Exception as e:
            messagebox.showerror("Error", f"Error al crear visualización 3D: {str(e)}")
            self.parent.status_var.set("Error al crear visualización 3D")

    def visualize_segmentation_3d(self, segmentation):
        """
        Crea una visualización 3D del resultado de la segmentación.
        
        Args:
            segmentation: Los datos de segmentación a visualizar.
        """
        try:
            self.parent.status_var.set("Creando visualización 3D de la segmentación...")
        
            # Crear un nuevo volumen para VTK
            volume = vtk.vtkImageData()
            volume.SetDimensions(self.parent.width, self.parent.height, self.parent.depth)
            volume.SetSpacing(1.0, 1.0, 1.0)
            volume.SetOrigin(0.0, 0.0, 0.0)
            volume.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        
            # Convertir datos de segmentación a formato VTK
            segmentation_data = segmentation.flatten().astype(np.uint8)
            vtk_data = numpy_to_vtk(segmentation_data, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
            volume.GetPointData().GetScalars().DeepCopy(vtk_data)
        
            # Crear un mapeador de contorno para mostrar las superficies de segmentación
            contour = vtk.vtkMarchingCubes()
            contour.SetInputData(volume)
        
            # Encontrar todos los valores únicos de etiquetas (excluyendo 0 que es el fondo)
            unique_labels = np.unique(segmentation)
            unique_labels = unique_labels[unique_labels > 0]
        
            if len(unique_labels) == 0:
                messagebox.showinfo("Sin datos", "No hay regiones segmentadas para visualizar en 3D.")
                return
        
            # Extraer un contorno por cada etiqueta
            for i, label in enumerate(unique_labels):
                contour.SetValue(i, label)
        
            # Crear mapeador de superficie
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(contour.GetOutputPort())
            mapper.ScalarVisibilityOn()
        
            # Crear mapa de colores
            lut = vtk.vtkLookupTable()
            lut.SetNumberOfTableValues(len(unique_labels) + 1)
            lut.SetTableRange(0, len(unique_labels))
            lut.Build()
        
            # Configurar colores para cada etiqueta
            import colorsys
            for i, label in enumerate(unique_labels):
                # Usar HSV para generar colores distintos
                hue = float(i) / len(unique_labels)
                lut.SetTableValue(i, *[*colorsys.hsv_to_rgb(hue, 1.0, 1.0), 1.0])
        
            mapper.SetLookupTable(lut)
        
            # Crear actor
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
        
            # Configurar propiedades
            actor.GetProperty().SetOpacity(0.7)
            actor.GetProperty().SetSpecular(0.3)
            
            # Configuración de la escena
            renderer = vtk.vtkRenderer()
            renderer.AddActor(actor)
            renderer.SetBackground(0.2, 0.2, 0.2)  # Fondo gris oscuro
        
            # Configurar la ventana de renderizado
            render_window = vtk.vtkRenderWindow()
            render_window.AddRenderer(renderer)
            render_window.SetSize(800, 600)
            render_window.SetWindowName("Visualización 3D de Segmentación")
        
            # Configurar interactor
            interactor = vtk.vtkRenderWindowInteractor()
            interactor.SetRenderWindow(render_window)
        
            # Configurar estilo de interacción
            style = vtk.vtkInteractorStyleTrackballCamera()
            interactor.SetInteractorStyle(style)
        
            # Inicializar y ajustar cámara
            renderer.ResetCamera()
            camera = renderer.GetActiveCamera()
            camera.Elevation(30)
            camera.Azimuth(30)
            camera.Zoom(1.2)
        
            # Iniciar visualización
            interactor.Initialize()
            render_window.Render()
        
            # Mostrar mensaje de estado
            self.parent.status_var.set("Visualización 3D generada. Cierre la ventana 3D para continuar.")
        
            # Iniciar el bucle de eventos
            interactor.Start()
        
        except Exception as e:
            messagebox.showerror("Error", f"Error al ejecutar la Visualización 3D: {str(e)}")
            self.parent.status_var.set("Error en la visualización")