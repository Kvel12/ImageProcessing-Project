"""
NIfTI Viewer - Aplicación para visualización y procesamiento de imágenes médicas.
Punto de entrada principal a la aplicación.
"""
import tkinter as tk
from ui.main_window import NiftiViewer

def main():
    """Función principal para iniciar la aplicación"""
    root = tk.Tk()
    app = NiftiViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main()