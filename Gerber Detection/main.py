import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gerber
from gerber.render.cairo_backend import GerberCairoContext

# Funktion zur Erkennung von Elementen in den Gerber-Dateien
def detect_top_layer_elements(input_files):
    # Vorbereiten von Listen für die einzelnen Elementtypen
    pads, outlines, pcb_profile = [], [], []

    for file in input_files:
        gerber_file = gerber.read(file)
        # Wir nehmen an, dass die Eingabedateien die oberste Schicht oder das Profil darstellen
        for primitive in gerber_file.primitives:
            if isinstance(primitive, gerber.primitives.Line):
                if 'profile' in file.lower():
                    # Profil-Konturen zur PCB-Umrandung hinzufügen
                    pcb_profile.append(primitive)
            elif isinstance(primitive, gerber.primitives.Circle):
                if primitive.hole_diameter == 0:
                    pads.append(primitive)
            elif isinstance(primitive, gerber.primitives.Rectangle) or isinstance(primitive, gerber.primitives.Obround):
                # Rechtecke und obrunde Formen stellen die Komponentenkanten dar
                outlines.append(primitive)

    return pads, outlines, pcb_profile

# Funktion zum Erstellen eines Bildes mit den erkannten Elementen
def generate_top_layer_image_with_highlighted_pin_1(outlines, pads, pcb_profile, output_file='top_layer_image_matplotlib.png'):
    fig, ax = plt.subplots()

    # Zeichnen der PCB-Umrandung (in Schwarz)
    for line in pcb_profile:
        start = line.start
        end = line.end
        ax.plot([start[0], end[0]], [start[1], end[1]], color='black', linewidth=1)

    # Zeichnen der äußeren Komponentenkanten (in Rot mit dünneren Linien)
    for outline in outlines:
        if hasattr(outline, 'position'):
            x, y = outline.position
            width, height = outline.width, outline.height
            rect = patches.Rectangle((x - width / 2, y - height / 2), width, height, linewidth=0.05, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

    # Zeichnen der Pads (in Grün)
    for pad in pads:
        if hasattr(pad, 'position'):
            x, y = pad.position
            diameter = pad.diameter if hasattr(pad, 'diameter') else max(pad.width, pad.height)
            circle = patches.Circle((x, y), diameter / 2, linewidth=0.05, edgecolor='green', facecolor='none')
            ax.add_patch(circle)

    # Achsenverhältnis beibehalten und Plot speichern
    ax.set_aspect('equal')
    plt.axis('off')
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

if __name__ == "__main__":
    # Festlegen der Eingabe-Gerber-Dateien für die obere Schicht
    gerber_files = ['copper_top.gbr', 'soldermask_top.gbr', 'silkscreen_top.gbr', 'profile.gbr']

    if not all(os.path.exists(file) for file in gerber_files):
        raise FileNotFoundError("Eine oder mehrere Gerber-Dateien wurden im aktuellen Verzeichnis nicht gefunden.")

    pads, outlines, pcb_profile = detect_top_layer_elements(gerber_files)
    # Erstellen des Bildes mit matplotlib
    generate_top_layer_image_with_highlighted_pin_1(outlines, pads, pcb_profile)
    print("Bild mit hervorgehobenen Pads und Komponentenkanten erfolgreich erstellt.")
