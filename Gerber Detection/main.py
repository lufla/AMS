import os
import cv2
import numpy as np
import gerber
from gerber.render.cairo_backend import GerberCairoContext

# Funktion zur Erkennung von Elementen in den Gerber-Dateien
def detect_top_layer_elements(input_files):
    # Vorbereiten von Listen für die einzelnen Elementtypen
    traces, pads, holes, outlines, pcb_profile = [], [], [], [], []

    for file in input_files:
        gerber_file = gerber.read(file)
        # Wir nehmen an, dass die Eingabedateien die oberste Schicht oder das Profil darstellen
        for primitive in gerber_file.primitives:
            if isinstance(primitive, gerber.primitives.Line):
                if 'profile' in file.lower():
                    # Profil-Konturen zur PCB-Umrandung hinzufügen
                    pcb_profile.append(primitive)
                else:
                    # Annahme: Dies sind Verbindungen zwischen Komponenten
                    traces.append(primitive)
            elif isinstance(primitive, gerber.primitives.Circle):
                if primitive.hole_diameter > 0:
                    holes.append(primitive)
                else:
                    pads.append(primitive)
            elif isinstance(primitive, gerber.primitives.Rectangle) or isinstance(primitive, gerber.primitives.Obround):
                # Rechtecke und obrunde Formen stellen die Komponentenkanten dar
                outlines.append(primitive)

    return traces, pads, holes, outlines, pcb_profile

# Funktion zum Erstellen eines Bildes mit den erkannten Elementen (nur Pin 1 jeder Komponente markieren und beschriften)
def generate_top_layer_image_with_highlighted_pin_1(outlines, pads, holes, pcb_profile, output_file='top_layer_image_with_highlighted_pin_1.png'):
    # Bestimmen der Umrandung
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')

    # Bestimmen der Umrandung für alle Elemente (Pads, Löcher, Umrisse und Profil)
    for outline in outlines:
        if hasattr(outline, 'position'):
            x, y = outline.position
            width, height = outline.width, outline.height
            min_x, min_y = min(min_x, x - width / 2), min(min_y, y - height / 2)
            max_x, max_y = max(max_x, x + width / 2), max(max_y, y + height / 2)

    for pad in pads:
        if hasattr(pad, 'position'):
            x, y = pad.position
            diameter = pad.diameter if hasattr(pad, 'diameter') else max(pad.width, pad.height)
            min_x, min_y = min(min_x, x - diameter / 2), min(min_y, y - diameter / 2)
            max_x, max_y = max(max_x, x + diameter / 2), max(max_y, y + diameter / 2)

    for hole in holes:
        x, y = hole.position
        diameter = hole.diameter if hasattr(hole, 'diameter') else 10
        min_x, min_y = min(min_x, x - diameter / 2), min(min_y, y - diameter / 2)
        max_x, max_y = max(max_x, x + diameter / 2), max(max_y, y + diameter / 2)

    # Fügen Sie etwas Abstand zur Umrandung hinzu
    padding = 50  # Erhöhter Abstand für bessere Visualisierung
    min_x -= padding
    min_y -= padding
    max_x += padding
    max_y += padding

    # Berechnen der Breite und Höhe des neuen Bildes
    width = int((max_x - min_x))
    height = int((max_y - min_y))

    # Erhöhen der Auflösung und Skalierungsfaktor
    scale_factor = 90  # Beibehalten eines hohen Skalierungsfaktors für Klarheit
    image_size = (width * scale_factor, height * scale_factor)
    img = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

    # Zeichnen der Komponentenkanten (in Grün für äußere, Rot für innere Komponenten)
    component_counter = 1  # Zähler für die Beschriftung der großen Komponenten
    for outline in outlines:
        if hasattr(outline, 'position'):
            # Zeichnen des grünen Quadrats um die äußere Komponente
            top_left = (int((outline.position[0] - outline.width / 2 - min_x) * scale_factor),
                        int((outline.position[1] - outline.height / 2 - min_y) * scale_factor))
            bottom_right = (int((outline.position[0] + outline.width / 2 - min_x) * scale_factor),
                            int((outline.position[1] + outline.height / 2 - min_y) * scale_factor))

            # Zeichnen des Rechtecks für die äußere Komponente in Grün, wenn es eine äußere Komponente ist
            if outline.width > 10 and outline.height > 10:
                cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            else:
                # Zeichnen des inneren Rechtecks in Rot
                cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # Zeichnen von Pin 1 in Gelb und ein Label hinzufügen
            pin_1_position = (top_left[0] + 15, top_left[1] + 15)
            cv2.circle(img, pin_1_position, 8, (0, 255, 255), -1)  # Gelber Kreis für Pin 1
            label = f"U{component_counter}"
            cv2.putText(img, label, (pin_1_position[0] + 10, pin_1_position[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, lineType=cv2.LINE_AA)
            component_counter += 1

    # Zeichnen der Pads (in Blau, sauber getrennt)
    for pad in pads:
        if hasattr(pad, 'position'):
            center = (int((pad.position[0] - min_x) * scale_factor), int((pad.position[1] - min_y) * scale_factor))
            radius = int(pad.diameter / 4 * scale_factor) if hasattr(pad, 'diameter') else int(max(pad.width, pad.height) / 4 * scale_factor)
            cv2.circle(img, center, radius, (255, 0, 0), -1, lineType=cv2.LINE_AA)

    # Zeichnen der Löcher (in Grün, mit angepasstem Radius)
    for hole in holes:
        center = (int((hole.position[0] - min_x) * scale_factor), int((hole.position[1] - min_y) * scale_factor))
        radius = int(hole.diameter / 4 * scale_factor) if hasattr(hole, 'diameter') else 15  # Reduzierter Radius
        cv2.circle(img, center, radius, (0, 255, 0), -1, lineType=cv2.LINE_AA)  # Verwenden von Anti-Aliasing

    # Zeichnen des PCB-Profils (in Weiß)
    for line in pcb_profile:
        start = (int((line.start[0] - min_x) * scale_factor), int((line.start[1] - min_y) * scale_factor))
        end = (int((line.end[0] - min_x) * scale_factor), int((line.end[1] - min_y) * scale_factor))
        cv2.line(img, start, end, (255, 255, 255), 3, lineType=cv2.LINE_AA)

    # Speichern des Bildes
    cv2.imwrite(output_file, img)

if __name__ == "__main__":
    # Festlegen der Eingabe-Gerber-Dateien für die obere Schicht
    gerber_files = ['copper_top.gbr', 'soldermask_top.gbr', 'silkscreen_top.gbr', 'profile.gbr']

    if not all(os.path.exists(file) for file in gerber_files):
        raise FileNotFoundError("Eine oder mehrere Gerber-Dateien wurden im aktuellen Verzeichnis nicht gefunden.")

    traces, pads, holes, outlines, pcb_profile = detect_top_layer_elements(gerber_files)
    # Entfernen der Vias, indem kleine Löcher gefiltert werden (Vias sind normalerweise kleine Löcher)
    filtered_holes = [hole for hole in holes if hole.diameter > 0.4]  # Anpassen der Schwelle nach Bedarf
    # Erstellen des Bildes ohne Leiterbahnen und Vias, aber mit Komponentenkanten und hervorgehobenen größeren Komponenten
    generate_top_layer_image_with_highlighted_pin_1(outlines, pads, filtered_holes, pcb_profile)
    print("Bild ohne Leiterbahnen und Vias, aber mit hervorgehobenen Pin 1 jeder Komponente und Labels erfolgreich erstellt.")
