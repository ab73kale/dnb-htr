import cv2
import os
from PIL import Image

# Einstellungen
input_image_path = "Zugangsbuch.jpg"
output_dir = "output_lines"
transcription_file = os.path.join(output_dir, "transcriptions.txt")

# Erstelle Ausgabeordner
os.makedirs(output_dir, exist_ok=True)

# Bild einlesen und in Graustufen umwandeln
image = cv2.imread(input_image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Binarisieren für bessere Linienerkennung
_, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

# Find horizontal contours
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (image.shape[1]//2, 1))
dilated = cv2.dilate(thresh, kernel, iterations=2)
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Zeilen extrahieren
lines = sorted(contours, key=lambda x: cv2.boundingRect(x)[1])
with open(transcription_file, "w", encoding="utf-8") as f:
    for i, cnt in enumerate(lines):
        x, y, w, h = cv2.boundingRect(cnt)
        if h < 30 or w < 300:  # Filter zu kleine
            continue
        roi = image[y:y+h, x:x+w]
        line_filename = f"zeile_{i+1:04}.png"
        cv2.imwrite(os.path.join(output_dir, line_filename), roi)
        f.write(f"{line_filename}\t\n")  # Leere Transkription einfügen
