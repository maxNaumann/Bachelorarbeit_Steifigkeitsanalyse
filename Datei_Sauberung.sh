#!/bin/bash

# Verzeichnis, in dem die Dateien gespeichert sind
verzeichnis="//home/vwwqvkl/bachelorarbeit/workworkworkworkwork"

# Schleife durch alle Dateien mit dem Dateinamen "Kurvendaten__"
for datei in "$verzeichnis"/Kurvendaten__*; do
    # Überprüfen, ob die Datei existiert und eine reguläre Datei ist
    if [ -f "$datei" ]; then
        # Ausgabe der Datei vor der Bearbeitung
        echo "Datei vor der Bearbeitung: $datei"
        
        # Bearbeiten der Datei: Löschen von "," am Anfang oder Ende einer Zeile
        sed -i 's/^,//' "$datei"  # Lösche "," am Anfang der Zeile
        sed -i 's/,$//' "$datei"  # Lösche "," am Ende der Zeile
        
        # Ausgabe der Datei nach der Bearbeitung
        echo "Datei nach der Bearbeitung: $datei"
    fi
done

