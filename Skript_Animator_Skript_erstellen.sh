#!/bin/bash

erstelle_dateinamen() {
	lokaler_dateipfad=$1

	# Extrahiere die relevanten Teile des Dateipfads
	Kopfstuetze=$(basename "$lokaler_dateipfad")
	Dichtefaktor=$(basename "$(dirname "$lokaler_dateipfad")")
	Modell=$(basename "$(dirname "$(dirname "$(dirname "$lokaler_dateipfad")")")")
	if [ "$Modell" = "VW313_2CM" ]; then
		Rueckenlehne="KEB47"
	else
		Rueckenlehne="KEB23"
	fi

	# Extrahiere den Pfad des übergeordneten Verzeichnisses
	uebergeordnetes_verzeichnis=$(dirname "$lokaler_dateipfad")

	# Finde die .log-Datei im übergeordneten Verzeichnis
	log_datei=$(find "$uebergeordnetes_verzeichnis" -maxdepth 1 -type f -name "*.log")

	# Extrahiere den ersten Zahlenwert aus der ersten Zeile der .log-Datei
	Kopfstuetze_Gewicht=$(head -n 1 "$log_datei" | grep -oP '\d+\.\d+')

	# Erstelle den gewünschten Dateinamen
	Dateiname="${Modell}__${Rueckenlehne}__${Dichtefaktor}__${Kopfstuetze}__${Kopfstuetze_Gewicht}.csv"
	Dateiname=${Dateiname::-4}
	Name_A4_Skript="A4_Skript_$Dateiname.ses"
	Name_Kurvendaten="Kurvendaten__$Dateiname.csv"
	Rueckgabe=("$Dateiname" "$Modell" "$Name_A4_Skript" "$Name_Kurvendaten" "$Kopfstuetze")
	echo "${Rueckgabe[@]}"
}


erstelle_A4_Skript() {
	lokaler_dateipfad=$1
	Dateiname=$2
	Modell=$3
	Name_A4_Skript=$4
	Name_Kurvendaten=$5
	Kopfstuetze=$6
	#echo "$Dateiname_Kurvendaten"
	if [ "$Modell" = "VW313_2CM" ]; then
		if [ "$Kopfstuetze" = "WL_SP04" ]; then
			Dateipfad_Ergaenzung_Daten="VW313_2CM_DKe_00020_23_a_k_e_k_bB00_cncap_WP_RESULT.erfh5.fz"
		else
			Dateipfad_Ergaenzung_Daten="VW313_2CM_DKe_00020_23_a_k_e_k_b00B_cncap_WP_RESULT.erfh5.fz"
		fi
	else
		if [ "$Kopfstuetze" = "WL_SP04" ]; then
			Dateipfad_Ergaenzung_Daten="VW316_8_keb23_6040_mMAL_mDLE___Whiplash_CNCAP_bB00_v001_RESULT.erfh5.fz"
		else
			Dateipfad_Ergaenzung_Daten="VW316_8_keb23_6040_mMAL_mDLE___Whiplash_CNCAP_b00B_v001_RESULT.erfh5.fz"
		fi
	fi
	Dateipfad_Daten="$lokaler_dateipfad/$Dateipfad_Ergaenzung_Daten"
	#echo "$Dateipfad_Kurvendaten"
	
	# Platzhalter in der Vorlage-Datei ersetzen
	if [ "$Modell" = "VW313_2CM" ]; then
		if [ "$Kopfstuetze" = "WL_SP04" ]; then
			vorlage_text=$(<A4_Kurvendaten_auslesen_Vorlage_VW313_SP04.txt)
		else
			vorlage_text=$(<A4_Kurvendaten_auslesen_Vorlage_VW313_SP06.txt)
		fi
	else
		if [ "$Kopfstuetze" = "WL_SP04" ]; then
			vorlage_text=$(<A4_Kurvendaten_auslesen_Vorlage_VW316_SP04.txt)
		else
			vorlage_text=$(<A4_Kurvendaten_auslesen_Vorlage_VW316_SP06.txt)
		fi
	fi
	vorlage_text=${vorlage_text//Dateipfad_zum_Ersetzen/$Dateipfad_Daten}
	vorlage_text=${vorlage_text//Dateiname_Kurvendaten/$Name_Kurvendaten}

	# In eine neue Datei schreiben
	echo "$vorlage_text" > "$Name_A4_Skript"
}

schreibe_array_in_datei() {
    local array_name="$1"  # Name des Arrays
    local datei="${array_name}.txt"  # Dateiname, aus dem Array-Namen generiert
    local -n array="$array_name"  # Erhalte den Namen des Arrays als Referenz
    for element in "${array[@]}"; do
        echo "$element" >> "$datei"
    done
}

echo "Skript startet"
lokaler_Dateipfad=($(find /home/vwwqvkl/bachelorarbeit/KVS_Daten_entpackt -type f -name "*.erfh5.fz" -exec dirname {} \; | grep -v 'Output' | sort -u))

Namen_A4_Skripte=()
Namen_Kurvendaten=()

for pfad in "${lokaler_Dateipfad[@]}"; do
	Rueckgabe=($(erstelle_dateinamen "$pfad"))
	Namen_Dateinamen+=("${Rueckgabe[0]}")
	Namen_Modell+=("${Rueckgabe[1]}")
	Namen_A4_Skripte+=("${Rueckgabe[2]}")
	Namen_Kurvendaten+=("${Rueckgabe[3]}")
	Kopfstuetze+=("${Rueckgabe[4]}")
done


variablen_liste="Namen_Dateinamen Namen_Modell Namen_A4_Skripte Namen_Kurvendaten Kopfstuetze"

for variable in $variablen_liste; do
	schreibe_array_in_datei "$variable"
done

echo "Erstellung der Skripte"
i=0
for pfad in "${lokaler_Dateipfad[@]}"; do
	erstelle_A4_Skript "${lokaler_Dateipfad[$i]}" "${Namen_Dateinamen[$i]}" "${Namen_Modell[$i]}" "${Namen_A4_Skripte[$i]}" "${Namen_Kurvendaten[$i]}" "${Kopfstuetze[$i]}"
	i=$((i + 1))
done

echo "Ausführung der Animator-Skripte und Erstellung der CSV-Dateien"
for skript in "${Namen_A4_Skripte[@]}"; do
	echo "A4 Skript wird ausgeführt"
	a4 -b "$skript" >/dev/null 2>&1 &   # Startet das Skript im Hintergrund
	pid=$!             # Speichert die Prozess-ID des Hintergrund-Jobs
	wait "$pid"        # Wartet auf das Ende des Hintergrund-Jobs
	echo "A4 Skript Ende"
done

echo "Skript beendet"





