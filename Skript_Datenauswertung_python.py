import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uuid
import os
import csv

def remove_last_entry_if_odd(dictionary):
	if len(dictionary) % 2 == 1:  
		key_number_to_remove = int(len(dictionary) / 2)  
		key_to_remove = list(dictionary.keys())[key_number_to_remove]  
		del dictionary[key_to_remove]  
		print(f"Removed entry with key '{key_to_remove}'")

def data_control(dataset):
	print(len(dataset))
	for key in dataset.keys():
		print(key)
		print(dataset[key][:5])
		print(dataset[key][-5:])

def erstelle_dateipfadliste_und_dateinamen(dateiname):
	dateipfadliste = []
	dateinamenliste = []
	basispfad = "..."
	with open(dateiname, 'r') as file:
		for line in file:
			dateiname = line.strip()
			dateinamenliste.append(dateiname)
			dateipfad = f"{basispfad}/{dateiname}" 
			dateipfadliste.append(dateipfad)
	return dateipfadliste, dateinamenliste

def extrahiere_infos_aus_dateipfad(dateipfad):
    parts = dateipfad.split('__')
    modell = parts[1]
    rückenlehne = parts[2]
    dichtefaktor = float(parts[3].split('_')[1])
    rückenlehnenteil = parts[4]
    
    infos = {  
        'Modell': modell,
        'Rückenlehne': rückenlehne,
        'Dichtefaktor': dichtefaktor,
        'Rückenlehnenteil': rückenlehnenteil
    }

    return infos

def extrahiere_daten_aus_datei(dateipfad):
    with open(dateipfad, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    daten_list = []

    num_columns = len(lines[0].split(','))


    for col in range(0, num_columns, 2):

        titel = lines[0].split(',')[col].strip()
        beschriftung_x = lines[1].split(',')[col].strip()
        beschriftung_y = lines[1].split(',')[col + 1].strip()
        einheit_x = lines[2].split(',')[col].strip()
        einheit_y = lines[2].split(',')[col + 1].strip()

        werte_x = []
        werte_y = []
        for line in lines[3:]:
            values = line.strip().split(',')
            werte_x.append(float(values[col]))
            werte_y.append(float(values[col + 1]))


        daten = {
            'Primary Key': str(uuid.uuid4()),
            'Titel': titel,
            'Beschriftung-X-Achse': beschriftung_x,
            'Beschriftung-Y-Achse': beschriftung_y,
            'Einheit-X-Achse': einheit_x,
            'Einheit-Y-Achse': einheit_y,
            'Werte-X-Achse': werte_x,
            'Werte-Y-Achse': werte_y
        }

        daten_list.append(daten)

    return daten_list

def erstelle_komplettes_daten_dict(dateipfad):

    infos = extrahiere_infos_aus_dateipfad(dateipfad)


    daten_list = extrahiere_daten_aus_datei(dateipfad)


    komplettes_daten_dict_list = []
    for i, daten in enumerate(daten_list):
        primary_key = str(uuid.uuid4())
        dict_name = f"dictionary_{i+1}__{primary_key}"
        komplettes_daten_dict = {**infos, **daten}
        komplettes_daten_dict_list.append((dict_name, komplettes_daten_dict))

    return komplettes_daten_dict_list

def find_extreme_times(df, group_by_columns):
    grouped = df.groupby(group_by_columns).agg(
        earliest_t_start=('BIO_t_start', 'min'),
        latest_t_start=('BIO_t_start', 'max'),
        earliest_t_ende=('BIO_t_ende', 'min'),
        latest_t_ende=('BIO_t_ende', 'max')
    ).reset_index()
    return grouped

def find_dichtefaktoren_by_group(df, group_by_columns, value_column):
    def max_rows(group):
        max_value = group[value_column].max()
        return group[group[value_column] == max_value]
    max_rows_df = df.groupby(group_by_columns).apply(max_rows).reset_index(drop=True)
    result = max_rows_df[group_by_columns + ['Dichtefaktor', 'BIO_t_start', 'BIO_t_ende']].drop_duplicates()
    return result

def plot_daten_gesammelt(x_werte, df_y, df_info, x_label='X-Achse', y_label='Y-Achse'):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    modelle = df_info['Modell'].unique()
    rueckenlehnen = df_info['Rückenlehnenteil'].unique()

    extreme_times_df = find_extreme_times(df_info, ['Modell', 'Rückenlehnenteil'])
    max_dichtefaktoren_df = find_dichtefaktoren_by_group(df_info, ['Modell', 'Rückenlehnenteil'], 'CNCAP_Bewertung_NIC')

    all_y_values = []
    for modell in modelle:
        for rueckenlehne in rueckenlehnen:
            df_filtered_y = df_y[(df_y['Modell'] == modell) & (df_y['Rückenlehnenteil'] == rueckenlehne)]
            if not df_filtered_y.empty:
                y_values_all = np.array(df_filtered_y['Y-Werte'].tolist())
                all_y_values.extend(y_values_all.flatten())

    y_min = min(all_y_values)
    y_max = max(all_y_values)
    x_min = min(x_werte)
    x_max = max(x_werte)

    subplot_idx = 0
    for modell in modelle:
        for rueckenlehne in rueckenlehnen:
            df_filtered_y = df_y[(df_y['Modell'] == modell) & (df_y['Rückenlehnenteil'] == rueckenlehne)]
            
            if not df_filtered_y.empty:
                ax = axs[subplot_idx // 2, subplot_idx % 2]

                y_values_all = np.array(df_filtered_y['Y-Werte'].tolist())
                y_values_min = np.min(y_values_all, axis=0)
                y_values_max = np.max(y_values_all, axis=0)

                ax.plot(x_werte, y_values_min, label='Min Y-Werte', color='lightblue')
                ax.plot(x_werte, y_values_max, label='Max Y-Werte', color='darkblue')
                ax.fill_between(x_werte, y_values_min, y_values_max, color='purple', alpha=0.3)

                df_max_dichtefaktor = max_dichtefaktoren_df[(max_dichtefaktoren_df['Modell'] == modell) & (max_dichtefaktoren_df['Rückenlehnenteil'] == rueckenlehne)]
                all_t_starts = []
                all_t_endes = []
                for _, row in df_max_dichtefaktor.iterrows():
                    dichtefaktor = row['Dichtefaktor']
                    df_dichtefaktor = df_filtered_y[df_filtered_y['Dichtefaktor'] == dichtefaktor]
                    if not df_dichtefaktor.empty:
                        y_values_dichtefaktor = np.array(df_dichtefaktor['Y-Werte'].tolist())
                        ax.plot(x_werte, y_values_dichtefaktor[0], label=f'Dichtefaktor {dichtefaktor}', linestyle='--', color='black')
                        t_start = row['BIO_t_start']
                        t_ende = row['BIO_t_ende']
                        all_t_starts.append(t_start)
                        all_t_endes.append(t_ende)

                df_extreme_times = extreme_times_df[(extreme_times_df['Modell'] == modell) & (extreme_times_df['Rückenlehnenteil'] == rueckenlehne)]
                if not df_extreme_times.empty:
                    earliest_t_start = df_extreme_times['earliest_t_start'].values[0]
                    latest_t_start = df_extreme_times['latest_t_start'].values[0]
                    earliest_t_ende = df_extreme_times['earliest_t_ende'].values[0]
                    latest_t_ende = df_extreme_times['latest_t_ende'].values[0]
                    ax.axvspan(earliest_t_start, latest_t_start, color='green', alpha=0.5, label='Start Kopfkontakt')
                    ax.axvspan(earliest_t_ende, latest_t_ende, color='red', alpha=0.5, label='Ende Kopfkontakt')

                ax.set_title(f'{modell} - {rueckenlehne}')
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)

                custom_lines = [plt.Line2D([0], [0], color='purple', lw=4, alpha=0.3)]
                ax.legend(custom_lines + [plt.Line2D([0], [0], linestyle='--', color='black'), plt.Line2D([0], [0], color='green'), plt.Line2D([0], [0], color='red')], [f'Bereich {y_label}', f'Dichtefaktor {row["Dichtefaktor"]}', 'Startzeit Korridor', 'Endzeit Korridor'], loc='upper right')

                subplot_idx += 1

    beschreibung = df_y.iloc[0]['Beschreibung']
    filename = f'Gesamt_Uebersicht_{beschreibung}'
    save_path = os.path.join(os.getcwd(), 'bilder', f'{filename}_plot.png')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    save_path_csv = os.path.join(os.getcwd(), 'Bilderdaten', f'{filename}_daten.csv')
    with open(save_path_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        for modell in modelle:
            for rueckenlehne in rueckenlehnen:
                df_filtered_y = df_y[(df_y['Modell'] == modell) & (df_y['Rückenlehnenteil'] == rueckenlehne)]
                if not df_filtered_y.empty:
                    y_values_all = np.array(df_filtered_y['Y-Werte'].tolist())
                    for y_values in y_values_all:
                         for i in range(len(x_werte)):
                            row = [x_werte[i]] + [y_values[i]]
                            writer.writerow(row)
                
                df_extreme_times = extreme_times_df[(extreme_times_df['Modell'] == modell) & (extreme_times_df['Rückenlehnenteil'] == rueckenlehne)]
                if not df_extreme_times.empty:
                    row_start_korridor = ['Startzeit Korridor', df_extreme_times['earliest_t_start'].values[0], df_extreme_times['latest_t_start'].values[0]]
                    row_ende_korridor = ['Endzeit Korridor', df_extreme_times['earliest_t_ende'].values[0], df_extreme_times['latest_t_ende'].values[0]]
                    writer.writerow(row_start_korridor)
                    writer.writerow(row_ende_korridor)

def plot_daten(werte_x, werte_y_list, beschriftung_x, beschriftung_y_list, einheit_x, einheit_y, vorsatz):
    

    plt.figure(figsize=(10, 6))
    for i, werte_y in enumerate(werte_y_list):
        plt.plot(werte_x, werte_y, marker='o', linestyle='-', label=beschriftung_y_list[i], markersize=3)
    

    plt.title(', '.join(beschriftung_y_list))
    plt.xlabel(f'{beschriftung_x} ({einheit_x})')
    plt.ylabel(f'{", ".join(beschriftung_y_list)} ({einheit_y})')
    

    plt.grid(True)
    
    dateiname = f'{vorsatz}__{"_".join(beschriftung_y_list)}__{beschriftung_x}'

    speicherpfad = os.path.join(os.path.dirname(__file__), 'bilder',f'{dateiname}_plot.png')
    

    plt.savefig(speicherpfad)
    plt.close()

    speicherpfad_csv = os.path.join(os.path.dirname(__file__), 'Bilderdaten', f'{dateiname}_daten.csv')

    with open(speicherpfad_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        for i in range(len(werte_x)):
            row = [werte_x[i]] + [werte_y[i] for werte_y in werte_y_list]
            writer.writerow(row)

def daten_umbennen(dict, arr):
    df = pd.DataFrame(columns =['Y-Werte'], index = arr)
    i = 0
    while i<len(arr):
        df.loc[arr[i]]= [dict[i][1]['Werte-Y-Achse']]
        i += 1

    df.loc['Modell'] = [dict[0][1]['Modell']]
    df.loc['Rückenlehne'] = [dict[0][1]['Rückenlehne']]
    df.loc['Dichtefaktor'] = [dict[0][1]['Dichtefaktor']]
    df.loc['Rückenlehnenteil'] = [dict[0][1]['Rückenlehnenteil']]
    return df

def extremwerte_finden(dict, df):
    df_filtered = pd.DataFrame(columns=['Maximum', 'Zeitpunkt_Max', 'Minimum', 'Zeitpunkt_Min'])

    i = 0
    indices = df.index.tolist()
    
    while i < len(df)-4:
        index = indices[i]
        values = df.iloc[i, 0]
        
        max_value = max(values, key=abs)
        min_value = min(values, key=abs)
        
        max_index = values.index(max_value)
        min_index = values.index(min_value)
        
        time_max = dict[0][1]['Werte-X-Achse'][max_index]
        time_min = dict[0][1]['Werte-X-Achse'][min_index]

        df_filtered.loc[f'{index}'] = [max_value, time_max, min_value, time_min]

        i += 1
    
    return df_filtered

def differenz(array1, array2):
    if len(array1) != len(array2):
        raise ValueError("Die Arrays müssen die gleiche Länge haben.")

    difference = []

    for i in range(len(array1)):
        difference.append(array1[i] - array2[i])
    
    return difference

def summe(array1, array2):
    
    if len(array1) != len(array2):
        raise ValueError("Die Arrays müssen die gleiche Länge haben.")

    sum = []

    for i in range(len(array1)):
        sum.append(array1[i] + array2[i])
    
    return sum

def array_sum_range(arr, start, end):
    if not isinstance(arr, list):
        raise TypeError("Das Eingabearray muss eine Liste sein")
    if not (0 <= start < len(arr)) or not (0 <= end < len(arr)):
        raise ValueError("Start und Ende müssen innerhalb der Array-Grenzen liegen")
    if start > end:
        raise ValueError("Die Startposition muss kleiner oder gleich der Endposition sein")
    
    return sum(arr[start:end+1])

def produkt(array1, Faktor):
    
    product = []

    for i in range(len(array1)):
        product.append(array1[i] * Faktor)
    
    return product

def durchschnitt(array1, array2):
    if len(array1) != len(array2):
        raise ValueError("Die Arrays müssen die gleiche Länge haben.")

    durchschnitt = []

    for i in range(len(array1)):
        durchschnitt.append(np.mean([array1[i], array2[i]]))
    
    return durchschnitt

def quadrieren(array):
    quadriert = []

    for i in range(len(array)):
        quadriert.append(array[i] * array[i])
    
    return quadriert

def division_begrenzt(array1, array2, start_index, end_index):
    if len(array1) != len(array2):
        raise ValueError("Die Arrays müssen die gleiche Länge haben.")
    if start_index < 0 or start_index >= len(array1):
        raise ValueError("Der Startindex ist außerhalb des gültigen Bereichs.")
    if end_index is not None and (end_index < 0 or end_index >= len(array1)):
        raise ValueError("Der Endindex ist außerhalb des gültigen Bereichs.")
    
    if end_index is None:
        end_index = len(array1) - 1

    division = []

    for i in range(start_index, end_index + 1):
        division.append(array1[i] / array2[i])
    
    return division

def resultierende2d(array1, array2):
    if len(array1) != len(array2):
        raise ValueError("Die Arrays müssen die gleiche Länge haben.")
    
    resultant = []

    for i in range(len(array1)):
        resultant.append(np.sqrt((array1[i] ** 2) + (array2[i] ** 2)))
    
    return resultant

def resultierende3d(array1, array2, array3):
    if len(array1) != len(array2):
        raise ValueError("Die Arrays müssen die gleiche Länge haben.")
    
    resultant = []

    for i in range(len(array1)):
        resultant.append(np.sqrt((array1[i] ** 2) + (array2[i] ** 2) + (array3[i] ** 2)))
    
    return resultant

def maxmin(array1):
     return [max(array1), min(array1)]
def maxminabs(array1):
     return [max(array1, key=abs), min(array1, key=abs)]

def maxmin_index(array1, index, return_type='both'):
    sub_array = array1[:index+1]
    max_abs_val = max(sub_array)
    min_abs_val = min(sub_array)
    
    if return_type == 'max':
        return max_abs_val
    elif return_type == 'min':
        return min_abs_val
    else:
        return [max_abs_val, min_abs_val]

def maxminabs_index(array1, index, return_type='both'):
    sub_array = array1[:index+1]
    max_abs_val = max(sub_array, key=abs)
    min_abs_val = min(sub_array, key=abs)
    
    if return_type == 'max':
        return max_abs_val
    elif return_type == 'min':
        return min_abs_val
    else:
        return [max_abs_val, min_abs_val]

def numerische_ableitung(t, y):
    dt = np.diff(t)  
    dy = np.diff(y)  

    ableitung = dy / dt
    
    ableitung = np.insert(ableitung, 0, ableitung[0])
    
    return ableitung

def find_fall_and_rise_indices(data):
    start_fall_index = None
    for i in range(1, len(data)):
        if data[i] < data[i - 1]:
            start_fall_index = i
            break

    if start_fall_index is None:
        return None, None  
    
    end_fall_index = None
    for i in range(start_fall_index + 100, len(data)):
        if data[i] == 0:
            end_fall_index = i
            break

    return start_fall_index, end_fall_index

def find_index_of_return_to_zero(array):
    peak_index = array.index(max(array, key=abs))
    
    for i in range(peak_index + 1, len(array)):
        if array[i] == 0:
            return i
    
    return None

def radtograd(array):
    grad = []

    for i in range(len(array)):
        grad.append((array[i] * (180/np.pi)))
    
    return grad

def NIC_berechnen(df, tmax):
    xii_t1 = durchschnitt(df.loc['xii_t1l']['Y-Werte'], df.loc['xii_t1r']['Y-Werte'])
    xi_t1 = durchschnitt(df.loc['xi_t1l']['Y-Werte'], df.loc['xi_t1r']['Y-Werte'])
    x_t1 = durchschnitt(df.loc['x_t1l']['Y-Werte'], df.loc['x_t1r']['Y-Werte'])

    xii_rel = differenz(xii_t1, df.loc['xii_1']['Y-Werte'])
    xi_rel = differenz(xi_t1, df.loc['xi_1']['Y-Werte'])
    x_rel = differenz(x_t1, df.loc['x_1']['Y-Werte'])

    xi_rel_sq = (quadrieren(xi_rel))
    CNCAP_NIC = summe(produkt(xii_rel, (1000/5)), xi_rel_sq)
    CNCAP_NIC_max = maxmin_index(CNCAP_NIC, tmax, return_type='max')
    CNCAP_NIC_min = maxmin_index(CNCAP_NIC, tmax, return_type='min')

    return [CNCAP_NIC, CNCAP_NIC_min, CNCAP_NIC_max, xi_t1, xii_rel, xi_rel, x_rel]

def KIN_Energie_berechnen(array, masse):
    KIN_Energie = []

    for i in range(len(array)):
        KIN_Energie.append(((1/2) * masse * (array[i] ** 2)))
    
    return KIN_Energie

def punktzahl_bestimmen(x1, y1, x2, y2, x3):
    m = (y2 - y1) / (x2 - x1)
    
    b = y1 - m * x1
    
    y3 = m * x3 + b
    
    y_min = min(y1, y2)
    y_max = max(y1, y2)
    y3 = max(y_min, min(y3, y_max))
    
    return y3

def initialisierung(komplettes_daten_dict_listen, dateinamen_liste, Sensordaten, i):
    eingabe_datei = komplettes_daten_dict_listen[dateinamen_liste[i]]
    modell=eingabe_datei[0][1]['Modell']
    Rueckenlehne=eingabe_datei[0][1]['Rückenlehne']
    Dichtefaktor=eingabe_datei[0][1]['Dichtefaktor']
    Rueckenlehnenteil=eingabe_datei[0][1]['Rückenlehnenteil']
    Vorsatz = f'{modell}__{Rueckenlehne}__{Dichtefaktor}__{Rueckenlehnenteil}'
    Zeit=eingabe_datei[0][1]['Werte-X-Achse']
    df = daten_umbennen(eingabe_datei, Sensordaten)
    return [eingabe_datei, modell, Rueckenlehne, Dichtefaktor, Rueckenlehnenteil, Vorsatz, Zeit, df]

def save_df_to_txt(df, title):
    save_path = f'./{title}.txt'
    with open(save_path, 'w') as file:
        file.write(df.to_string())

def add_to_df(df, modell, rückenlehne, dichtefaktor, werte, beschreibung):
    new_row = pd.DataFrame({
        'Modell': [modell],
        'Rückenlehnenteil': [rückenlehne],
        'Dichtefaktor': [dichtefaktor],
        'Y-Werte': [werte],
        'Beschreibung': [beschreibung]
    })
    return pd.concat([df, new_row], ignore_index=True)

def main():
     
    dateiname = "Namen_Kurvendaten.txt"
    dateipfade_liste, dateinamen_liste = erstelle_dateipfadliste_und_dateinamen(dateiname)

    komplettes_daten_dict_listen = {}
    for dateipfad, dateiname in zip(dateipfade_liste, dateinamen_liste):
        komplettes_daten_dict_listen[dateiname] = erstelle_komplettes_daten_dict(dateipfad)
    komplettes_daten_dict_listen_df = pd.DataFrame(komplettes_daten_dict_listen)
    save_df_to_txt(komplettes_daten_dict_listen_df, 'komplettes_daten_dict_listen')
    df_grafen_x_1 = pd.DataFrame(columns=['Modell', 'Rückenlehnenteil', 'Dichtefaktor', 'Y-Werte', 'Beschreibung'])
    df_grafen_x_t1 = pd.DataFrame(columns=['Modell', 'Rückenlehnenteil', 'Dichtefaktor', 'Y-Werte', 'Beschreibung'])
    df_grafen_xi_1 = pd.DataFrame(columns=['Modell', 'Rückenlehnenteil', 'Dichtefaktor', 'Y-Werte', 'Beschreibung'])
    df_grafen_xi_t1 = pd.DataFrame(columns=['Modell', 'Rückenlehnenteil', 'Dichtefaktor', 'Y-Werte', 'Beschreibung'])
    df_grafen_xii_1 = pd.DataFrame(columns=['Modell', 'Rückenlehnenteil', 'Dichtefaktor', 'Y-Werte', 'Beschreibung'])
    df_grafen_xii_t1 = pd.DataFrame(columns=['Modell', 'Rückenlehnenteil', 'Dichtefaktor', 'Y-Werte', 'Beschreibung'])
    df_grafen_xiii_1 = pd.DataFrame(columns=['Modell', 'Rückenlehnenteil', 'Dichtefaktor', 'Y-Werte', 'Beschreibung'])
    df_grafen_xiii_t1 = pd.DataFrame(columns=['Modell', 'Rückenlehnenteil', 'Dichtefaktor', 'Y-Werte', 'Beschreibung'])

    df_grafen_x_rel = pd.DataFrame(columns=['Modell', 'Rückenlehnenteil', 'Dichtefaktor', 'Y-Werte', 'Beschreibung'])
    df_grafen_xi_rel = pd.DataFrame(columns=['Modell', 'Rückenlehnenteil', 'Dichtefaktor', 'Y-Werte', 'Beschreibung'])
    df_grafen_xii_rel = pd.DataFrame(columns=['Modell', 'Rückenlehnenteil', 'Dichtefaktor', 'Y-Werte', 'Beschreibung'])
    df_grafen_xiii_rel = pd.DataFrame(columns=['Modell', 'Rückenlehnenteil', 'Dichtefaktor', 'Y-Werte', 'Beschreibung'])

    df_grafen_NIC = pd.DataFrame(columns=['Modell', 'Rückenlehnenteil', 'Dichtefaktor', 'Y-Werte', 'Beschreibung'])

    df_grafen_z_1 = pd.DataFrame(columns=['Modell', 'Rückenlehnenteil', 'Dichtefaktor', 'Y-Werte', 'Beschreibung'])

    df_grafen_phi_z_1 = pd.DataFrame(columns=['Modell', 'Rückenlehnenteil', 'Dichtefaktor', 'Y-Werte', 'Beschreibung'])
    df_grafen_phi_z_t1 = pd.DataFrame(columns=['Modell', 'Rückenlehnenteil', 'Dichtefaktor', 'Y-Werte', 'Beschreibung'])
    df_grafen_phi_y_1 = pd.DataFrame(columns=['Modell', 'Rückenlehnenteil', 'Dichtefaktor', 'Y-Werte', 'Beschreibung'])
    df_grafen_phi_y_t1 = pd.DataFrame(columns=['Modell', 'Rückenlehnenteil', 'Dichtefaktor', 'Y-Werte', 'Beschreibung'])

    df_grafen_F_z_Kontakt = pd.DataFrame(columns=['Modell', 'Rückenlehnenteil', 'Dichtefaktor', 'Y-Werte', 'Beschreibung'])

    i = 0
    Datenauswertung_alle = pd.DataFrame(columns=[
        'Modell', 'Rückenlehne', 'Dichtefaktor', 'Rückenlehnenteil', 
        'CNCAP_NIC_min', 'CNCAP_NIC_max',
        'CNCAP_Scheerkraft_max_index', 'CNCAP_Scheerkraft_min_index',
        'CNCAP_Druckkraft_max_index', 'CNCAP_Druckkraft_min_index',
        'CNCAP_Biegemoment_UN_OC_max_index', 'CNCAP_Biegemoment_UN_OC_min_index',
        'CNCAP_Biegemoment_LN_max_index', 'CNCAP_Biegemoment_LN_min_index',
        'CNCAP_Bewertung_gesamt', 'CNCAP_Bewertung_LN', 'CNCAP_Bewertung_UN', 'CNCAP_Bewertung_NIC',
        'BIO_x_1_max', 'BIO_x_t1_max',
        'BIO_x_1_max', 'BIO_x_t1_max', 'BIO_x_t8_max', 'BIO_x_l1_max',
        'BIO_xi_1_max', 'BIO_xi_t1_max', 'BIO_xi_t8_max', 'BIO_xi_l1_max',
        'BIO_xii_1_max', 'BIO_xii_t1_max', 'BIO_xii_t8_max', 'BIO_xii_l1_max',
        'BIO_xiii_1_max', 'BIO_xiii_t1_max', 'BIO_xiii_t8_max', 'BIO_xiii_l1_max',
        'BIO_x_rel_max_index', 'BIO_xi_rel_max_index', 'BIO_xii_rel_max_index', 'BIO_xiii_rel_max_index', 'BIO_xiii_rel_summe', 
        'BIO_xii_1_avg', 'BIO_xii_t1_avg', 'BIO_xii_t8_avg', 'BIO_xii_l1_avg',
        'BIO_xiii_1_avg', 'BIO_xiii_t1_avg', 'BIO_xiii_t8_avg', 'BIO_xiii_l1_avg',
        'BIO_xiii_1_summe', 'BIO_xiii_t1_summe', 'BIO_xiii_rel_summe',
        'BIO_s_res_1_max', 'BIO_s_res_t1_max',
        'BIO_phi_z_1_max', 'BIO_phi_z_t1_max',
        'BIO_phi_y_1_max', 'BIO_phi_y_t1_max',
        'BIO_dphi_res_1_t1_max', 'BIO_dphi_res_1_t1_min',
        'BIO_dphi_res_t1_t8_max', 'BIO_dphi_res_t1_t8_min',
        'BIO_dphi_res_t8_l1_max', 'BIO_dphi_res_t8_l1_min',
        'BIO_dphi_z_1_t1_max', 'BIO_dphi_z_1_t1_min',
        'BIO_dphi_z_t1_t8_max', 'BIO_dphi_z_t1_t8_min',
        'BIO_dphi_z_t8_l1_max', 'BIO_dphi_z_t8_l1_min',
        'BIO_Scheerkraft_max', 'BIO_Scheerkraft_min',
        'BIO_Scheerkraft_res_max', 'BIO_Scheerkraft_res_min',
        'BIO_Kopfkontaktkraft_x_max', 'BIO_Kopfkontaktkraft_x_min',
        'BIO_Kopfkontaktkraft_z_max', 'BIO_Kopfkontaktkraft_z_min',
        'BIO_Kopf_v_x_max', 'Bio_Kopf_v_x_min',
        'BIO_Kopf_v_res_max', 'Bio_Kopf_v_res_min',
        'BIO_Federrate_avg',
        'BIO_Energie_Kopf_max', 'BIO_Energie_Kopf_min',
        'BIO_Energie_Oberkörper_max', 'BIO_Energie_Oberkörper_min',
        'BIO_Energie_Gesamt_max', 'BIO_Energie_Gesamt_min', 'BIO_Energie_Gesamt_Summe',
        'BIO_Wirkung', 'BIO_T_Kopfkontakt',
        'BIO_t_start', 'BIO_t_ende'
        ])
    
    Sensordaten = [
            'F_x_Kontakt',
            'F_y_Kontakt',
            'F_z_Kontakt',
            'phiii_x_1',
            'phiii_x_l1',
            'phiii_x_t1l',
            'phiii_x_t1r',
            'phiii_x_t8',
            'phiii_y_1',
            'phiii_y_l1',
            'phiii_y_t1l',
            'phiii_y_t1r',
            'phiii_y_t8',
            'phiii_z_1',
            'phiii_z_l1',
            'phiii_z_t1l',
            'phiii_z_t1r',
            'phiii_z_t8',
            'phi_x_1',
            'phi_x_l1',
            'phi_x_t1l',
            'phi_x_t1r',
            'phi_x_t8',
            'phi_y_1',
            'phi_y_l1',
            'phi_y_t1l',
            'phi_y_t1r',
            'phi_y_t8',
            'phi_z_1',
            'phi_z_l1',
            'phi_z_t1l',
            'phi_z_t1r',
            'phi_z_t8',
            'phii_x_1',
            'phii_x_l1',
            'phii_x_t1l',
            'phii_x_t1r',
            'phii_x_t8',
            'phii_y_1',
            'phii_y_l1',
            'phii_y_t1l',
            'phii_y_t1r',
            'phii_y_t8',
            'phii_z_1',
            'phii_z_l1',
            'phii_z_t1l',
            'phii_z_t1r',
            'phii_z_t8',
            'F_x_UN',
            'F_x_LN',
            'F_y_UN',
            'F_y_LN',
            'F_z_UN',
            'F_z_LN',
            'M_y_UN',
            'M_y_LN',
            'xii_1',
            'xii_l1',
            'xii_t1l',
            'xii_t1r',
            'xii_t8',
            'yii_1',
            'yii_l1',
            'yii_t1l',
            'yii_t1r',
            'yii_t8',
            'zii_1',
            'zii_l1',
            'zii_t1l',
            'zii_t1r',
            'zii_t8',
            'x_1',
            'x_l1',
            'x_t1l',
            'x_t1r',
            'x_t8',
            'y_1',
            'y_l1',
            'y_t1l',
            'y_t1r',
            'y_t8',
            'z_1',
            'z_l1',
            'z_t1l',
            'z_t1r',
            'z_t8',
            'xi_1',
            'xi_l1',
            'xi_t1l',
            'xi_t1r',
            'xi_t8',
            'yi_1',
            'yi_l1',
            'yi_t1l',
            'yi_t1r',
            'yi_t8',
            'zi_1',
            'zi_l1',
            'zi_t1l',
            'zi_t1r',
            'zi_t8',
            ]
        
    while i<len(dateinamen_liste):
        
        eingabe_datei, Modell, Rueckenlehne, Dichtefaktor, Rueckenlehnenteil, Vorsatz, Zeit, df = initialisierung(komplettes_daten_dict_listen, dateinamen_liste, Sensordaten, i)
         
        tmax = find_index_of_return_to_zero(df.loc['F_x_Kontakt']['Y-Werte'])
        t_start, t_ende = find_fall_and_rise_indices(df.loc['F_x_Kontakt']['Y-Werte'])
        T_Kopfkontakt = Zeit[t_ende] - Zeit[t_start]
        print(f' Zeitpunkt kein Kopfkontakt mehr: {Zeit[tmax]}')
        df_extremwerte = extremwerte_finden(eingabe_datei, df)
        #######Berechnungen für C-NCAP
        CNCAP_NIC, CNCAP_NIC_min, CNCAP_NIC_max, xi_t1, BIO_xii_rel, BIO_xi_rel, BIO_x_rel = NIC_berechnen(df, tmax)
        CNCAP_Scheerkraft_UN = df.loc['F_x_UN']['Y-Werte']
        CNCAP_Scheerkraft_LN = df.loc['F_x_LN']['Y-Werte']
        CNCAP_Scheerkraft = differenz(CNCAP_Scheerkraft_UN, CNCAP_Scheerkraft_LN)
        CNCAP_Scheerkraft_max, CNCAP_Scheerkraft_min = maxmin_index(CNCAP_Scheerkraft, tmax)
        BIO_xii_rel_max = maxminabs_index(BIO_xii_rel, tmax, return_type='max')
        BIO_xi_rel_max = maxminabs_index(BIO_xi_rel, tmax, return_type='max')
        BIO_x_rel_max = maxminabs_index(BIO_x_rel, tmax, return_type='max')

        CNCAP_Druckkraft_UN = df.loc['F_z_UN']['Y-Werte']
        CNCAP_Druckkraft_LN = df.loc['F_z_LN']['Y-Werte']
        CNCAP_Druckkraft = differenz(CNCAP_Druckkraft_UN, CNCAP_Druckkraft_LN)
        CNCAP_Druckkraft_max, CNCAP_Druckkraft_min = maxmin_index(CNCAP_Druckkraft, tmax)

        CNCAP_Biegemoment_UN = df.loc['M_y_UN']['Y-Werte']
        CNCAP_Biegemoment_LN = df.loc['M_y_LN']['Y-Werte']
        CNCAP_Biegemoment_UN_OC = differenz(CNCAP_Biegemoment_UN, produkt(CNCAP_Scheerkraft_UN, 17.778))
        CNCAP_Biegemoment_UN_OC_max, CNCAP_Biegemoment_UN_OC_min = maxminabs_index(CNCAP_Biegemoment_UN_OC, tmax)
        CNCAP_Biegemoment_LN_max, CNCAP_Biegemoment_LN_min = maxminabs_index(CNCAP_Biegemoment_LN, tmax)

        #######Berechnungen für Biokinematik
        BIO_Scheerkraft_UN = df.loc['F_y_UN']['Y-Werte']
        BIO_Scheerkraft_LN = df.loc['F_y_LN']['Y-Werte']
        BIO_Scheerkraft = differenz(BIO_Scheerkraft_UN, BIO_Scheerkraft_LN)
        BIO_Scheerkraft_max, BIO_Scheerkraft_min = maxmin(BIO_Scheerkraft)

        BIO_Scheerkraft_UN_res = resultierende2d(BIO_Scheerkraft_UN, CNCAP_Scheerkraft_UN)
        BIO_Scheerkraft_LN_res = resultierende2d(BIO_Scheerkraft_LN, CNCAP_Scheerkraft_LN)
        BIO_Scheerkraft_res = differenz(BIO_Scheerkraft_UN_res, BIO_Scheerkraft_LN_res)
        BIO_Scheerkraft_res_max, BIO_Scheerkraft_res_min = maxmin(BIO_Scheerkraft_res)

        BIO_dphi_x_1_t1 = differenz(df.loc['phi_x_1']['Y-Werte'], durchschnitt(df.loc['phi_x_t1l']
        ['Y-Werte'], df.loc['phi_x_t1r']['Y-Werte']))
        BIO_dphi_y_1_t1 = differenz(df.loc['phi_y_1']['Y-Werte'], durchschnitt(df.loc['phi_y_t1l']
        ['Y-Werte'], df.loc['phi_y_t1r']['Y-Werte']))
        BIO_dphi_res_1_t1 = resultierende2d(BIO_dphi_x_1_t1, BIO_dphi_y_1_t1)
        BIO_dphi_res_1_t1_max, BIO_dphi_res_1_t1_min = radtograd(maxminabs(BIO_dphi_res_1_t1))

        BIO_dphi_x_t1_t8 = differenz(df.loc['phi_x_t8']['Y-Werte'], durchschnitt(df.loc['phi_x_t1l']
        ['Y-Werte'], df.loc['phi_x_t1r']['Y-Werte']))
        BIO_dphi_y_t1_t8 = differenz(df.loc['phi_y_t8']['Y-Werte'], durchschnitt(df.loc['phi_y_t1l']
        ['Y-Werte'], df.loc['phi_y_t1r']['Y-Werte']))
        BIO_dphi_res_t1_t8 = resultierende2d(BIO_dphi_x_t1_t8, BIO_dphi_y_t1_t8)
        BIO_dphi_res_t1_t8_max, BIO_dphi_res_t1_t8_min = radtograd(maxminabs(BIO_dphi_res_t1_t8))

        BIO_dphi_x_t8_l1 = differenz(df.loc['phi_x_t8']['Y-Werte'], df.loc['phi_x_l1']
        ['Y-Werte'])
        BIO_dphi_y_t8_l1 = differenz(df.loc['phi_y_t8']['Y-Werte'], df.loc['phi_y_l1']
        ['Y-Werte'])
        BIO_dphi_res_t8_l1 = resultierende2d(BIO_dphi_x_t8_l1, BIO_dphi_y_t8_l1)
        BIO_dphi_res_t8_l1_max, BIO_dphi_res_t8_l1_min = radtograd(maxminabs(BIO_dphi_res_t8_l1))
        

        BIO_phi_z_1 = df.loc['phi_z_1']['Y-Werte']
        BIO_phi_z_t1 = durchschnitt(df.loc['phi_z_t1l']['Y-Werte'], df.loc['phi_z_t1r']['Y-Werte'])
        BIO_phi_z_1_max = max(BIO_phi_z_1, key=abs)
        BIO_phi_z_t1_max = max(BIO_phi_z_t1, key=abs)


        BIO_dphi_z_1_t1 = differenz(df.loc['phi_z_1']['Y-Werte'], durchschnitt(df.loc['phi_z_t1l']
        ['Y-Werte'], df.loc['phi_z_t1r']['Y-Werte']))
        BIO_dphi_z_1_t1_max, BIO_dphi_z_1_t1_min = radtograd(maxminabs(BIO_dphi_z_1_t1))
        BIO_dphi_z_t1_t8 = differenz(df.loc['phi_z_t8']['Y-Werte'], durchschnitt(df.loc['phi_z_t1l']
        ['Y-Werte'], df.loc['phi_x_t1r']['Y-Werte']))
        BIO_dphi_z_t1_t8_max, BIO_dphi_z_t1_t8_min = radtograd(maxminabs(BIO_dphi_z_t1_t8))
        BIO_dphi_z_t8_l1 = differenz(df.loc['phi_z_t8']['Y-Werte'], df.loc['phi_z_l1']
        ['Y-Werte'])
        BIO_dphi_z_t8_l1_max, BIO_dphi_z_t8_l1_min = radtograd(maxminabs(BIO_dphi_z_t8_l1))

        BIO_phi_y_1 = df.loc['phi_y_1']['Y-Werte']
        BIO_phi_y_t1 = durchschnitt(df.loc['phi_y_t1l']['Y-Werte'], df.loc['phi_y_t1r']['Y-Werte'])
        BIO_phi_y_1_max = max(df.loc['phi_y_1']['Y-Werte'], key=abs)
        BIO_phi_y_t1_max = max(durchschnitt(df.loc['phi_y_t1l']['Y-Werte'], df.loc['phi_y_t1r']['Y-Werte']), key=abs)


        BIO_x_1_max = max(df.loc['x_1']['Y-Werte'], key=abs)
        BIO_x_t1_max = max(durchschnitt(df.loc['x_t1r']['Y-Werte'], df.loc['x_t1l']['Y-Werte']), key=abs)
        BIO_x_t8_max = max(df.loc['x_t8']['Y-Werte'], key=abs)
        BIO_x_l1_max = max(df.loc['x_l1']['Y-Werte'], key=abs)

        BIO_s_res_1_max = max(resultierende3d(df.loc['x_1']['Y-Werte'], df.loc['y_1']['Y-Werte'], df.loc['z_1']['Y-Werte']), key=abs)
        BIO_s_res_t1_max = max(resultierende3d(durchschnitt(df.loc['x_t1r']['Y-Werte'], df.loc['x_t1l']['Y-Werte']), durchschnitt(df.loc['y_t1r']['Y-Werte'], df.loc['y_t1l']['Y-Werte']), durchschnitt(df.loc['z_t1r']['Y-Werte'], df.loc['z_t1l']['Y-Werte'])), key=abs)

        BIO_xi_1_max = max(df.loc['xi_1']['Y-Werte'], key=abs)
        BIO_xi_t1_max = max(durchschnitt(df.loc['xi_t1r']['Y-Werte'], df.loc['xi_t1l']['Y-Werte']), key=abs)
        BIO_xi_t8_max = max(df.loc['xi_t8']['Y-Werte'], key=abs)
        BIO_xi_l1_max = max(df.loc['xi_l1']['Y-Werte'], key=abs)

        BIO_xii_1_max = max(df.loc['xii_1']['Y-Werte'], key=abs)
        BIO_xii_t1_max = max(durchschnitt(df.loc['xii_t1r']['Y-Werte'], df.loc['xii_t1l']['Y-Werte']), key=abs)
        BIO_xii_t8_max = max(df.loc['xii_t8']['Y-Werte'], key=abs)
        BIO_xii_l1_max = max(df.loc['xii_l1']['Y-Werte'], key=abs)

        BIO_xiii_1_alle = numerische_ableitung(Zeit, df.loc['xii_1']['Y-Werte'])
        BIO_xiii_t1_alle = numerische_ableitung(Zeit, durchschnitt(df.loc['xii_t1r']['Y-Werte'], df.loc['xii_t1l']['Y-Werte']))
        BIO_xiii_t8_alle = numerische_ableitung(Zeit, df.loc['xii_t8']['Y-Werte'])
        BIO_xiii_l1_alle = numerische_ableitung(Zeit, df.loc['xii_l1']['Y-Werte'])

        BIO_xiii_1_max = max(BIO_xiii_1_alle, key=abs)
        BIO_xiii_t1_max = max(BIO_xiii_t1_alle, key=abs)
        BIO_xiii_t8_max = max(BIO_xiii_t8_alle, key=abs)
        BIO_xiii_l1_max = max(BIO_xiii_l1_alle, key=abs)

        BIO_xii_1_avg = np.mean(np.abs(df.loc['xii_1']['Y-Werte']))
        BIO_xii_t1_avg = np.mean(np.abs(durchschnitt(df.loc['xii_t1r']['Y-Werte'], df.loc['xii_t1l']['Y-Werte'])))
        BIO_xii_t8_avg = np.mean(np.abs(df.loc['xii_t8']['Y-Werte']))
        BIO_xii_l1_avg = np.mean(np.abs(df.loc['xii_l1']['Y-Werte']))
        
        BIO_xiii_1_avg = np.mean(BIO_xiii_1_alle)
        BIO_xiii_t1_avg = np.mean(BIO_xiii_t1_alle)
        BIO_xiii_t8_avg = np.mean(BIO_xiii_t8_alle)
        BIO_xiii_l1_avg = np.mean(BIO_xiii_l1_alle)

        BIO_xiii_1_summe = np.sum(np.abs(BIO_xiii_1_alle))
        BIO_xiii_t1_summe = np.sum(np.abs(BIO_xiii_t1_alle))

        BIO_xiii_rel =differenz(BIO_xiii_t1_alle, BIO_xiii_1_alle)
        BIO_xiii_rel_max = maxminabs_index(BIO_xiii_rel, tmax, return_type='max')
        BIO_xiii_rel_summe = np.sum(np.abs(differenz(BIO_xiii_t1_alle, BIO_xiii_1_alle)))

        BIO_x_1_gesammelt = df.loc['x_1']['Y-Werte']
        BIO_x_t1_gesammelt = durchschnitt(df.loc['x_t1r']['Y-Werte'], df.loc['x_t1l']['Y-Werte'])
        BIO_xi_1_gesammelt = df.loc['xi_1']['Y-Werte']
        BIO_xi_t1_gesammelt = durchschnitt(df.loc['xi_t1r']['Y-Werte'], df.loc['xi_t1l']['Y-Werte'])
        BIO_xii_1_gesammelt = df.loc['xii_1']['Y-Werte']
        BIO_xii_t1_gesammelt = durchschnitt(df.loc['xii_t1r']['Y-Werte'], df.loc['xii_t1l']['Y-Werte'])
        BIO_z_1_gesammelt = df.loc['z_1']['Y-Werte']

        BIO_Federrate_avg = np.mean(division_begrenzt(df.loc['F_x_Kontakt']['Y-Werte'], df.loc['xii_1']['Y-Werte'], t_start, t_ende))

        BIO_Kopfkontaktkraft_x_max, BIO_Kopfkontaktkraft_x_min = maxmin(df.loc['F_x_Kontakt']['Y-Werte'])
        BIO_Kopfkontaktkraft_z_max, BIO_Kopfkontaktkraft_z_min = maxmin(df.loc['F_z_Kontakt']['Y-Werte'])
        BIO_Kopf_v_x_max, Bio_Kopf_v_x_min = maxmin(df.loc['xi_1']['Y-Werte'])
        BIO_Kopf_v_res_max, Bio_Kopf_v_res_min = maxmin(resultierende3d(df.loc['xi_1']['Y-Werte'], df.loc['yi_1']['Y-Werte'], df.loc['zi_1']['Y-Werte']))

        Energie_Kopf = KIN_Energie_berechnen(df.loc['xi_1']['Y-Werte'], 4.58)
        Energie_Kopf_max, Energie_Kopf_min = maxmin(Energie_Kopf)
        Energie_Oberkörper = KIN_Energie_berechnen(resultierende3d(xi_t1, df.loc['xi_l1']['Y-Werte'], df.loc['xi_t8']['Y-Werte']), 42.86)
        Energie_Oberkörper_max, Energie_Oberkörper_min = maxmin(Energie_Oberkörper)
        Energie_Gesamt = summe(Energie_Kopf,Energie_Oberkörper)
        Energie_Gesamt_max, Energie_Gesamt_min = maxmin(Energie_Gesamt)
        Energie_Gesamt_Summe = np.sum(Energie_Gesamt)
        Wirkung = np.trapz(Energie_Gesamt, x=Zeit)

        Bewertung_CNCAP_NIC = round(punktzahl_bestimmen(30, 0, 8, 2, CNCAP_NIC_max), ndigits=2)
        Bewertung_CNCAP_Fx_UN = round(punktzahl_bestimmen(0.73, 0, 0.34, 1.5, maxminabs_index(CNCAP_Scheerkraft_UN, tmax, return_type='max')), ndigits=2)
        Bewertung_CNCAP_Fz_UN = round(punktzahl_bestimmen(1.13, 0, 0.47, 1.5, maxminabs_index(CNCAP_Druckkraft_UN, tmax, return_type='max')), ndigits=2)
        Bewertung_CNCAP_My_UN = round(punktzahl_bestimmen(40, 0, 12, 1.5, abs(CNCAP_Biegemoment_UN_OC_max)), ndigits=2)
        Bewertung_CNCAP_Fx_LN = round(punktzahl_bestimmen(0.73, 0, 0.34, 1.5, maxminabs_index(CNCAP_Scheerkraft_LN, tmax, return_type='max')), ndigits=2)
        Bewertung_CNCAP_Fz_LN = round(punktzahl_bestimmen(1.48, 0, 0.26, 1.5, maxminabs_index(CNCAP_Druckkraft_LN, tmax, return_type='max')), ndigits=2)
        Bewertung_CNCAP_My_LN = round(punktzahl_bestimmen(40, 0, 12, 1.5, abs(CNCAP_Biegemoment_LN_max)), ndigits=2)
        Bewertung_CNCAP_UN = min([Bewertung_CNCAP_Fx_UN, Bewertung_CNCAP_Fz_UN, Bewertung_CNCAP_My_UN])
        Bewertung_CNCAP_LN = min([Bewertung_CNCAP_Fx_LN, Bewertung_CNCAP_Fz_LN, Bewertung_CNCAP_My_LN])
        Bewertung_CNCAP_gesamt = Bewertung_CNCAP_NIC + Bewertung_CNCAP_UN + Bewertung_CNCAP_LN

        ergebnis = [
            Modell, Rueckenlehne, Dichtefaktor, Rueckenlehnenteil, 
            CNCAP_NIC_min, CNCAP_NIC_max, 
            CNCAP_Scheerkraft_max, CNCAP_Scheerkraft_min, 
            CNCAP_Druckkraft_max, CNCAP_Druckkraft_min, 
            CNCAP_Biegemoment_UN_OC_max, CNCAP_Biegemoment_UN_OC_min,
            CNCAP_Biegemoment_LN_max, CNCAP_Biegemoment_LN_min,
            Bewertung_CNCAP_gesamt, Bewertung_CNCAP_LN, Bewertung_CNCAP_UN, Bewertung_CNCAP_NIC,
            BIO_x_1_max, BIO_x_t1_max, 
            BIO_x_1_max, BIO_x_t1_max, BIO_x_t8_max, BIO_x_l1_max,
            BIO_xi_1_max, BIO_xi_t1_max, BIO_xi_t8_max, BIO_xi_l1_max,
            BIO_xii_1_max, BIO_xii_t1_max, BIO_xii_t8_max, BIO_xii_l1_max,
            BIO_xiii_1_max, BIO_xiii_t1_max, BIO_xiii_t8_max, BIO_xiii_l1_max,
            BIO_x_rel_max, BIO_xi_rel_max, BIO_xii_rel_max, BIO_xiii_rel_max, BIO_xiii_rel_summe, 
            BIO_xii_1_avg, BIO_xii_t1_avg, BIO_xii_t8_avg, BIO_xii_l1_avg,
            BIO_xiii_1_avg, BIO_xiii_t1_avg, BIO_xiii_t8_avg, BIO_xiii_l1_avg,
            BIO_xiii_1_summe, BIO_xiii_t1_summe, BIO_xiii_rel_summe,
            BIO_s_res_1_max, BIO_s_res_t1_max,
            BIO_phi_z_1_max, BIO_phi_z_t1_max,
            BIO_phi_y_1_max, BIO_phi_y_t1_max,
            BIO_dphi_res_1_t1_max, BIO_dphi_res_1_t1_min, 
            BIO_dphi_res_t1_t8_max, BIO_dphi_res_t1_t8_min, 
            BIO_dphi_res_t8_l1_max, BIO_dphi_res_t8_l1_min, 
            BIO_dphi_z_1_t1_max, BIO_dphi_z_1_t1_min, 
            BIO_dphi_z_t1_t8_max, BIO_dphi_z_t1_t8_min, 
            BIO_dphi_z_t8_l1_max, BIO_dphi_z_t8_l1_min,
            BIO_Scheerkraft_max, BIO_Scheerkraft_min, 
            BIO_Scheerkraft_res_max, BIO_Scheerkraft_res_min,
            BIO_Kopfkontaktkraft_x_max, BIO_Kopfkontaktkraft_x_min,
            BIO_Kopfkontaktkraft_z_max, BIO_Kopfkontaktkraft_z_min,
            BIO_Kopf_v_x_max, Bio_Kopf_v_x_min,
            BIO_Kopf_v_res_max, Bio_Kopf_v_res_min,
            BIO_Federrate_avg,
            Energie_Kopf_max, Energie_Kopf_min,
            Energie_Oberkörper_max, Energie_Oberkörper_min,
            Energie_Gesamt_max, Energie_Gesamt_min, Energie_Gesamt_Summe,
            Wirkung, T_Kopfkontakt,
            Zeit[t_start], Zeit[t_ende]
            ]
        Datenauswertung_alle.loc[f'{i+1}'] = ergebnis
        
        plot_daten(Zeit, [BIO_xiii_1_alle], 'Zeit', ['Beschleunigungsaenderung_Kopf_X'], 'ms', 'mm/ms2', Vorsatz)
        plot_daten(Zeit, [df.loc['xii_1']['Y-Werte']], 'Zeit', ['Beschleunigung_Kopf_X'], 'ms', 'mm/ms2', Vorsatz)
        plot_daten(Zeit, [durchschnitt(df.loc['xii_t1r']['Y-Werte'], df.loc['xii_t1l']['Y-Werte'])], 'Zeit', ['Beschleunigung_Unterer_Nacken_X'], 'ms', 'mm/ms2', Vorsatz)
        plot_daten(Zeit, [CNCAP_Biegemoment_UN_OC], 'Zeit', ['Biegemoment_UN'], 'ms', 'Nm', Vorsatz)
        plot_daten(Zeit, [CNCAP_Biegemoment_LN], 'Zeit', ['Biegemoment_LN'], 'ms', 'Nm', Vorsatz)
        plot_daten(Zeit, [df.loc['F_x_Kontakt']['Y-Werte']], 'Zeit', ['Kopfkontaktkraft_x'], 'ms', 'kN', Vorsatz)
        plot_daten(Zeit, [CNCAP_NIC], 'Zeit', ['NIC'], 'ms', '(mm/ms)2', Vorsatz)
        plot_daten(Zeit, [BIO_xii_rel], 'Zeit', ['xii_rel'], 'ms', 'mm/ms2', Vorsatz)
        plot_daten(Zeit, [BIO_xi_rel], 'Zeit', ['xi_rel'], 'ms', 'mm/ms', Vorsatz)
        plot_daten(Zeit, [df.loc['x_1']['Y-Werte']], 'Zeit', ['x_1'], 'ms', 'mm', Vorsatz)
        plot_daten(Zeit, [durchschnitt(df.loc['x_t1r']['Y-Werte'], df.loc['x_t1l']['Y-Werte'])], 'Zeit', ['x_t1'], 'ms', 'mm', Vorsatz)
        plot_daten(Zeit, [df.loc['xi_1']['Y-Werte']], 'Zeit', ['xi_1'], 'ms', 'mm/ms', Vorsatz)
        plot_daten(Zeit, [durchschnitt(df.loc['xi_t1r']['Y-Werte'], df.loc['xi_t1l']['Y-Werte'])], 'Zeit', ['xi_t1'], 'ms', 'mm/ms', Vorsatz)
        
        df_grafen_x_1 = add_to_df(df_grafen_x_1, Modell, Rueckenlehnenteil, Dichtefaktor, BIO_x_1_gesammelt, 'x_1')
        df_grafen_x_t1 = add_to_df(df_grafen_x_t1, Modell, Rueckenlehnenteil, Dichtefaktor, BIO_x_t1_gesammelt, 'x_t1')
        df_grafen_xi_1 = add_to_df(df_grafen_xi_1, Modell, Rueckenlehnenteil, Dichtefaktor, BIO_xi_1_gesammelt, 'xi_1')
        df_grafen_xi_t1 = add_to_df(df_grafen_xi_t1, Modell, Rueckenlehnenteil, Dichtefaktor, BIO_xi_t1_gesammelt, 'xi_t1')
        df_grafen_xii_1 = add_to_df(df_grafen_xii_1, Modell, Rueckenlehnenteil, Dichtefaktor, BIO_xii_1_gesammelt, 'xii_1')
        df_grafen_xii_t1 = add_to_df(df_grafen_xii_t1, Modell, Rueckenlehnenteil, Dichtefaktor, BIO_xii_t1_gesammelt, 'xii_t1')
        df_grafen_xiii_1 = add_to_df(df_grafen_xiii_1, Modell, Rueckenlehnenteil, Dichtefaktor, BIO_xiii_1_alle, 'xiii_1')
        df_grafen_xiii_t1 = add_to_df(df_grafen_xiii_t1, Modell, Rueckenlehnenteil, Dichtefaktor, BIO_xiii_t1_alle, 'xiii_t1')
        df_grafen_x_rel = add_to_df(df_grafen_x_rel, Modell, Rueckenlehnenteil, Dichtefaktor, BIO_x_rel, 'x_rel')
        df_grafen_xi_rel = add_to_df(df_grafen_xi_rel, Modell, Rueckenlehnenteil, Dichtefaktor, BIO_xi_rel, 'xi_rel')
        df_grafen_xii_rel = add_to_df(df_grafen_xii_rel, Modell, Rueckenlehnenteil, Dichtefaktor, BIO_xii_rel, 'xii_rel')
        df_grafen_xiii_rel = add_to_df(df_grafen_xiii_rel, Modell, Rueckenlehnenteil, Dichtefaktor, BIO_xiii_rel, 'xiii_rel')
        df_grafen_NIC = add_to_df(df_grafen_NIC, Modell, Rueckenlehnenteil, Dichtefaktor, CNCAP_NIC, 'NIC')
        df_grafen_z_1 = add_to_df(df_grafen_z_1, Modell, Rueckenlehnenteil, Dichtefaktor, BIO_z_1_gesammelt, 'z_1')
        df_grafen_phi_z_1 = add_to_df(df_grafen_phi_z_1, Modell, Rueckenlehnenteil, Dichtefaktor, BIO_phi_z_1, 'phi_z_1')
        df_grafen_phi_z_t1 = add_to_df(df_grafen_phi_z_t1, Modell, Rueckenlehnenteil, Dichtefaktor, BIO_phi_z_t1, 'phi_z_t1')
        df_grafen_phi_y_1 = add_to_df(df_grafen_phi_y_1, Modell, Rueckenlehnenteil, Dichtefaktor, BIO_phi_y_1, 'phi_y_1')
        df_grafen_phi_y_t1 = add_to_df(df_grafen_phi_y_t1, Modell, Rueckenlehnenteil, Dichtefaktor, BIO_phi_y_t1, 'phi_y_t1')
        df_grafen_F_z_Kontakt = add_to_df(df_grafen_F_z_Kontakt, Modell, Rueckenlehnenteil, Dichtefaktor, df.loc['F_z_Kontakt']['Y-Werte'], 'F_z_Kontakt')
        
        sensordaten_path = os.path.join(os.getcwd(), 'Sensordaten', Modell)
        os.makedirs(sensordaten_path, exist_ok=True)

        for sensor in Sensordaten:
            sensor_values = df.loc[sensor]['Y-Werte']
            csv_filename = f"{Modell}_{Rueckenlehne}_{Dichtefaktor}_{Rueckenlehnenteil}_{sensor}.csv"
            csv_filepath = os.path.join(sensordaten_path, csv_filename)
            with open(csv_filepath, mode='w', newline='') as file:
                writer = csv.writer(file)
                for t, value in zip(Zeit, sensor_values):
                    writer.writerow([t, value])

        i += 1
    save_df_to_txt(Datenauswertung_alle, 'Datenauswertung_vollstaendig')
    Datenauswertung_alle.to_csv('Datenauswertung_vollsteandig.csv', index=False) 
    plot_daten_gesammelt(Zeit, df_grafen_x_1, Datenauswertung_alle, x_label='Zeit [ms]', y_label='x_1 [mm]')
    plot_daten_gesammelt(Zeit, df_grafen_x_t1, Datenauswertung_alle, x_label='Zeit [ms]', y_label='x_t1 [mm]')
    plot_daten_gesammelt(Zeit, df_grafen_xi_1, Datenauswertung_alle, x_label='Zeit [ms]', y_label='xi_1 [mm/ms]')
    plot_daten_gesammelt(Zeit, df_grafen_xi_t1, Datenauswertung_alle, x_label='Zeit [ms]', y_label='xi_t1 [mm/ms]')
    plot_daten_gesammelt(Zeit, df_grafen_xii_1, Datenauswertung_alle, x_label='Zeit [ms]', y_label='xii_1 [mm/ms^2]')
    plot_daten_gesammelt(Zeit, df_grafen_xii_t1, Datenauswertung_alle, x_label='Zeit [ms]', y_label='xii_t1 [mm/ms^2]')
    plot_daten_gesammelt(Zeit, df_grafen_xiii_1, Datenauswertung_alle, x_label='Zeit [ms]', y_label='xiii_1 [mm/ms^3]')
    plot_daten_gesammelt(Zeit, df_grafen_xiii_t1, Datenauswertung_alle, x_label='Zeit [ms]', y_label='xiii_t1 [mm/ms^3]')
    plot_daten_gesammelt(Zeit, df_grafen_x_rel, Datenauswertung_alle, x_label='Zeit [ms]', y_label='x_rel [mm]')
    plot_daten_gesammelt(Zeit, df_grafen_xi_rel, Datenauswertung_alle, x_label='Zeit [ms]', y_label='xi_rel [mm/ms]')
    plot_daten_gesammelt(Zeit, df_grafen_xii_rel, Datenauswertung_alle, x_label='Zeit [ms]', y_label='xii_rel [mm/ms^2]')
    plot_daten_gesammelt(Zeit, df_grafen_xiii_rel, Datenauswertung_alle, x_label='Zeit [ms]', y_label='xiii_rel [mm/ms^3]')
    plot_daten_gesammelt(Zeit, df_grafen_NIC, Datenauswertung_alle, x_label='Zeit [ms]', y_label='NIC [mm^2/ms^2]')
    plot_daten_gesammelt(Zeit, df_grafen_z_1, Datenauswertung_alle, x_label='Zeit [ms]', y_label='z_1 [mm]')
    plot_daten_gesammelt(Zeit, df_grafen_phi_z_1, Datenauswertung_alle, x_label='Zeit [ms]', y_label='phi_z_1 [rad]')
    plot_daten_gesammelt(Zeit, df_grafen_phi_z_t1, Datenauswertung_alle, x_label='Zeit [ms]', y_label='phi_z_t1 [rad]')
    plot_daten_gesammelt(Zeit, df_grafen_phi_y_1, Datenauswertung_alle, x_label='Zeit [ms]', y_label='phi_y_1 [rad]')
    plot_daten_gesammelt(Zeit, df_grafen_phi_y_t1, Datenauswertung_alle, x_label='Zeit [ms]', y_label='phi_y_t1 [rad]')
    plot_daten_gesammelt(Zeit, df_grafen_F_z_Kontakt, Datenauswertung_alle, x_label='Zeit [ms]', y_label='Fz_Kopf')



if __name__ == "__main__":
    main()

