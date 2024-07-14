import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau, mannwhitneyu, ttest_ind
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit
import os
import logging

def save_df_to_txt(df, title):
    save_path = f'./{title}.txt'
    with open(save_path, 'w') as file:
        file.write(df.to_string())

def save_normality_results(results, averages, overall_normality, filename):
    with open(filename, 'w') as file:
        file.write('Normality Test Results:\n\n')
        for column, tests in results.items():
            file.write(f'Column: {column}\n')
            for test, result in tests.items():
                file.write(f'  {test}:\n')
                for key, value in result.items():
                    file.write(f'    {key}: {value}\n')
            file.write('\n')
        
        file.write('Averages:\n')
        for test, stats in averages.items():
            file.write(f'  {test}:\n')
            for key, value in stats.items():
                file.write(f'    {key}: {value}\n')
        
        file.write('\nOverall Normality:\n')
        for test, is_normal in overall_normality.items():
            file.write(f'  {test}: {"Normal" if is_normal else "Not Normal"}\n')

'''
def perform_regression(df, x_column_names, y_column_name):
    regression_results = []

    for x_column_name in x_column_names:
        linear_coefficients = np.polyfit(df[x_column_name], df[y_column_name], 1)

        def nonlinear_func(x, a, b, c, d):
            return a * x**3 + b * x**2 + c * x + d
            popt, _ = curve_fit(nonlinear_func, df[x_column_name], df[y_column_name])

        result = {
            'X_Column': x_column_name,
            'Y_Column': y_column_name,
            'Regression': ['Linear', 'Nonlinear'],
            'Coefficient_1': [linear_coefficients[0], popt[0]],
            'Coefficient_2': [linear_coefficients[1], popt[1]],
            'Coefficient_3': [0, popt[2]],
            'Coefficient_4': [0, popt[3]],
            'Function_Equation_Specific': [f'{linear_coefficients[1]}x + {linear_coefficients[0]}', f'{popt[3]}x^3 + {popt[2]}x^2 + {popt[1]}x + {popt[0]}'],
            'Function_Equation_General': ['ax + b', 'ax^3 + bx^2 + cx + d']
        }
        regression_results.append(pd.DataFrame(result))

    final_results = pd.concat(regression_results, ignore_index=True)
    return final_results
'''
def heatmap(corr_matrix, t_matrix, p_matrix, rückenlehnenteil, fahrzeug, method):
    cmap = plt.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=-1, vmax=1)

    plt.figure(figsize=(10, len(corr_matrix) * 0.5))

    num_digits = corr_matrix.applymap(lambda x: len(str(round(x, 3))))
    max_digits = num_digits.max().max()
    cell_width = max_digits * 0.3
    heatmap_width = corr_matrix.shape[1] * cell_width

    plt.figure(figsize=(heatmap_width, 8))

    heatmap = plt.imshow(corr_matrix, cmap=cmap, norm=norm, aspect='auto', alpha=0.3)
    plt.colorbar(heatmap, orientation='vertical')
    
    plt.xticks(np.arange(corr_matrix.shape[1]), corr_matrix.columns, rotation=90)
    plt.yticks(np.arange(corr_matrix.shape[0]), corr_matrix.index)
    plt.title(f'Teil-Korrelationsmatrix Heatmap ({method})')

    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            corr_value = round(corr_matrix.iloc[i, j], 3)
            t_value = round(t_matrix.iloc[i, j], 3)
            p_value = p_matrix.iloc[i, j]
            if pd.isna(p_value):
                p_value_formatted = 'NaN'
                plt.text(j, i, f"r={corr_value}\nU={t_value}\np={p_value_formatted}", ha='center', va='center', color='black')
            else:
                p_value_formatted = f"{p_value:.3e}"
                if p_value > 0.05 or abs(t_value) < 2:
                    plt.gca().add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, color='grey', alpha=1))
                plt.text(j, i, f"r={corr_value}\nt={t_value}\np={p_value_formatted}", ha='center', va='center', color='black')

    plt.tight_layout()
    plt.savefig(f'./heatmaps_2/partial_correlation_matrix_heatmap_{fahrzeug}_{rückenlehnenteil}_{method}.png')
    plt.close()

def chi2_test(data, bins=10):
    observed_freq, bin_edges = np.histogram(data, bins=bins)
    expected_freq = len(data) * np.diff(stats.norm.cdf(bin_edges))
    expected_freq = expected_freq * observed_freq.sum() / expected_freq.sum()
    chi2_stat, p_value = stats.chisquare(observed_freq, expected_freq)
    return chi2_stat, p_value

def chi2_test(data, bins=10):
    observed_freq, bin_edges = np.histogram(data, bins=bins)
    expected_freq = len(data) * np.diff(stats.norm.cdf(bin_edges))
    expected_freq = expected_freq * observed_freq.sum() / expected_freq.sum()  # Scale expected frequencies
    chi2_stat, p_value = stats.chisquare(observed_freq, expected_freq)
    return chi2_stat, p_value

def check_normal_distribution(data):
    # Replace NaNs with 0
    data = data.fillna(0)
    
    encoded_data, _ = encode_parameters(data)
    results = {}
    averages = {
        'Shapiro-Wilk': {'Statistic': 0, 'p-value': 0},
        'Kolmogorov-Smirnov': {'Statistic': 0, 'p-value': 0},
        'Anderson-Darling': {'Statistic': 0, 'Critical Value': 0},
        'D\'Agostino\'s K-squared': {'Statistic': 0, 'p-value': 0},
        'Chi-Squared': {'Statistic': 0, 'p-value': 0}
    }
    n = len(encoded_data.columns)
    
    for column in encoded_data.columns:
        shapiro_test = stats.shapiro(encoded_data[column])
        ks_test = stats.kstest(encoded_data[column], 'norm')
        anderson_test = stats.anderson(encoded_data[column], dist='norm')
        dagostino_test = stats.normaltest(encoded_data[column])
        chi2_stat, chi2_p_value = chi2_test(encoded_data[column])

        critical_value = anderson_test.critical_values[2]  # 5% significance level

        results[column] = {
            'Shapiro-Wilk': {
                'Statistic': shapiro_test.statistic, 
                'p-value': shapiro_test.pvalue, 
                'Significant': shapiro_test.pvalue < 0.05
            },
            'Kolmogorov-Smirnov': {
                'Statistic': ks_test.statistic, 
                'p-value': ks_test.pvalue, 
                'Significant': ks_test.pvalue < 0.05
            },
            'Anderson-Darling': {
                'Statistic': anderson_test.statistic, 
                'Critical Values': anderson_test.critical_values, 
                'Significance Levels': anderson_test.significance_level, 
                'Significant': anderson_test.statistic > critical_value  # 5% significance level
            },
            'D\'Agostino\'s K-squared': {
                'Statistic': dagostino_test.statistic, 
                'p-value': dagostino_test.pvalue, 
                'Significant': dagostino_test.pvalue < 0.05
            },
            'Chi-Squared': {
                'Statistic': chi2_stat,
                'p-value': chi2_p_value,
                'Significant': chi2_p_value < 0.05
            }
        }

        # Add up for averages, ignoring NaNs
        if not np.isnan(shapiro_test.statistic):
            averages['Shapiro-Wilk']['Statistic'] += shapiro_test.statistic / n
        if not np.isnan(shapiro_test.pvalue):
            averages['Shapiro-Wilk']['p-value'] += shapiro_test.pvalue / n
        if not np.isnan(ks_test.statistic):
            averages['Kolmogorov-Smirnov']['Statistic'] += ks_test.statistic / n
        if not np.isnan(ks_test.pvalue):
            averages['Kolmogorov-Smirnov']['p-value'] += ks_test.pvalue / n
        if not np.isnan(anderson_test.statistic):
            averages['Anderson-Darling']['Statistic'] += anderson_test.statistic / n
        if not np.isnan(dagostino_test.statistic):
            averages['D\'Agostino\'s K-squared']['Statistic'] += dagostino_test.statistic / n
        if not np.isnan(dagostino_test.pvalue):
            averages['D\'Agostino\'s K-squared']['p-value'] += dagostino_test.pvalue / n
        if not np.isnan(chi2_stat):
            averages['Chi-Squared']['Statistic'] += chi2_stat / n
        if not np.isnan(chi2_p_value):
            averages['Chi-Squared']['p-value'] += chi2_p_value / n
        if not np.isnan(critical_value):
            averages['Anderson-Darling']['Critical Value'] += critical_value / n

    # Determine overall normality based on average p-values
    overall_normality = {
        'Shapiro-Wilk': averages['Shapiro-Wilk']['p-value'] >= 0.05,
        'Kolmogorov-Smirnov': averages['Kolmogorov-Smirnov']['p-value'] >= 0.05,
        'Anderson-Darling': averages['Anderson-Darling']['Statistic'] <= averages['Anderson-Darling']['Critical Value'],
        'D\'Agostino\'s K-squared': averages['D\'Agostino\'s K-squared']['p-value'] >= 0.05,
        'Chi-Squared': averages['Chi-Squared']['p-value'] >= 0.05
    }
    
    return results, averages, overall_normality

def check_dichte_parameter_correlation(directory, threshold, log_file):
    logging.basicConfig(filename=log_file, level=logging.INFO)

    dichte_parameter_correlations = {}
    all_parameters = set()

    for file_name in os.listdir(directory):
        if file_name.startswith("Korrelationsmatrix__V") and file_name.endswith(".txt"):
            _, fahrzeug, rueckenlehnenteil = file_name.split("__")
            rueckenlehnenteil = rueckenlehnenteil.split(".")[0]

            correlation_matrix = pd.read_csv(os.path.join(directory, file_name), sep='\s+')
            all_parameters.update(correlation_matrix.columns)
            dichte_parameter_correlations[f'{fahrzeug}__{rueckenlehnenteil}'] = [correlation_matrix.loc['Dichtefaktor']]

    indices_sets = [set(df[0].index) for df in dichte_parameter_correlations.values()]
    common_keys = set.intersection(*indices_sets)

    if len(common_keys) == 0:
        logging.info("Es gibt keinen klaren erkennbaren Zusammenhang von Parametern")
    else:
        sorted_keys = sorted(common_keys, key=lambda x: sum(abs(entry[0][x]) for entry in dichte_parameter_correlations.values()), reverse=True)
        
        equal_corr_statements = []
        unequal_corr_statements = []
        for key in sorted_keys:
            correlation_values = [entry[0][key] for entry in dichte_parameter_correlations.values()]
            correlation_values = [float(value) for value in correlation_values if not pd.isna(value)]
            avg_correlation = sum(correlation_values) / len(correlation_values)
            correlation_close = all(abs(value - avg_correlation) < threshold for value in correlation_values)
            if correlation_close:
                equal_corr_statements.append(f"Der Korrelationsfaktor für '{key}' ist überall ungefähr gleich.")
            else:
                unequal_corr_statements.append(f"Der Korrelationsfaktor für '{key}' ist nicht überall ungefähr gleich.")

        equal_corr_statements.sort()
        unequal_corr_statements.sort()

        for statement in equal_corr_statements:
            logging.info(statement)
        logging.info("=" * 50)
        for statement in unequal_corr_statements:
            logging.info(statement)
    
    return common_keys

def calculate_correlations_and_tests(data):
    correlations = {
        'Pearson': data.corr(method='pearson'),
        'Spearman': data.corr(method='spearman'),
        'Kendall': data.corr(method='kendall')
    }
            
    t_values = {method: pd.DataFrame(index=data.columns, columns=data.columns) for method in correlations.keys()}
    p_values = {method: pd.DataFrame(index=data.columns, columns=data.columns) for method in correlations.keys()}
    
    for method, corr_matrix in correlations.items():
        for i, col1 in enumerate(data.columns):
            for j, col2 in enumerate(data.columns):
                if i != j:
                    col1_data = data[col1].replace([np.inf, -np.inf], np.nan).dropna()
                    col2_data = data[col2].replace([np.inf, -np.inf], np.nan).dropna()    
                    common_indices = col1_data.index.intersection(col2_data.index)
                    col1_data = col1_data.loc[common_indices]
                    col2_data = col2_data.loc[common_indices]                
                    
                    if col1_data.nunique() > 1 and col2_data.nunique() > 1:
                        if method == 'Pearson':
                            corr, _ = pearsonr(col1_data, col2_data)
                        elif method == 'Spearman':
                            corr, _ = spearmanr(col1_data, col2_data)
                        elif method == 'Kendall':
                            corr, _ = kendalltau(col1_data, col2_data)
                    else:
                        corr = np.nan
                    
                    if col1_data.nunique() > 1 and col2_data.nunique() > 1:
                        stat, p = ttest_ind(col1_data, col2_data)
                    else:
                        stat, p = np.nan, np.nan

                    t_values[method].loc[col1, col2] = stat
                    p_values[method].loc[col1, col2] = p
    
    return correlations, t_values, p_values

def save_and_plot_correlations(combined_data, parameter_columns, results_columns, rückenlehnenteil, fahrzeug, method_prefix):
    correlations, t_values, p_values = calculate_correlations_and_tests(combined_data)

    for method, corr_matrix in correlations.items():
        partial_correlation_matrix = corr_matrix.loc[parameter_columns, results_columns]
        t_matrix = t_values[method].loc[parameter_columns, results_columns]
        p_matrix = p_values[method].loc[parameter_columns, results_columns]
        
        partial_correlation_matrix_cleaned = partial_correlation_matrix.dropna(axis=0, how='all').dropna(axis=1, how='all')
        t_matrix_cleaned = t_matrix.dropna(axis=0, how='all').dropna(axis=1, how='all')
        p_matrix_cleaned = p_matrix.dropna(axis=0, how='all').dropna(axis=1, how='all')

        common_indices = partial_correlation_matrix_cleaned.index.intersection(t_matrix_cleaned.index).intersection(p_matrix_cleaned.index)
        common_columns = partial_correlation_matrix_cleaned.columns.intersection(t_matrix_cleaned.columns).intersection(p_matrix_cleaned.columns)


        partial_correlation_matrix_cleaned = partial_correlation_matrix_cleaned.loc[common_indices, common_columns].dropna(axis=0, how='any').dropna(axis=1, how='any')
        t_matrix_cleaned = t_matrix_cleaned.loc[common_indices, common_columns].dropna(axis=0, how='any').dropna(axis=1, how='any')
        p_matrix_cleaned = p_matrix_cleaned.loc[common_indices, common_columns].dropna(axis=0, how='any').dropna(axis=1, how='any')


        save_df_to_txt(partial_correlation_matrix_cleaned, f'/Korrelationsmatrix_2/Korrelationsmatrix_{method_prefix}_{method}')
        heatmap(partial_correlation_matrix_cleaned, t_matrix_cleaned, p_matrix_cleaned, rückenlehnenteil, fahrzeug, method)

def encode_parameters(parameters):
    label_encoders = {}
    encoded_parameters = parameters.copy()
    
    for column in parameters.columns:
        if parameters[column].dtype == 'object':
            le = LabelEncoder()
            encoded_parameters[column] = le.fit_transform(parameters[column])
            label_encoders[column] = le
    
    return encoded_parameters, label_encoders

def analyze_group(group, parameter_columns, Werte_Untersuchung, rückenlehnenteil, fahrzeug):
    parameters = group[parameter_columns]
    results = group.drop(columns=parameter_columns)

    encoded_parameters, _ = encode_parameters(parameters)
    combined_data = pd.concat([encoded_parameters, results], axis=1)
    save_and_plot_correlations(combined_data, parameter_columns, results.columns, rückenlehnenteil, fahrzeug, f'{fahrzeug}__{rückenlehnenteil}')

    '''
    all_results = pd.DataFrame()
    for y_column in Werte_Untersuchung:
        regression_results = perform_regression(group, ['Dichtefaktor'], y_column)
        all_results = pd.concat([all_results, regression_results], axis=1)

    save_df_to_txt(all_results, f'Ergebnis_Funktionsregression__{fahrzeug}__{rückenlehnenteil}')
    '''

def data_analysis(file_path, directory, log_file, parameter_columns, Werte_Untersuchung):
    data = pd.read_csv(file_path, sep='\s+')
    grouped_data = data.groupby(['Rückenlehnenteil', 'Modell'])

    parameters = data[parameter_columns]
    results, averages, overall_normality = check_normal_distribution(data)
    save_normality_results(results, averages, overall_normality, 'Ergebnis_Normalverteilung.txt')
    print("Test fuer Datennormalverteilung abgeschlossen")
    encoded_parameters, _ = encode_parameters(parameters)
    combined_data = pd.concat([encoded_parameters, data.drop(columns=parameter_columns)], axis=1)
    save_and_plot_correlations(combined_data, parameter_columns, data.drop(columns=parameter_columns).columns, 'Alle_Lehnen', 'Alle_Fahrzeuge', 'Alle')

    for (rückenlehnenteil, fahrzeug), group in grouped_data:
        analyze_group(group, parameter_columns, Werte_Untersuchung, rückenlehnenteil, fahrzeug)
    
    check_dichte_parameter_correlation(directory, 0.1, log_file)

def main():
    file_path = './Datenauswertung_teilweise_1.txt'
    directory = './'
    log_file = './log.log'
    parameter_columns = ['Modell', 'Rückenlehne', 'Dichtefaktor', 'Rückenlehnenteil', 'CNCAP_NIC_max', 'CNCAP_NIC_min']
    Werte_Untersuchung = ['T_Kopfkontakt', 'Energie_Gesamt_max', 'BIO_xiii_1_avg', 'Federrate_avg', 'CNCAP_NIC_max', 'Bewertung_CNCAP_gesamt']

    data_analysis(file_path, directory, log_file, parameter_columns, Werte_Untersuchung)

if __name__ == "__main__":
    main()

