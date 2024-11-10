import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(bigger_parent_folder):
    """
    Loads pairwise and summarised data from JSON files in given folder structure.
    """
    extracted_data = {}

    for parent_folder in os.listdir(bigger_parent_folder):
        parent_folder_path = os.path.join(bigger_parent_folder, parent_folder)

        if os.path.isdir(parent_folder_path):
            extracted_data[parent_folder] = {'pairwise': [], 'summarised': []}

            # Load pairwise data
            pairwise_folder_path = os.path.join(parent_folder_path, 'pairwise')
            if os.path.isdir(pairwise_folder_path):
                for filename in os.listdir(pairwise_folder_path):
                    if filename.endswith('.json'):
                        with open(os.path.join(pairwise_folder_path, filename)) as f:
                            data = json.load(f)
                            df = pd.DataFrame(data)
                            extracted_data[parent_folder]['pairwise'].append(
                                (filename.replace('.json', ''), len(df.columns)))

            # Load summarised data
            summarised_folder_path = os.path.join(parent_folder_path, 'summarised')
            if os.path.isdir(summarised_folder_path):
                for filename in os.listdir(summarised_folder_path):
                    if filename.endswith('.json'):
                        with open(os.path.join(summarised_folder_path, filename)) as f:
                            data = json.load(f)
                            df = pd.DataFrame(data)
                            extracted_data[parent_folder]['summarised'].append(
                                (filename.replace('.json', ''), len(df.columns)))

    return extracted_data


def calculate_hit_rates(summarised_data, pairwise_data):
    """
    Calculates Hit Rate |A∩B| / |A|.
    """
    model_dict = {model: value for model, value in summarised_data}
    model_pairs = [(model_pair, pairwise_count / model_dict.get(model_pair.split('-')[-1], 1))
                   for model_pair, pairwise_count in pairwise_data]
    return np.array(model_pairs, dtype=[('model_pair', 'U20'), ('hit_rate', 'f4')])


def calculate_sso_coefficients(summarised_data, pairwise_data):
    """
    Calculates Szymkiewicz–Simpson Overlap Coefficient |A∩B| / min(|A|, |B|).
    """
    model_dict = {model: value for model, value in summarised_data}
    sso_coefficients = [
        (model_pair, pairwise_count / min(model_dict.get(model_name, 1), model_dict.get(reviewer_name, 1)))
        for model_pair, pairwise_count in pairwise_data
        if (model_name := model_pair.split('-')[-1]) and (reviewer_name := model_pair.split('-')[0])
    ]
    return np.array(sso_coefficients, dtype=[('model_pair', 'U20'), ('sso_coefficient', 'f4')])


def calculate_jaccard_indices(summarised_data, pairwise_data):
    """
    Calculates Jaccard Index |A∩B| / |A∪B|.
    """
    model_dict = {model: value for model, value in summarised_data}
    jaccard_indices = [
        (model_pair,
         pairwise_count / (model_dict.get(model_name, 0) + model_dict.get(reviewer_name, 0) - pairwise_count))
        for model_pair, pairwise_count in pairwise_data
        if (model_name := model_pair.split('-')[-1]) and (reviewer_name := model_pair.split('-')[0])
    ]
    return np.array(jaccard_indices, dtype=[('model_pair', 'U20'), ('jaccard_index', 'f4')])


def calculate_sd_coefficient(summarised_data, pairwise_data):
    """
    Calculates Sørensen–Dice Coefficient 2|A∩B| / (|A| + |B|).
    """
    model_dict = {model: value for model, value in summarised_data}
    sd_coefficients = [
        (model_pair, 2 * pairwise_count / (model_dict.get(model_name, 0) + model_dict.get(reviewer_name, 0)))
        for model_pair, pairwise_count in pairwise_data
        if (model_name := model_pair.split('-')[-1]) and (reviewer_name := model_pair.split('-')[0])
    ]
    return np.array(sd_coefficients, dtype=[('model_pair', 'U20'), ('sd_coefficient', 'f4')])


def combine_metrics(hit_rates, sso_coefficients, jaccard_indices, sd_coefficients):
    """
    Combines calculated metrics into a single dictionary.
    """
    all_metrics = {}
    for model_pair, hit_rate in hit_rates:
        all_metrics[model_pair] = {'hit_rate': hit_rate}

    for model_pair, sso in sso_coefficients:
        all_metrics[model_pair]['sso_coefficient'] = sso

    for model_pair, jaccard in jaccard_indices:
        all_metrics[model_pair]['jaccard_index'] = jaccard

    for model_pair, sd in sd_coefficients:
        all_metrics[model_pair]['sd_coefficient'] = sd

    return all_metrics


def main():
    bigger_parent_folder = "./extracted"
    extracted_data = load_data(bigger_parent_folder)

    all_metrics = {}
    for paper, data in extracted_data.items():
        if data['summarised'] and data['pairwise']:
            hit_rates = calculate_hit_rates(data['summarised'], data['pairwise'])
            sso_coefficients = calculate_sso_coefficients(data['summarised'], data['pairwise'])
            jaccard_indices = calculate_jaccard_indices(data['summarised'], data['pairwise'])
            sd_coefficients = calculate_sd_coefficient(data['summarised'], data['pairwise'])

            all_metrics[paper] = combine_metrics(hit_rates, sso_coefficients, jaccard_indices, sd_coefficients)

    print(all_metrics)  # Display combined metrics

    # Plotting (optional)
    for paper, metrics in all_metrics.items():
        plot_metrics(metrics, title=f"Metrics for {paper}")


def plot_metrics(metrics, title="Metrics Comparison"):
    """
    Plots the average values for each metric for a given paper.
    """
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.plot(kind='bar', figsize=(10, 6))
    plt.title(title)
    plt.ylabel('Average Values')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Run the main function
if __name__ == "__main__":
    main()