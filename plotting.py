import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('macosx')


def load_metrics(file_path):
    """Load metrics data from a JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def plot_avg_of_dataset(average_data, metrics: list = None):
    """
    Plot the average metrics for each category with bars for each metric, scaled to percentages.

    :param average_data: Dictionary of average metrics by category.
    """

    # Check if average_data is empty
    if not average_data:
        print("No data available to plot.")
        return

    # Extract categories and metrics
    categories = list(average_data.keys())
    all_metrics = list(next(iter(average_data.values())).keys())

    # Metrics for Plotting
    metrics, metric_names = (metrics, "_".join(metrics)) if metrics else (all_metrics, 'all_metrics')

    # Data for plotting, scaled to %
    values = {metric: [average_data[category][metric] * 100 for category in categories] for metric in metrics}

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define a colormap
    colors = plt.cm.Pastel1(np.linspace(0, 0.5, len(metrics)))

    # Set bar width and x positions with extra spacing
    width = 0.15
    x = np.arange(len(categories)) * (len(metrics) * width + 0.2)  # Adding padding between groups

    # Plot bars for each metric
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, values[metric], width=width, label=metric, color=colors[i])

    # Formatting
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_ylabel('Metric Value (%)')
    ax.set_title('Metrics for Dataset')

    # Set y-axis limit to 0-100%
    ax.set_ylim(0, np.clip(max(max(values.values())) * 1.5, 0, 100))

    # Adjust legend and layout
    ax.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout(pad=3.0)

    # Save the plot as a PNG file
    plt.savefig(f'figures/avg_dataset_{metric_names}.png', format='png', dpi=300)

    plt.show()


def plot_avg_of_all_metrics(data):
    """
    Plot the average of all metrics for each category as a single bar, scaled to percentages,
    using colors from the Pastel2 colormap.

    :param data: Dictionary of average metrics by category.
    """

    # Compute the average metric for each category, scaled to %
    categories = list(data.keys())

    avg_values = [
        np.mean(list(metrics.values())) * 100 for metrics in data.values()
    ]

    # Use colors from Pastel2 colormap
    pastel_colors = plt.cm.Pastel2(np.linspace(0, 1, len(categories)))

    # Plotting
    fig, ax = plt.subplots(figsize=(4, 6))

    # Create the bar plot with Pastel2 colors
    ax.bar(categories, avg_values, color=pastel_colors)

    # Formatting
    ax.set_ylabel(f'Metric Average (%)')
    ax.set_title('Metrics Average for Dataset')
    ax.set_ylim(0, np.clip(np.max(avg_values) * 1.5, 0, 100))
    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha='right')

    plt.tight_layout()

    # Save the plot as a PNG file
    plt.savefig(f'figures/avg_metrics.png', format='png', dpi=300)

    plt.show()


if __name__ == "__main__":
    # Define file paths
    metrics_file_path = 'processed/all_metrics.json'
    average_file_path = 'processed/average_metrics.json'

    # Load data
    metrics_data = load_metrics(metrics_file_path)
    average_data = load_metrics(average_file_path)

    # Plot data
    plot_avg_of_dataset(average_data)
    plot_avg_of_dataset(average_data, metrics=['hit_rate'])
    plot_avg_of_all_metrics(average_data)

    print(f'Plots created and saved under /figures.')
