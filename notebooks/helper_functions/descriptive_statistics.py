import os
import pandas as pd
from scipy.stats import kurtosis, entropy
import matplotlib.pyplot as plt
from jinja2 import Template

def calculate_numerical_stats(col_data):
    """
    Calculate statistics for a numerical column.

    Parameters:
    - col_data (pd.Series): The numerical column to analyze.

    Returns:
    - dict: A dictionary containing calculated statistics.
    """
    return {
        'Count': col_data.count(),
        'Mean': round(col_data.mean(), 2),
        'Std Dev': round(col_data.std(), 2),
        'Min': round(col_data.min(), 2),
        '25%': round(col_data.quantile(0.25), 2),
        '50% (Median)': round(col_data.quantile(0.5), 2),
        '75%': round(col_data.quantile(0.75), 2),
        'Max': round(col_data.max(), 2),
        'Skewness': round(col_data.skew(), 2),
        'Kurtosis': round(kurtosis(col_data, nan_policy='omit'), 2)
    }

def descriptive_statistics(data, categorical_columns=None, numerical_columns=None, segment_by=None, show_output=True):
    """
    Calculate descriptive statistics for numerical and categorical columns in a dataset,
    and optionally calculate segmented statistics for numerical columns grouped by categorical columns.

    Parameters:
    - data (pd.DataFrame): The dataset to analyze.
    - categorical_columns (list, optional): List of categorical feature names.
    - numerical_columns (list, optional): List of numerical feature names.
    - segment_by (list, optional): List of categorical columns to group numerical columns by.
    - show_output (bool, optional): Whether to print the results. Default is True.

    Returns:
    - dict: Contains two keys:
        - 'descriptive_statistics': A DataFrame with general descriptive statistics.
        - 'segmented_statistics': A dictionary of segmented statistics by category.
    """
    # General descriptive statistics
    stats = []
    # Numerical columns
    if numerical_columns and not segment_by:
        for column in numerical_columns:
            col_data = data[column]
            stats.append({
                'Column': column,
                'Type': 'Numerical',
                **calculate_numerical_stats(col_data)
            })

    # Categorical columns
    if categorical_columns:
        for column in categorical_columns:
            col_data = data[column]
            value_counts = col_data.value_counts()
            stats.append({
                'Column': column,
                'Type': 'Categorical',
                'Count': col_data.count(),
                'Unique': col_data.nunique(),
                'Most Frequent': value_counts.idxmax() if not value_counts.empty else None,
                'Counts of Most Frequent': value_counts.max() if not value_counts.empty else None,
                'Least Frequent': value_counts.idxmin() if len(value_counts) > 0 else None,
                'Counts of Least Frequent': value_counts.min() if len(value_counts) > 0 else None,
                'Base 2 Entropy': round(entropy(value_counts, base=2), 2) if not value_counts.empty else None
            })
    descriptive_stats_df = pd.DataFrame(stats)

    # Segmented statistics
    segmented_stats = {}
    if segment_by and numerical_columns:
        for num_col in numerical_columns:
            segmented_data = {}
            for category, group in data.groupby(segment_by):
                col_data = group[num_col].dropna()

                # Clean up category for user-friendly keys
                if isinstance(category, tuple):
                    category_key = "_".join(map(str, category)).replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
                else:
                    category_key = str(category).replace(" ", "_")

                segmented_data[category_key] = calculate_numerical_stats(col_data)

            # Convert to DataFrame
            stats_df = pd.DataFrame(segmented_data).T
            segmented_stats[num_col] = stats_df

            # Optionally print segmented stats
            if show_output:
                print(f"\nStatistics for {num_col} segmented by {segment_by}:")
                print(stats_df.to_string())

    # Return combined results
    return {
        'descriptive_statistics': descriptive_stats_df,
        'segmented_statistics': segmented_stats
    }

def plot_histogram(data, column, title, xlabel, ylabel, file_path, bins=30, kind="bar", xticks_rotation=0):
    """
    Plots and saves a histogram.

    Parameters:
    - data: Data to plot (Pandas Series or dictionary).
    - column: Column name (used for the title and labeling).
    - title: Title of the histogram.
    - xlabel: X-axis label.
    - ylabel: Y-axis label.
    - file_path: Path to save the plot.
    - bins: Number of bins (for numerical data). Default is 30.
    - kind: "bar" for categorical data or "hist" for numerical data.
    - xticks_rotation: Rotation angle for x-axis ticks (for categorical data).
    """
    plt.figure(figsize=(8, 6))
    if kind == "hist":
        plt.hist(data, bins=bins, edgecolor="black", alpha=0.7)
    elif kind == "bar":
        plt.bar(data.index, data.values, edgecolor="black", alpha=0.7)
        plt.xticks(rotation=xticks_rotation, ha='right')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()

def generate_histograms(data, categorical_columns=None, numerical_columns=None, segment_by=None, output_dir="output_images"):
    """
    Generates histograms for numerical and/or low-cardinality categorical columns in a DataFrame.
    Optionally generates grouped histograms for numerical columns by a given categorical column.

    Parameters:
    - data (pd.DataFrame): The input DataFrame.
    - categorical_columns (list, optional): List of categorical columns to generate histograms for.
    - numerical_columns (list, optional): List of numerical columns to generate histograms for.
    - group_by_column (str, optional): A categorical column to group numerical histograms by.
    - output_dir (str, optional): Directory to save histograms. Default is "output_images".

    Returns:
    - list: A list of tuples (column_type, column_name, category, file_path) for generated histograms.
    """
    os.makedirs(output_dir, exist_ok=True)
    histogram_paths = []

    # Generate ungrouped numerical histograms
    if numerical_columns and not segment_by:
        for num_col in numerical_columns:
            file_path = os.path.join(output_dir, f"{num_col}_numerical.png")
            plot_histogram(
                data=data[num_col].dropna(),
                column=num_col,
                title=f"Histogram of {num_col} (Numerical)",
                xlabel=num_col,
                ylabel="Frequency",
                file_path=file_path,
                kind="hist",
            )
            histogram_paths.append(("numerical", num_col, None, file_path))

    # Generate ungrouped categorical histograms
    if categorical_columns and not segment_by:
        for cat_col in categorical_columns:
            value_counts = data[cat_col].value_counts()
            file_path = os.path.join(output_dir, f"{cat_col}_categorical.png")
            plot_histogram(
                data=value_counts,
                column=cat_col,
                title=f"Histogram of {cat_col} (Categorical)",
                xlabel='',
                ylabel="Frequency",
                file_path=file_path,
                kind="bar",
                xticks_rotation=30,
            )
            histogram_paths.append(("categorical", cat_col, None, file_path))

    # Generate segmented histograms if a segment_by is provided
    if segment_by and numerical_columns:
        for num_col in numerical_columns:
            for category, group in data.groupby(segment_by):
                col_data = group[num_col].dropna()
                # Clean up category for user-friendly keys
                if isinstance(category, tuple):
                    category_key = "_".join(map(str, category)).replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
                else:
                    category_key = str(category).replace(" ", "_")

                file_path = os.path.join(output_dir, f"{segment_by}_{num_col}_{category_key}.png")
                plot_histogram(
                    data=col_data,
                    column=num_col,
                    title=f"Histogram of {num_col} for {segment_by} = {category_key}",
                    xlabel=num_col,
                    ylabel="Frequency",
                    file_path=file_path,
                    kind="hist",
                )
                histogram_paths.append(("segmented", num_col, category_key, file_path))

    return histogram_paths

def generate_html_report(
    all_data,
    numerical_stats_func=None,
    categorical_stats_func=None,
    segmented_stats_funcs=None,  # Accept a list of segmented stats functions
    numerical_histograms_func=None,
    categorical_histograms_func=None,
    segmented_histograms_funcs=None,  # Accept a list of segmented histogram functions
    output_dir="output_images",
    output_file="report.html"
):
    """
    Generate an HTML report for descriptive statistics and histograms.

    Parameters:
    - all_data (pd.DataFrame): The input DataFrame.
    - numerical_stats_func (callable, optional): Function to calculate numerical descriptive statistics.
    - categorical_stats_func (callable, optional): Function to calculate categorical descriptive statistics.
    - segmented_stats_funcs (list, optional): List of functions to calculate segmented statistics.
    - numerical_histograms_func (callable, optional): Function to generate numerical histograms.
    - categorical_histograms_func (callable, optional): Function to generate categorical histograms.
    - segmented_histograms_funcs (list, optional): List of functions to generate segmented histograms.
    - output_dir (str, optional): Directory to save histogram images. Default is "output_images".
    - output_file (str, optional): File name for the HTML report. Default is "report.html".
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect descriptive statistics
    numerical_stats = (
        numerical_stats_func(all_data) if numerical_stats_func else pd.DataFrame()
    )
    categorical_stats = (
        categorical_stats_func(all_data) if categorical_stats_func else pd.DataFrame()
    )

    # Collect segmented statistics from multiple functions
    segmented_stats = {}
    if segmented_stats_funcs:
        for func in segmented_stats_funcs:
            func_result = func(all_data)
            if func_result:  # Ensure result is not empty
                segmented_stats.update(func_result)

    # Generate histograms
    numerical_histograms = (
        numerical_histograms_func(all_data, output_dir=output_dir)
        if numerical_histograms_func
        else []
    )
    categorical_histograms = (
        categorical_histograms_func(all_data, output_dir=output_dir)
        if categorical_histograms_func
        else []
    )

    # Collect segmented histograms from multiple functions
    segmented_histograms = []
    if segmented_histograms_funcs:
        for func in segmented_histograms_funcs:
            func_result = func(all_data, output_dir=output_dir)
            if func_result:  # Ensure result is not empty
                segmented_histograms.extend(func_result)

    # Merge all histogram paths
    all_histograms = numerical_histograms + categorical_histograms + segmented_histograms

    # Load Jinja2 template
    template = Template("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Conversion Data Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2, h3 { color: #2c3e50; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 30px; }
            table, th, td { border: 1px solid #ddd; }
            th, td { text-align: left; padding: 8px; }
            th { background-color: #f2f2f2; }
            img { max-width: 100%; height: auto; margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <h1>Data Analysis Report</h1>

        {% if numerical_stats is not none and not numerical_stats.empty %}
        <h2>Numerical Descriptive Statistics</h2>
        <table>
            <thead>
                <tr>
                    {% for col in numerical_stats.columns %}
                        <th>{{ col }}</th>
                    {% endfor %}
                </tr>
            </thead>
            <tbody>
                {% for _, row in numerical_stats.iterrows() %}
                <tr>
                    {% for cell in row %}
                        <td>{{ cell }}</td>
                    {% endfor %}
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% endif %}

        {% if categorical_stats is not none and not categorical_stats.empty %}
        <h2>Categorical Descriptive Statistics</h2>
        <table>
            <thead>
                <tr>
                    {% for col in categorical_stats.columns %}
                        <th>{{ col }}</th>
                    {% endfor %}
                </tr>
            </thead>
            <tbody>
                {% for _, row in categorical_stats.iterrows() %}
                <tr>
                    {% for cell in row %}
                        <td>{{ cell }}</td>
                    {% endfor %}
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% endif %}

        {% if segmented_stats is not none and segmented_stats %}
        <h2>Segmented Statistics</h2>
        {% for segment, stats_df in segmented_stats.items() %}
            <h3>Segmented by {{ segment }}</h3>
            <table>
                <thead>
                    <tr>
                        <th>Category</th>
                        {% for col in stats_df.columns %}
                            <th>{{ col }}</th>
                        {% endfor %}
                    </tr>
                </thead>
                <tbody>
                    {% for index, row in stats_df.iterrows() %}
                    <tr>
                        <td>{{ index }}</td>
                        {% for cell in row %}
                            <td>{{ cell }}</td>
                        {% endfor %}
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        {% endfor %}
        {% endif %}

        <h2>Histograms</h2>
        {% for hist_type, column_name, category, file_path in histograms %}
            <h3>{{ hist_type | capitalize }} Histogram: {{ column_name }}{% if category %} ({{ category }}){% endif %}</h3>
            <img src="{{ file_path }}" alt="Histogram for {{ column_name }}">
        {% endfor %}
    </body>
    </html>
    """)

    # Render the HTML
    html_content = template.render(
        numerical_stats=numerical_stats,
        categorical_stats=categorical_stats,
        segmented_stats=segmented_stats,
        histograms=all_histograms
    )

    # Save the HTML report
    with open(output_file, "w") as file:
        file.write(html_content)

    print(f"HTML report generated: {output_file}")