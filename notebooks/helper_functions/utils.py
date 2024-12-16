import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate

def calculate_contingency_sparsity(data):
    """
    Calculate contingency matrices between all pairs of categorical variables in a DataFrame
    and compute the percentage of zero values in each matrix.

    Parameters:
    - data (pd.DataFrame): Input DataFrame containing categorical variables.

    Returns:
    - dict: A dictionary where keys are tuples of variable pairs and values are the
            percentage of zero values in their contingency matrix.
    """
    categorical_columns = data.select_dtypes(include=['object']).columns
    sparsity_dict = {}

    for i, col1 in enumerate(categorical_columns):
        for col2 in categorical_columns[i + 1:]:  # Avoid duplicate pairs and self-pairs
            contingency_table = pd.crosstab(data[col1], data[col2])
            total_cells = contingency_table.size
            zero_cells = np.sum(contingency_table.values == 0)
            sparsity_percentage = round((zero_cells / total_cells) * 100)
            sparsity_dict[(col1, col2)] = sparsity_percentage

    return dict(sorted(sparsity_dict.items(), key=lambda item: item[1], reverse=True))

def hour_of_day_group(hour_of_week):
    """
    Convert hour of the week to hour of the day and assign a time-of-day group.

    Parameters:
    - hour_of_week (int): Hour of the week (0–167).

    Returns:
    - str: The time-of-day group.

    Raises:
    - ValueError: If hour_of_week is out of the range 0–167.
    """
    if not (0 <= hour_of_week <= 167):
        raise ValueError(f"hour_of_week out of bounds (0–167): {hour_of_week}")

    hour_of_day = hour_of_week % 24  # Convert to hour of the day
    if 0 <= hour_of_day <= 5:
        return 1
    elif 6 <= hour_of_day <= 11:
        return 2
    elif 12 <= hour_of_day <= 17:
        return 3
    elif 18 <= hour_of_day <= 23:
        return 4

def day_of_week(hour_of_week):
    """
    Determine the day of the week based on the hour of the week.

    Parameters:
    - hour_of_week (int): Hour of the week (0–167).

    Returns:
    - str: The name of the day (e.g., "Monday", "Tuesday").

    Raises:
    - ValueError: If hour_of_week is out of the range 0–167.
    """
    if not (0 <= hour_of_week <= 167):
        raise ValueError(f"hour_of_week out of bounds (0–167): {hour_of_week}")
    return hour_of_week // 24

def evaluate_fairness_by_group(y_true, y_pred, sensitive_feature_column, metric="Accuracy"):
    """
    Evaluate fairness metrics by a specified sensitive feature column and sort the results.

    Parameters:
    - y_true (pd.Series or np.array): True labels.
    - y_pred (pd.Series or np.array): Predicted labels.
    - sensitive_feature_column (pd.Series): Column specifying the sensitive feature.
    - metric (str, optional): Metric to sort the results by. Default is "Accuracy".

    Returns:
    - sorted_metrics (pd.DataFrame): Group-wise metrics sorted by the specified metric.
    - overall_metrics (dict): Overall metrics for the model.
    """
    # Compute fairness metrics
    metric_frame = MetricFrame(
        metrics={
            'Accuracy': accuracy_score,
            'Selection Rate': selection_rate
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_feature_column
    )

    # Sort group-wise metrics by the specified metric
    sorted_metrics = metric_frame.by_group.sort_values(by=metric, ascending=False)

    # Get overall metrics
    overall_metrics = metric_frame.overall

    return sorted_metrics, overall_metrics
