# ML Pipeline

Project: Online Conversions Prediction
Author: Liubov Tovbin

## Installation

Use Conda environment manager.
To create a new Condas environment with all necessary installations run:

 conda create --name my_env --file requirements.txt
 
Activate the environment:

    conda activate my_env

## Execution

To execute the entire pipeline run:

 python Main.py

To review and get an explanation about each step, go to notebooks/ and examine the notebooks in the provided order.

## Results

For data analysis resulst, see the notebooks/0_Data_Exploration.ipynb

### Model Performance

Model Performance Metrics:
{
        'Accuracy': 0.83, 
        'Precision': 0.63, 
        'Recall': 0.81, '
        F1-Score': 0.71, 
        'ROC-AUC': 0.92
}

The tuned XGBoost model with all initial features is the best-performing model, improving across all metrics. An attempt to modify a feature set resulted in a much lower performance across all metrics, even compared to the baseline, untuned XGBoost.

The XGBoost tuned correctly classified 83% of the instances. It is better at avoiding false positives, with 63% precision. Avoiding false positives is critical if we want to optimize the ad spend. The model best identifies actual conversions, with 81% recall, which is essential if missing a conversion is costly. 

The 71% F1 score confirms that the tuned model is better at balancing precision and recall. The 83% ROC-AUC shows the model's ability to distinguish between positive and negative classes across all thresholds, and it is also the highest for the tuned XGBoost model.


## Improvement Areas

The possible areas of improvement include but not limited to:

1. Pipeline optimization to allow scalability.
The current pipeline version avoids unnecessary data copying and storing. However, it is possible to apply batch processing techniques for scalability.

2. Include feature selection and hyperparameters training in the pipeline. That will allow us to find the best model without manual intervention.

3. Analyse the model for possible bias more thoroughly and apply mitigation techniques if needed.

4. Evaluate if adjusting the decision threshold (default: 0.5) can further optimize precision-recall trade-offs based on your business goals.