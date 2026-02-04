# SHAP - Explainable AI
def shapplotisoforest(data):

  # Import Necessary Libraries
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt

  # Read the data
  data = pd.read_csv(data)

  # Label all insiders as '1'
  data_copy = data.copy()
  data_copy.loc[data['insider']==2, 'insider'] = 1
  data_copy.loc[data['insider']==3, 'insider'] = 1
  df = data_copy

  # Split data into training and testing sets
  from sklearn.model_selection import train_test_split
  # Initialize X and Y variables
  X = df.drop('insider', axis=1)
  Y = df['insider']

  # Training and Testing sets (stratified to ensure both classes present)
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

  # Sample training data for performance (10K instances)
  import numpy as np
  if len(X_train) > 10000:
    sample_indices = np.random.choice(len(X_train), size=10000, replace=False)
    X_train = X_train.iloc[sample_indices]
    Y_train = Y_train.iloc[sample_indices]

  # Isolation Forest
  from sklearn.ensemble import IsolationForest
  # Initialize the model
  isoforest = IsolationForest(random_state=16)
  # Train the model
  isoforest.fit(X_train, Y_train)

  # Import SHAP
  import shap

  # Sample a small subset for README/visualization purposes (computing SHAP for entire test set takes hours)
  sample_size = min(100, len(X_test))
  X_test_sample = X_test.sample(n=sample_size, random_state=42)

  # Create a SHAP explainer for the model
  explainer = shap.Explainer(isoforest.predict, X_test_sample, max_evals=1000)

  # Compute SHAP values for the sample
  shap_values = explainer(X_test_sample)

  # Plot
  # shapbarplot = shap.plots.bar(shap_values)
  # shapsummary = shap.summary_plot(shap_values, plot_type='violin')
  shapbeeswarm = shap.plots.beeswarm(shap_values)
  # shapwaterfall = shap.plots.waterfall(shap_values[1])

  # Creating Object
  # if isinstance(shap_values, shap.Explanation):
  #   shap_values = shap_values.values

  # Force Plot
  # shapforceplot = shap.force_plot(explainer.expected_value, shap_values[1], X_test[0, :], matplotlib=True)

  return shapbeeswarm