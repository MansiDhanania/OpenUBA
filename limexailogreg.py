# LIME - Explainable AI
def limeplotlogreg(data, inputnumber):

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
  data = data_copy

  # Split data into training and testing sets
  from sklearn.model_selection import train_test_split
  # Initialize X and Y variables
  X = data.drop('insider', axis=1)
  Y = data['insider']

  # Training and Testing sets (stratified to ensure both classes present)
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

  # Sample training data for performance (10K instances)
  import numpy as np
  if len(X_train) > 10000:
    sample_indices = np.random.choice(len(X_train), size=10000, replace=False)
    X_train = X_train.iloc[sample_indices]
    Y_train = Y_train.iloc[sample_indices]

  # Scale the dataset
  from sklearn.preprocessing import StandardScaler
  # Initialize the scaler
  scaler = StandardScaler()
  # Fit to dataset
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  # Logistic Regression
  from sklearn.linear_model import LogisticRegression
  # Initialize the model
  logreg = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=16)
  # Train the model
  logreg.fit(X_train, Y_train)

  # Import LIME
  import lime
  import lime.lime_tabular

  # Scale X for LIME (model expects scaled data)
  X_scaled = scaler.transform(X)

  # Initialize
  explainer = lime.lime_tabular.LimeTabularExplainer(X_scaled, feature_names=X.columns.values.tolist(),
                                                     class_names=['insider'], verbose=True,
                                                     mode='classification')

  # Create Explanation for one input
  exp = explainer.explain_instance(X_scaled[inputnumber], logreg.predict_proba, num_features=5)
  limeplot = exp.show_in_notebook(show_table=True)

  return limeplot