import pandas as pd
import numpy as np
import pygad
from lifelines import CoxPHFitter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. UPLOAD AND READ DATA
# Replace with your local path or Kaggle path
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# 2. PREPROCESSING
# Drop ID and handle missing BMI values
df = df.drop(columns=['id'])
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# Encode categorical features
le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# For Cox Model, we need: Time (Age) and Event (Stroke)
# Features: everything except 'age' and 'stroke'
X = df.drop(columns=['stroke', 'age'])
y_time = df['age']
y_event = df['stroke']

# Split data
X_train, X_test, y_time_train, y_time_test, y_event_train, y_event_test = train_test_split(
    X, y_time, y_event, test_size=0.2, random_state=42
)

# Combine for lifelines compatibility
train_df = pd.concat([X_train, y_time_train, y_event_train], axis=1)
test_df = pd.concat([X_test, y_time_test, y_event_test], axis=1)

# 3. GENETIC ALGORITHM FOR FEATURE SELECTION
def fitness_func(ga_instance, solution, solution_idx):
    # Select features based on GA chromosome (binary 1s and 0s)
    selected_features = [X.columns[i] for i in range(len(solution)) if solution[i] == 1]
    
    if len(selected_features) == 0:
        return 0
    
    try:
        cph = CoxPHFitter(penalizer=0.1)
        # Use selected features + time and event columns
        cols_to_use = selected_features + ['age', 'stroke']
        cph.fit(train_df[cols_to_use], duration_col='age', event_col='stroke')
        
        # Concordance index is our fitness metric (higher is better)
        return cph.concordance_index_
    except:
        return 0

# Configure GA
ga_instance = pygad.GA(
    num_generations=20,
    num_parents_mating=5,
    fitness_func=fitness_func,
    sol_per_pop=10,
    num_genes=len(X.columns),
    gene_space=[0, 1], # Binary selection
    parent_selection_type="sss",
    crossover_type="single_point",
    mutation_type="random",
    mutation_percent_genes=10
)

print("Starting Genetic Algorithm Optimization...")
ga_instance.run()

# 4. BEST MODEL & EVALUATION
solution, solution_fitness, solution_idx = ga_instance.best_solution()
best_features = [X.columns[i] for i in range(len(solution)) if solution[i] == 1]

print(f"\nBest Features Selected: {best_features}")
print(f"Best Training Concordance Index: {solution_fitness:.4f}")

# Final Train and Prediction
final_cph = CoxPHFitter()
final_cols = best_features + ['age', 'stroke']
final_cph.fit(train_df[final_cols], duration_col='age', event_col='stroke')

# Evaluation on Test Set
test_c_index = final_cph.score(test_df[final_cols], scoring_method="concordance_index")
print(f"Test Set Concordance Index: {test_c_index:.4f}")

# Make Risk Predictions (Partial Hazards)
predictions = final_cph.predict_partial_hazard(test_df[best_features])
print("\nSample Risk Scores (Partial Hazard):")
print(predictions.head())
