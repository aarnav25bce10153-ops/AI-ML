import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load and prepare data
df = pd.read_csv("diabetes.csv")

# Ensure we are picking the right target (usually the last column)
X = df.iloc[:, :-1] 
y = df.iloc[:, -1]  

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale the features
# KNN is distance-based, so scaling is MANDATORY
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Finding the Optimal K (The "Elbow Method")
error_rate = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    # Using cross-validation for a more stable error estimate
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
    error_rate.append(1 - scores.mean())

# 5. Train the final model with the best K (e.g., let's say it's 11)
best_k = error_rate.index(min(error_rate)) + 1
print(f"Optimal K found: {best_k}")

knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train_scaled, y_train)

# 6. Evaluation
y_pred = knn_final.predict(X_test_scaled)
print("\n--- Model Performance ---")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
plt.figure(figsize=(10,6))
plt.plot(range(1,40), error_rate, color='blue', linestyle='dashed', marker='o')
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()