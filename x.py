import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# Data brute de 15 zones industrielles clés au Maroc
np.random.seed(42)
data_immobilier = {
    'Superficie_Zone_M2':np.random.uniform(1000, 9000, 15),
    'Distance_Port_KM':   [12.5,  45.0,  8.2,  75.5,  22.1,  35.4, 14.8,  90.0,  5.1,  55.2, 19.3,  68.4, 31.0,  40.1,  28.7],
    'Prix_Loyer_MAD_Mois': [45000, 98000, 38000, 185000, 68000, 120000, 52000, 210000, 49000, 145000, 61000, 170000, 78000, 112000, 91000]
}

# Fabrication du DataFrame principal
df = pd.DataFrame(data_immobilier)

print("Le tableau brute 'df' est initialisé  !")
print(df.head())

#tache 1 : visualisation de data:

plt.figure(figsize=(10,6))
df_sorted = df.sort_values(by="Superficie_Zone_M2")
plt.plot(df_sorted["Superficie_Zone_M2"], df_sorted["Prix_Loyer_MAD_Mois"], color="red", marker="o", linestyle="--", linewidth=2)
plt.title("Analyse logistique: relation entre la superficie et le prix du loyer")
plt.xlabel("Superficie de la zone industuelle")
plt.ylabel("Prix du loyer")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
print("_"*100)

# tache 2 : separation de data:

X = df[["Superficie_Zone_M2", "Distance_Port_KM"]]    #input
y = df[["Prix_Loyer_MAD_Mois"]]            #output
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train)
print("_"*100)

# tahce 3 : scaling :

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#tache 4 : training final

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train_scaled, y_train)

# tache 4 : diagnostic final :

y_pred = rf_model.predict(X_test_scaled)
score = r2_score(y_pred, y_test)
mae = mean_absolute_error(y_pred, y_test)
print(f"R² = {score*100:.2f}%")
print(f"l'erreur moyenne absolue est : {mae:.2f}")