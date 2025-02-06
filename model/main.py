import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os


def create_model(data):
    X = data.drop(columns=['diagnosis'])
    y = data['diagnosis']

    # Standardisation des données
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Séparation en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Entraînement du modèle
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Test du modèle
    y_pred = model.predict(X_test)
    print('Accuracy of our model:', accuracy_score(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred))

    return model, scaler


def get_clean_data():
    # Chargement des données
    data = pd.read_csv(r"C:\Users\Fddkk\PycharmProjects\BreatCancer\data\data.csv")

    # Suppression des colonnes inutiles
    if 'Unnamed: 32' in data.columns:
        data = data.drop(columns=['Unnamed: 32'])
    if 'id' in data.columns:
        data = data.drop(columns=['id'])

    # Conversion des étiquettes 'M' et 'B' en valeurs numériques
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    return data


def main():
    data = get_clean_data()
    model, scaler = create_model(data)

    # Création du dossier model s'il n'existe pas
    os.makedirs('model', exist_ok=True)

    # Sauvegarde du modèle et du scaler
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)


if __name__ == '__main__':
    main()
