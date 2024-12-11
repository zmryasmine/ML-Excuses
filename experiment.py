import pandas as pd
import os
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Recall, AUC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, ParameterGrid
from sklearn.metrics import f1_score, roc_curve, auc, confusion_matrix, classification_report, roc_auc_score, recall_score, precision_score, RocCurveDisplay, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from skimage import io, color, transform
import mlflow.keras
import mlflow
import mlflow as mlf
import mlflow.sklearn as mlfsk
from scikeras.wrappers import KerasClassifier



def run_experiment():
    mlf.set_tracking_uri("http://127.0.0.1:5000")

    # Set up a new experiment
    mlf.set_experiment("Brain_Tumor_Classification")

    def load_images_from_csv(directory, csv_file):
        """
        Charge les images d'un répertoire et les étiquettes d'un fichier CSV.

        Parameters:
        - directory: str, chemin vers le répertoire contenant les images.
        - csv_file: str, chemin vers le fichier CSV contenant les noms d'images et leurs étiquettes.

        Returns:
        - X: numpy array de forme (nombre_échantillons, hauteur*largeur),
            contenant les données des images en vecteurs.
        - y: numpy array de forme (nombre_échantillons,), contenant les étiquettes pour chaque image.
        """
        # Charger le fichier CSV
        metadata = pd.read_csv(csv_file)

        X = []
        y = []

        # Parcourir chaque ligne du CSV
        for index, row in metadata.iterrows():
            image_name = row['image_name']
            target = row['target']

            # Créer le chemin complet vers l'image
            image_path = os.path.join(directory, image_name)

            try:
                # Chargement et prétraitement de l'image
                image = io.imread(image_path)
                image = transform.resize(image, (100, 100, 3))  # Redimensionner l'image
                image = color.rgb2gray(image)  # Convertir l'image en niveaux de gris
                X.append(image.flatten())  # Aplatir l'image en un vecteur
                y.append(target)  # Ajouter l'étiquette de l'image
            except Exception as e:
                print(f"Erreur lors du chargement de {image_path}: {e}")

        X = np.array(X)
        y = np.array(y)

        return X, y

    # Chemin vers le répertoire contenant les images et le fichier CSV
    dataset_directory = "dataset\\balanced_dataset"
    csv_file = "dataset\\balanced_metadata.csv"

    # Charger les données
    X, y = load_images_from_csv(dataset_directory, csv_file)

    # Division des données en ensembles d'entraînement et de test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)

    # Start an MLflow run
    with mlf.start_run():

        nEstimators=30
        maxDepth=None
        minSamplesSplit=2
        maxFeatures='sqrt'
        
        # Train the Random Forest model
        model = RandomForestClassifier(n_estimators=nEstimators, max_depth=maxDepth, min_samples_split=minSamplesSplit, max_features=maxFeatures, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Log parameters and metrics
        mlf.log_param("n_estimators", nEstimators)
        mlf.log_param("max_depth", maxDepth)
        mlf.log_param("min_samples_split", minSamplesSplit)
        mlf.log_param("max_features", maxFeatures)
        mlf.log_metric("accuracy", accuracy)
        mlf.log_metric("precision", precision)
        mlf.log_metric("recall", recall)
        mlf.log_metric("f1_score", f1)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")

        # Save and log the confusion matrix plot as an artifact
        cm_filepath = "confusion_matrix_rf.png"
        plt.savefig(cm_filepath)
        mlf.log_artifact(cm_filepath)

        print("Confusion matrix plot logged as an artifact.")

if __name__ == "__main__":
    run_experiment()