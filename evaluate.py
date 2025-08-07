import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve
)
import xgboost as xgb
import yaml
from src.model_runner import generate_synthetic_data  # Make sure this path is correct


def evaluate_with_synthetic_data(config):

    # Load original dataset
    original_df = pd.read_csv(config["data_path"])

    # Generate synthetic fraud data
    synthetic_df = generate_synthetic_data(
        config_path=config["vae_config_path"],
        model_path=config["vae_model_path"],
        n_samples=config["n_samples"]
    )

    # Combine real and synthetic data
    combined_df = pd.concat([original_df, synthetic_df], ignore_index=True)
    print(f" Combined dataset shape: {combined_df.shape}")

    # Split features and labels
    X = combined_df.drop(columns=[config["label_column"]])
    y = combined_df[config["label_column"]]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["test_size"],
        stratify=y,
        random_state=config["random_state"]
    )

    # Initialize classifiers
    rf_model = RandomForestClassifier(
        n_estimators=config["rf_n_estimators"],
        random_state=config["random_state"],
        class_weight=config["class_weight"]
    )

    lr_model = LogisticRegression(
        solver=config["lr_solver"],
        class_weight=config["class_weight"],
        max_iter=config["lr_max_iter"]
    )

    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        n_estimators=config["xgb_n_estimators"],
        learning_rate=config["xgb_learning_rate"],
        random_state=config["random_state"],
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()
    )

    models = [
        ("Random Forest", rf_model),
        ("Logistic Regression", lr_model),
        ("XGBoost", xgb_model)
    ]

    for name, model in models:
        print(f"\n Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Display classification report
        print(f"\n Classification Report for {name}:")
        print(classification_report(y_test, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.title(f"{name} - Confusion Matrix")
        plt.grid(False)
        plt.show()

        # ROC Curve
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.title(f"{name} - ROC Curve")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.show()

            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            plt.plot(recall, precision, label=name)
            plt.title(f"{name} - Precision-Recall Curve")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.grid(True)
            plt.legend()
            plt.show()


if __name__ == "__main__":
    with open("config/eval_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    evaluate_with_synthetic_data(config)
