import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
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
    print(f"Combined dataset shape: {combined_df.shape}")

    # Split features and labels
    X = combined_df.drop(columns=[config["label_column"]])
    y = combined_df[config["label_column"]]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["test_size"],
        stratify=y,
        random_state=config["random_state"]
    )

    # Initialize classifiers using config
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

    # Train classifiers
    rf_model.fit(X_train, y_train)
    lr_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    # Evaluate
    for name, model in zip(["Random Forest", "Logistic Regression", "XGBoost"], [rf_model, lr_model, xgb_model]):
        y_pred = model.predict(X_test)
        print(f"\nClassification Report for {name}:")
        print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    with open("config/eval_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    evaluate_with_synthetic_data(config)
