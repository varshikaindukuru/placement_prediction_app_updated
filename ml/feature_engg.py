import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier


def normalization(df):
    df = df.copy()

    target = "PlacementStatus"
    if target not in df.columns:
        raise ValueError(f"Missing required column: {target}")

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[target]
    )

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)

    return (
        pd.DataFrame(train_scaled, columns=train_df.columns, index=train_df.index),
        pd.DataFrame(test_scaled, columns=test_df.columns, index=test_df.index)
    )


def data_cleaning(df):
    df = df.copy()

    for col in ["Workshops", "AptitudeTestScore"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    numerical_cols = [
        col for col in [
            "CGPA",
            "AttendancePercent",
            "AptitudeTestScore",
            "CodingTestScore",
            "Internships"
        ]
        if col in df.columns
    ]

    if numerical_cols:
        imputer = KNNImputer(n_neighbors=3)
        df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

    return df


def simple_linear_regression(df):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    x = df[["SGPA_Sem1"]]
    y = df["PlacementStatus"]

    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(train_x, train_y)
    y_pred = model.predict(test_x)

    return {
        "mse": mean_squared_error(test_y, y_pred),
        "r2": r2_score(test_y, y_pred)
    }


def gradient_descent(x, y, learning_rate=0.01, iterations=100):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    m = len(y)
    theta0 = 0.0
    theta1 = 0.0
    mse_history = []

    for _ in range(iterations):
        y_pred = theta0 + theta1 * x
        error = y_pred - y

        mse_history.append(np.mean(error ** 2))

        theta0 -= learning_rate * (2 / m) * np.sum(error)
        theta1 -= learning_rate * (2 / m) * np.sum(error * x)

    return theta0, theta1, mse_history


def simple_linear_regression_GD(df):
    from sklearn.metrics import mean_squared_error, r2_score

    x = df[["SGPA_Sem3"]]
    y = df["PlacementStatus"]

    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    theta0, theta1, mse_history = gradient_descent(
        train_x, train_y, learning_rate=0.01, iterations=100
    )

    y_pred = theta0 + theta1 * test_x.to_numpy().ravel()

    return {
        "mse": mean_squared_error(test_y, y_pred),
        "r2": r2_score(test_y, y_pred),
        "mse_history": mse_history
    }


def simple_logistic_regression(df):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    x = df[["CGPA"]]
    y = df["PlacementStatus"]

    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(train_x, train_y)

    return {
        "accuracy": accuracy_score(test_y, model.predict(test_x)),
        "intercept": model.intercept_.tolist()
    }


def decision_tree(df):
    from sklearn.metrics import accuracy_score

    x = df[["CGPA", "Internships", "AptitudeTestScore"]]
    y = df["PlacementStatus"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeClassifier(random_state=42)
    model.fit(x_train, y_train)

    return {
        "model": model,
        "accuracy": accuracy_score(y_test, model.predict(x_test))
    }


def decision_tree_pruned(df):
    from sklearn.metrics import accuracy_score

    x = df[["CGPA", "Internships", "AptitudeTestScore"]]
    y = df["PlacementStatus"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    model.fit(x_train, y_train)

    return {
        "model": model,
        "accuracy": accuracy_score(y_test, model.predict(x_test))
    }


def random_forest_model(df):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    x = df[["CGPA", "Internships", "AptitudeTestScore"]]
    y = df["PlacementStatus"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42
    )
    model.fit(x_train, y_train)

    return {
        "model": model,
        "accuracy": accuracy_score(y_test, model.predict(x_test))
    }


def compare(sample_df, m1, m2, m3):
    x = sample_df[["CGPA", "Internships", "AptitudeTestScore"]]

    return pd.DataFrame({
        "Student": sample_df.index,
        "Actual": sample_df["PlacementStatus"].values,
        "Decision Tree": m1.predict(x),
        "Pruned Decision Tree": m2.predict(x),
        "Random Forest": m3.predict(x)
    })