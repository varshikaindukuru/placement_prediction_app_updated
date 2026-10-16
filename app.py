
import os
from flask import Flask, render_template, request

from ml.data_loader import load_data
from ml import eda as eda_module
from ml import feature_engg as fe_module
from models.predict import predict_student, get_feature_info

app = Flask(__name__)
app.secret_key = "dev-secret-key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    BASE_DIR, "data", "placement_predict_50k_Dataset1.csv"
)

_cache = {}


def get_df():
    if "df" not in _cache:
        _cache["df"] = load_data(DATA_PATH)
    return _cache["df"]


def make_summary(df):
    missing = df.isna().sum()
    return {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_total": int(missing.sum()),
        "missing_counts": missing[missing > 0].to_dict(),
        "target_distribution": (
            df["PlacementStatus"].value_counts().to_dict()
            if "PlacementStatus" in df.columns else {}
        )
    }


@app.route("/")
def index():
    return render_template(
        "index.html",
        data_available=os.path.exists(DATA_PATH),
        data_path=DATA_PATH
    )


@app.route("/load")
def load_page():
    try:
        df = get_df()
        raw_summary = make_summary(df)
        cleaned_df = df.copy()
        cleaned_summary = make_summary(cleaned_df)
        preview_cols = cleaned_df.columns.tolist()
        preview_rows = cleaned_df.head(10).to_dict(orient="records")
        error = None
    except Exception as e:
        raw_summary = None
        cleaned_summary = None
        preview_cols = []
        preview_rows = []
        error = str(e)

    return render_template(
        "data_loading.html",
        raw_summary=raw_summary,
        cleaned_summary=cleaned_summary,
        preview_cols=preview_cols,
        preview_rows=preview_rows,
        error=error
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    error = None
    values = {}

    try:
        feature_info = get_feature_info()
        if request.method == "POST":
            values = request.form.to_dict()
            result = predict_student(values)
    except Exception as e:
        error = str(e)
        feature_info = {
            "numeric_cols": [],
            "categorical_cols": [],
            "categories": {}
        }

    return render_template(
        "predict.html",
        numeric_cols=feature_info["numeric_cols"],
        categorical_cols=feature_info["categorical_cols"],
        categories=feature_info["categories"],
        values=values,
        result=result,
        error=error
    )


@app.route("/eda")
def eda_page():
    try:
        df = get_df()
        summary = eda_module.get_summary_stats(df)
        images = eda_module.generate_charts(df)
        error = None
    except Exception as e:
        summary = None
        images = None
        error = str(e)

    return render_template(
        "eda.html",
        summary=summary,
        stats=summary,
        images=images,
        error=error
    )


@app.route("/feature-engg")
def feature_engg_page():
    try:
        df = get_df()
        train_normalized, test_normalized = fe_module.normalization(df.copy())
        standardized_df = fe_module.data_cleaning(df.copy())
        _cache["standardized_df"] = standardized_df

        result = {
            "normalized_head": train_normalized.head().to_html(classes="table"),
            "standardized_head": standardized_df.head().to_html(classes="table")
        }
        error = None
    except Exception as e:
        result = None
        error = str(e)

    return render_template(
        "feature_engg.html",
        result=result,
        error=error
    )


@app.route("/linear-regression")
def linear_regression_page():
    try:
        df = _cache.get("standardized_df")
        if df is None:
            df = fe_module.data_cleaning(get_df().copy())
            _cache["standardized_df"] = df

        gd_result = fe_module.simple_linear_regression_GD(df)
        sk_result = fe_module.simple_linear_regression(df)
        error = None
    except Exception as e:
        gd_result = None
        sk_result = None
        error = str(e)

    return render_template(
        "linear_regression.html",
        gd_result=gd_result,
        sk_result=sk_result,
        error=error
    )


@app.route("/logistic-regression")
def logistic_regression_page():
    try:
        df = _cache.get("standardized_df")
        if df is None:
            df = fe_module.data_cleaning(get_df().copy())
            _cache["standardized_df"] = df

        result = fe_module.simple_logistic_regression(df)
        error = None
    except Exception as e:
        result = None
        error = str(e)

    return render_template(
        "logistic_regression.html",
        result=result,
        error=error
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
