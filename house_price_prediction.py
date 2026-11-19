
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor


# File locations
project_folder = Path(__file__).resolve().parent
data_file = project_folder / "AmesHousing.csv"
results_folder = project_folder / "results"
results_folder.mkdir(exist_ok=True)

if not data_file.exists():
    raise FileNotFoundError(
        "AmesHousing.csv was not found. "
        "Put the dataset in the same folder as this Python file."
    )


# Load the housing data
df = pd.read_csv(data_file)

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\nColumn information:")
df.info()

print("\nColumns with the most missing values:")
print(df.isnull().sum().sort_values(ascending=False).head(15))

print("\nSale price statistics:")
print(df["SalePrice"].describe())


# Check how sale prices are distributed
plt.figure(figsize=(9, 5))
plt.hist(df["SalePrice"], bins=40, edgecolor="black")
plt.xlabel("Sale Price ($)")
plt.ylabel("Number of Houses")
plt.title("Distribution of House Sale Prices")
plt.tight_layout()
plt.savefig(results_folder / "sale_price_distribution.png", dpi=300)
plt.close()


# Separate the house details from the prices
X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]

print("\nIdentifier columns:")
print(df[["Order", "PID"]].head(10))
print(df[["Order", "PID"]].nunique())

# Order and PID identify records rather than describing the houses
X = X.drop(columns=["Order", "PID"])

print("\nFeatures:", X.shape)
print("Target:", y.shape)


# Keep 15% of the data aside for the final test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.15,
    random_state=42
)

# Split the rest into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.15 / 0.85,
    random_state=42
)

X_train = X_train.copy()
X_val = X_val.copy()
X_test = X_test.copy()

print("\nData split:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)


# MS SubClass is a category even though it is stored as a number
for data in [X_train, X_val, X_test]:
    data["MS SubClass"] = data["MS SubClass"].astype(str)

numerical_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nNumerical features:", len(numerical_features))
print("Categorical features:", len(categorical_features))


# Look at missing values before choosing how to handle them
missing_train = X_train.isnull().sum()

print("\nMissing values in training data:")
print(
    missing_train[missing_train > 0]
    .sort_values(ascending=False)
    .head(20)
)

# Some missing values appear to mean a feature is absent
print("\nPool checks:")
print("Missing Pool QC:", X_train["Pool QC"].isna().sum())
print("Zero Pool Area:", (X_train["Pool Area"] == 0).sum())
print(
    "Missing quality and zero area:",
    (
        X_train["Pool QC"].isna()
        & (X_train["Pool Area"] == 0)
    ).sum()
)

print("\nGarage checks:")
print("Missing Garage Qual:", X_train["Garage Qual"].isna().sum())
print(
    "Missing quality and zero area:",
    (
        X_train["Garage Qual"].isna()
        & (X_train["Garage Area"] == 0)
    ).sum()
)

print("\nFireplace checks:")
print("Missing Fireplace Qu:", X_train["Fireplace Qu"].isna().sum())
print(
    "Missing quality and no fireplaces:",
    (
        X_train["Fireplace Qu"].isna()
        & (X_train["Fireplaces"] == 0)
    ).sum()
)

garage_exceptions = X_train[
    X_train["Garage Qual"].isna()
    & (X_train["Garage Area"] != 0)
]

print("\nGarage records with missing quality and non-zero/unknown area:")
print(
    garage_exceptions[
        [
            "Garage Type", "Garage Finish", "Garage Area",
            "Garage Cars", "Garage Yr Blt",
            "Garage Qual", "Garage Cond"
        ]
    ]
)

garage_missing = X_train[X_train["Garage Qual"].isna()]

print("\nGarage quality missing:", len(garage_missing))
print("Zero area:", (garage_missing["Garage Area"] == 0).sum())
print("Positive area:", (garage_missing["Garage Area"] > 0).sum())
print("Unknown area:", garage_missing["Garage Area"].isna().sum())


# Fill numerical gaps with the median
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

# Give missing categories their own label and encode them
categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="constant", fill_value="Missing")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Fit on training data only to avoid validation leakage
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)

print("\nProcessed training shape:", X_train_processed.shape)
print("Processed validation shape:", X_val_processed.shape)


# Use the same metrics for each model
def evaluate_model(y_true, predictions):
    mae = mean_absolute_error(y_true, predictions)
    rmse = np.sqrt(mean_squared_error(y_true, predictions))
    r2 = r2_score(y_true, predictions)
    return mae, rmse, r2


def show_results(name, y_true, predictions):
    mae, rmse, r2 = evaluate_model(y_true, predictions)

    print(f"\n{name}")
    print(f"MAE: ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R2: {r2:.4f}")

    return mae, rmse, r2


def make_xgb(depth=5, reg_lambda=1):
    return XGBRegressor(
        n_estimators=100,
        max_depth=depth,
        learning_rate=0.1,
        reg_lambda=reg_lambda,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )


# Start with a simple model that always predicts the mean price
dummy_model = DummyRegressor(strategy="mean")
dummy_model.fit(X_train_processed, y_train)

dummy_predictions = dummy_model.predict(X_val_processed)
show_results("Dummy Regressor - Validation", y_val, dummy_predictions)


# Compare against Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train_processed, y_train)

linear_predictions = linear_model.predict(X_val_processed)
show_results("Linear Regression - Validation", y_val, linear_predictions)


# Initial XGBoost model
xgb_model = make_xgb(depth=3, reg_lambda=1)
xgb_model.fit(X_train_processed, y_train)

xgb_predictions = xgb_model.predict(X_val_processed)
show_results("XGBoost - Validation", y_val, xgb_predictions)

train_predictions = xgb_model.predict(X_train_processed)

train_mae = mean_absolute_error(y_train, train_predictions)
val_mae = mean_absolute_error(y_val, xgb_predictions)

print("\nInitial XGBoost training MAE:", round(train_mae, 2))
print("Initial XGBoost validation MAE:", round(val_mae, 2))
print("MAE gap:", round(val_mae - train_mae, 2))


# See how tree depth affects training and validation error
depth_results = []

for depth in [2, 3, 5, 7, 10]:
    model = make_xgb(depth=depth, reg_lambda=1)
    model.fit(X_train_processed, y_train)

    train_pred = model.predict(X_train_processed)
    val_pred = model.predict(X_val_processed)

    train_mae = mean_absolute_error(y_train, train_pred)
    val_mae = mean_absolute_error(y_val, val_pred)

    depth_results.append({
        "Max Depth": depth,
        "Training MAE": train_mae,
        "Validation MAE": val_mae,
        "MAE Gap": val_mae - train_mae
    })

depth_results_df = pd.DataFrame(depth_results)

print("\nTree-depth results:")
print(depth_results_df.round(2).to_string(index=False))

depth_results_df.to_csv(
    results_folder / "COMP1818_depth_results.csv",
    index=False
)


# Plot the depth results for the report
plt.figure(figsize=(7, 4.5))

plt.plot(
    depth_results_df["Max Depth"],
    depth_results_df["Training MAE"],
    marker="o",
    linewidth=2,
    label="Training MAE"
)

plt.plot(
    depth_results_df["Max Depth"],
    depth_results_df["Validation MAE"],
    marker="s",
    linewidth=2,
    label="Validation MAE"
)

plt.xlabel("Maximum Tree Depth")
plt.ylabel("Mean Absolute Error ($)")
plt.title("Effect of Tree Depth on XGBoost Performance")
plt.xticks([2, 3, 5, 7, 10])
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    results_folder / "xgboost_depth.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# Test different L2 strengths at depth 7
regularisation_results = []

for strength in [0, 1, 10, 50]:
    model = make_xgb(depth=7, reg_lambda=strength)
    model.fit(X_train_processed, y_train)

    train_pred = model.predict(X_train_processed)
    val_pred = model.predict(X_val_processed)

    train_mae = mean_absolute_error(y_train, train_pred)
    val_mae = mean_absolute_error(y_val, val_pred)

    regularisation_results.append({
        "L2 Strength": strength,
        "Training MAE": train_mae,
        "Validation MAE": val_mae,
        "MAE Gap": val_mae - train_mae
    })

regularisation_df = pd.DataFrame(regularisation_results)

print("\nL2 regularisation results:")
print(regularisation_df.round(2).to_string(index=False))

regularisation_df.to_csv(
    results_folder / "COMP1818_l2_results.csv",
    index=False
)


# Check what happens when less training data is available
size_results = []

for fraction in [0.25, 0.50, 0.75, 1.00]:

    if fraction == 1.00:
        X_subset = X_train_processed
        y_subset = y_train
    else:
        X_subset, _, y_subset, _ = train_test_split(
            X_train_processed,
            y_train,
            train_size=fraction,
            random_state=42
        )

    model = make_xgb(depth=5, reg_lambda=1)
    model.fit(X_subset, y_subset)

    predictions = model.predict(X_val_processed)

    size_results.append({
        "Training Fraction": fraction,
        "Training Houses": len(y_subset),
        "Validation MAE": mean_absolute_error(y_val, predictions)
    })

size_results_df = pd.DataFrame(size_results)

print("\nTraining-size results:")
print(size_results_df.round(2).to_string(index=False))

size_results_df.to_csv(
    results_folder / "COMP1818_training_size_results.csv",
    index=False
)


# Try increasing L2 regularisation when the training set is smaller
adaptive_results = []
full_training_size = len(y_train)

for fraction in [0.25, 0.50, 0.75, 1.00]:

    if fraction == 1.00:
        X_subset = X_train_processed
        y_subset = y_train
    else:
        X_subset, _, y_subset, _ = train_test_split(
            X_train_processed,
            y_train,
            train_size=fraction,
            random_state=42
        )

    n = len(y_subset)

    for method, strength in [
        ("Fixed", 1.0),
        ("Adaptive", full_training_size / n)
    ]:
        model = make_xgb(depth=5, reg_lambda=strength)
        model.fit(X_subset, y_subset)

        predictions = model.predict(X_val_processed)

        adaptive_results.append({
            "Training Houses": n,
            "Method": method,
            "L2 Strength": strength,
            "Validation MAE": mean_absolute_error(y_val, predictions)
        })

adaptive_results_df = pd.DataFrame(adaptive_results)

print("\nInitial fixed vs adaptive comparison:")
print(adaptive_results_df.round(2).to_string(index=False))

adaptive_results_df.to_csv(
    results_folder / "COMP1818_adaptive_initial.csv",
    index=False
)


# Repeat with different random subsets to check if the pattern holds
repeated_results = []

for seed in [11, 22, 33, 44, 55]:
    for fraction in [0.25, 0.50, 0.75, 1.00]:

        if fraction == 1.00:
            X_subset = X_train_processed
            y_subset = y_train
        else:
            X_subset, _, y_subset, _ = train_test_split(
                X_train_processed,
                y_train,
                train_size=fraction,
                random_state=seed
            )

        n = len(y_subset)

        # Both methods use the same subset for a fair comparison
        for method, strength in [
            ("Fixed", 1.0),
            ("Adaptive", len(y_train) / n)
        ]:
            model = make_xgb(depth=5, reg_lambda=strength)
            model.fit(X_subset, y_subset)

            predictions = model.predict(X_val_processed)

            repeated_results.append({
                "Seed": seed,
                "Training Houses": n,
                "Method": method,
                "L2 Strength": strength,
                "Validation MAE": mean_absolute_error(y_val, predictions)
            })

repeated_df = pd.DataFrame(repeated_results)

print("\nRepeated experiment results:")
print(repeated_df.round(2).to_string(index=False))

repeated_df.to_csv(
    results_folder / "COMP1818_repeated_results.csv",
    index=False
)


# Calculate the mean and standard deviation across the five runs
summary_df = (
    repeated_df
    .groupby(["Training Houses", "Method"])["Validation MAE"]
    .agg(["mean", "std"])
    .reset_index()
)

print("\nRegularisation summary:")
print(summary_df.round(2).to_string(index=False))

summary_df.to_csv(
    results_folder / "COMP1818_regularisation_results.csv",
    index=False
)


# Combine training and validation data for the final models
X_train_final = pd.concat([X_train, X_val], axis=0)
y_train_final = pd.concat([y_train, y_val], axis=0)

print("\nFinal training shape:", X_train_final.shape)
print("Test shape:", X_test.shape)

# Refit preprocessing without using any test-set information
final_preprocessor = clone(preprocessor)

X_train_final_processed = final_preprocessor.fit_transform(
    X_train_final
)

X_test_final_processed = final_preprocessor.transform(
    X_test
)


# Check for missing values after preprocessing
def count_missing(matrix):
    if hasattr(matrix, "toarray"):
        return int(np.isnan(matrix.data).sum())
    return int(np.isnan(matrix).sum())


print("\nMissing values in final training:",
      count_missing(X_train_final_processed))

print("Missing values in final testing:",
      count_missing(X_test_final_processed))


# Train the three models using the combined training data
final_dummy = DummyRegressor(strategy="mean")
final_dummy.fit(X_train_final_processed, y_train_final)

final_linear = LinearRegression()
final_linear.fit(X_train_final_processed, y_train_final)

final_xgb = make_xgb(depth=5, reg_lambda=1)
final_xgb.fit(X_train_final_processed, y_train_final)

print("\nAll final models trained successfully.")


# Evaluate only after the model settings have been chosen
final_models = {
    "Dummy Regressor": final_dummy,
    "Linear Regression": final_linear,
    "XGBoost": final_xgb
}

final_results = []

for name, model in final_models.items():
    predictions = model.predict(X_test_final_processed)

    mae, rmse, r2 = evaluate_model(y_test, predictions)

    final_results.append({
        "Model": name,
        "Test MAE": mae,
        "Test RMSE": rmse,
        "Test R2": r2
    })

final_results_df = pd.DataFrame(final_results)

print("\nFinal held-out test results:")
print(final_results_df.round(4).to_string(index=False))

final_results_df.to_csv(
    results_folder / "COMP1818_final_model_results.csv",
    index=False
)

print("\nFinished running all experiments.")
print("Results saved in:", results_folder)
