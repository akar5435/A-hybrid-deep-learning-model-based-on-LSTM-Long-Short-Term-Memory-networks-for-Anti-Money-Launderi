import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, mean_squared_error
from imblearn.over_sampling import SMOTE
from scikeras.wrappers import KerasClassifier

# Load the dataset
data_path = 'C:/python/find_LSTM_2024_1.0/HI-Small_Trans.csv'
df = pd.read_csv(data_path)

# Dataset information
print("Dataset Shape:", df.shape)
print("Columns:", df.columns)
print(df.head())
print(df.info())

# Take a sample of 50000 rows for analysis
sampled_df = df.sample(n=50000, random_state=42)

# Check target variable distribution in the sampled DataFrame
print("\nValue Counts for 'Is Laundering' in Sampled DataFrame:")
print(sampled_df['Is Laundering'].value_counts())

# Plot target variable distribution
sns.countplot(data=sampled_df, x='Is Laundering')
plt.title("Distribution of 'Is Laundering' in Sampled DataFrame")
plt.show()

# Drop columns that are unlikely to be useful for the model
cols_to_drop = ['Timestamp', 'From Bank', 'To Bank', 'Account', 'Account.1']
# only columns that exist in the DataFrame
cols_to_drop = [col for col in cols_to_drop if col in sampled_df.columns]
sampled_df.drop(columns=cols_to_drop, axis=1, inplace=True)

# Handle missing values check
print("Missing Values in Each Column:\n", sampled_df.isnull().sum())

# Define independent (X) and dependent (y) variables
X = sampled_df.drop(columns=["Is Laundering"], axis=1)
y = sampled_df["Is Laundering"]

# Identify numeric and categorical columns
numeric_cols = X.select_dtypes(exclude="object").columns.tolist()
categorical_cols = X.select_dtypes(include="object").columns.tolist()
print("Numeric Columns:", numeric_cols)
print("Categorical Columns:", categorical_cols)

num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())
])

cat_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ordinalencoder", OrdinalEncoder())
])

# Column transformer
transformer = ColumnTransformer(transformers=[
    ("ordinal", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols),
    ("scaler", RobustScaler(), numeric_cols)
], remainder="passthrough")

# Split data into training, validation, and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# Transform training and test data
X_train_transformed = transformer.fit_transform(X_train)
X_valid_transformed = transformer.transform(X_valid)
X_test_transformed = transformer.transform(X_test)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train)

#class distribution post SMOTE
sns.countplot(x=y_train_resampled)
plt.title("Class Distribution After SMOTE")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks([0, 1], ['0', '1'])
plt.show()

# Reshape data for CNN-LSTM model
X_train_resampled = X_train_resampled.reshape((X_train_resampled.shape[0], X_train_resampled.shape[1], 1))
X_valid_transformed = X_valid_transformed.reshape((X_valid_transformed.shape[0], X_valid_transformed.shape[1], 1))
X_test_transformed = X_test_transformed.reshape((X_test_transformed.shape[0], X_test_transformed.shape[1], 1))

# One-hot encode the target variable
y_train_resampled = to_categorical(y_train_resampled)
y_valid_categorical = to_categorical(y_valid)
y_test_categorical = to_categorical(y_test)

# Define a function to generate the model for grid search
def generate_model(filters=64, dropout_rate=0.5, l2_strength=0.01):
    model = Sequential([
        Conv1D(filters=filters, kernel_size=3, activation='relu', kernel_regularizer=l2(l2_strength), input_shape=(X_train_resampled.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Dropout(dropout_rate),
        LSTM(64, activation='tanh', return_sequences=True, kernel_regularizer=l2(l2_strength)),
        Dropout(dropout_rate),
        LSTM(32, activation='tanh', kernel_regularizer=l2(l2_strength)),
        Dropout(dropout_rate),
        Dense(16, activation='relu', kernel_regularizer=l2(l2_strength)),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Wrap the model with KerasClassifier for scikit-learn compatibility
model = KerasClassifier(model=generate_model, verbose=0)

# Define the parameter grid for grid search (without batch_size and epochs)
param_grid = {
    'model__filters': [32, 64],       # Tune number of filters in Conv1D
    'model__dropout_rate': [0.2, 0.3, 0.5]  # Tune dropout rate
}

# Set up GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)

# Run grid search
grid_result = grid.fit(X_train_resampled, y_train_resampled)

# Display the best parameters and accuracy
print("Best Parameters:", grid_result.best_params_)
print("Best Accuracy:", grid_result.best_score_)


best_model = grid_result.best_estimator_.model_

history = best_model.fit(
    X_train_resampled,
    y_train_resampled,
    validation_data=(X_valid_transformed, y_valid_categorical),
    epochs=10,
    batch_size=32,
    verbose=1
)

def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_history(history)

# Evaluate the best model
best_model = grid_result.best_estimator_
val_loss, val_accuracy = best_model.model_.evaluate(X_valid_transformed, y_valid_categorical)
print("Loss:", val_loss)
print("Accuracy:", val_accuracy)

# ROC AUC Curve and Metric (MSE) Function
def plot_roc_auc_and_errors(model, X_test, y_test_categorical):
    y_test_probs = model.predict(X_test)

    y_test_true_classes = np.argmax(y_test_categorical, axis=1)
    y_test_pred_probs = y_test_probs[:, 1]

    roc_auc = roc_auc_score(y_test_true_classes, y_test_pred_probs)
    print("ROC AUC Score:", roc_auc)

    fpr, tpr, _ = roc_curve(y_test_true_classes, y_test_pred_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color="green")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()

    y_test_pred_classes = np.argmax(y_test_probs, axis=1)
    mse = mean_squared_error(y_test_true_classes, y_test_pred_classes)
    print("Mean Squared Error (MSE):", mse)

plot_roc_auc_and_errors(best_model.model_, X_test_transformed, y_test_categorical)

# Evaluation function
def evaluate_model_cnn_lstm(model, X_test, y_test):
    y_test_pred = model.predict(X_test)
    y_test_pred_classes = np.argmax(y_test_pred, axis=1)
    y_test_true_classes = np.argmax(y_test, axis=1)

    test_accuracy = round(accuracy_score(y_test_true_classes, y_test_pred_classes) * 100, 2)
    test_precision = round(precision_score(y_test_true_classes, y_test_pred_classes, average="weighted") * 100, 2)
    test_recall = round(recall_score(y_test_true_classes, y_test_pred_classes, average="weighted") * 100, 2)
    test_f1 = round(f1_score(y_test_true_classes, y_test_pred_classes, average="weighted") * 100, 3)

    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score'],
        'Score': [test_accuracy, test_precision, test_recall, test_f1]
    })
    print("CNN-LSTM Model Evaluation Metrics:\n")
    print(metrics_df)

    print("\nClassification Report:\n")
    print(classification_report(y_test_true_classes, y_test_pred_classes))

    conf_matrix = confusion_matrix(y_test_true_classes, y_test_pred_classes)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("CNN-LSTM Model Confusion Matrix")
    plt.show()

    return {
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1-score': test_f1,
        'confusion_matrix': conf_matrix
    }

# Evaluate the best CNN-LSTM model
results = evaluate_model_cnn_lstm(best_model.model_, X_test_transformed, y_test_categorical)