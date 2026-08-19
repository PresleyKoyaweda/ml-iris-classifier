import joblib


def test_train_produces_model_and_metadata_files(trained_model):
    assert trained_model["model_path"].exists()
    assert trained_model["metadata_path"].exists()


def test_metadata_has_expected_shape(trained_model):
    metadata = trained_model["metadata"]
    assert metadata["model_name"] == "iris_classifier"
    assert metadata["model_type"] == "RandomForestClassifier"
    assert metadata["features"] == [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]
    assert metadata["target_classes"] == ["setosa", "versicolor", "virginica"]
    assert set(metadata["metrics"]) == {"accuracy", "precision", "recall", "f1"}


def test_model_meets_accuracy_threshold(trained_model):
    metrics = trained_model["metadata"]["metrics"]
    assert metrics["accuracy"] >= 0.9
    assert metrics["f1"] >= 0.9


def test_model_predicts_valid_class(trained_model):
    model = joblib.load(trained_model["model_path"])
    prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])
    assert prediction[0] in (0, 1, 2)
