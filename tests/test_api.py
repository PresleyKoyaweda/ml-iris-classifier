import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(trained_model):
    import main

    return TestClient(main.app)


def test_health_reports_model_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True


def test_info_returns_training_metadata(client):
    r = client.get("/info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_name"] == "iris_classifier"
    assert body["target_classes"] == ["setosa", "versicolor", "virginica"]


def test_predict_returns_class_and_probabilities(client):
    r = client.post(
        "/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in ("setosa", "versicolor", "virginica")
    assert abs(sum(body["probability"].values()) - 1.0) < 1e-6


def test_predict_missing_field_returns_422(client):
    r = client.post("/predict", json={"sepal_length": 5.1})
    assert r.status_code == 422


def test_predict_batch_returns_one_result_per_input(client):
    r = client.post(
        "/predict-batch",
        json=[
            {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
            {"sepal_length": 6.7, "sepal_width": 3.1, "petal_length": 4.7, "petal_width": 1.5},
        ],
    )
    assert r.status_code == 200
    assert len(r.json()) == 2
