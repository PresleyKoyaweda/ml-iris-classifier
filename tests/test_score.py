import json

import pytest
import score


@pytest.fixture(autouse=True)
def _load_model(scoring_model_copy):
    score.init()


def test_run_accepts_json_string_payload():
    payload = json.dumps({"data": [[5.1, 3.5, 1.4, 0.2]]})
    result = score.run(payload)
    assert "prediction" in result
    assert result["prediction"] in (0, 1, 2)
    assert len(result["probabilities"]) == 3


def test_run_accepts_raw_list_payload():
    result = score.run([[6.7, 3.1, 4.7, 1.5]])
    assert "prediction" in result
    assert "error" not in result


def test_run_invalid_payload_returns_error_not_exception():
    result = score.run("not valid json {")
    assert "error" in result
