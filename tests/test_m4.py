"""Tests for Module 4: Evaluation."""
import sys, os
from math import isnan
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, EvalResult

def test_load_test_set():
    ts = load_test_set()
    assert len(ts) > 0 and "question" in ts[0] and "ground_truth" in ts[0]

def test_evaluate_returns_metrics():
    r = evaluate_ragas(["q"], ["a"], [["c"]], ["gt"])
    for k in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        assert k in r and isinstance(r[k], (int, float))

def test_failure_analysis_returns():
    results = [EvalResult("Q1", "A1", ["C1"], "GT1", 0.5, 0.6, 0.4, 0.3)]
    f = failure_analysis(results, bottom_n=1)
    assert len(f) == 1

def test_failure_has_diagnosis():
    results = [EvalResult("Q1", "A1", ["C1"], "GT1", 0.5, 0.6, 0.4, 0.3)]
    f = failure_analysis(results, bottom_n=1)
    if f:
        assert "diagnosis" in f[0] and "suggested_fix" in f[0]

def test_failure_preserves_question():
    results = [EvalResult("Q1", "A1", ["C1"], "GT1", 0.5, 0.6, 0.4, 0.3)]
    f = failure_analysis(results, bottom_n=1)
    if f:
        assert f[0]["question"] == "Q1"

def test_evaluate_sanitizes_nan(monkeypatch):
    import src.m4_eval as m4_eval

    class FakeResult:
        def to_pandas(self):
            class FakeFrame:
                def iterrows(self):
                    yield 0, {
                        "faithfulness": float("nan"),
                        "answer_relevancy": float("nan"),
                        "context_precision": 0.5,
                        "context_recall": 1.0,
                    }

                def __getitem__(self, key):
                    class FakeSeries:
                        def mean(self_nonlocal):
                            if key in {"faithfulness", "answer_relevancy"}:
                                return float("nan")
                            if key == "context_precision":
                                return 0.5
                            return 1.0
                    return FakeSeries()
            return FakeFrame()

    def fake_evaluate(*args, **kwargs):
        return FakeResult()

    monkeypatch.setitem(sys.modules, "ragas", type("FakeRagasModule", (), {"evaluate": fake_evaluate}))
    monkeypatch.setitem(
        sys.modules,
        "ragas.metrics",
        type(
            "FakeMetricsModule",
            (),
            {
                "faithfulness": object(),
                "answer_relevancy": object(),
                "context_precision": object(),
                "context_recall": object(),
            },
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        type("FakeDatasetsModule", (), {"Dataset": type("FakeDataset", (), {"from_dict": staticmethod(lambda data: data)})}),
    )

    result = m4_eval.evaluate_ragas(["q1"], ["a1"], [["c1"]], ["gt1"])

    for key in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        assert isinstance(result[key], (int, float))
        assert not isnan(result[key])
