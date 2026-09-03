import importlib

import pytest


response = importlib.import_module("scripts.07_response_simulator")


def make_alert(risk, src_ip):
    return {
        "src_ip": src_ip,
        "dst_ip": "10.10.1.10",
        "predicted_threat": "BruteForce",
        "risk_score": risk,
    }


@pytest.mark.parametrize(
    ("risk", "action"),
    [(40, "monitor"), (75, "escalate_to_tier2"), (90, "recommend_isolate_lab_host")],
)
def test_response_is_proportional(risk, action):
    result = response.recommend_action(make_alert(risk=risk, src_ip="10.10.1.5"))

    assert result["recommended_action"] == action
    assert result["mode"] == "dry-run"
    assert result["human_approval_required"] is True


@pytest.mark.parametrize("address", ["8.8.8.8", "203.0.113.10", "not-an-ip"])
def test_non_lab_source_is_never_recommended_for_isolation(address):
    result = response.recommend_action(make_alert(risk=99, src_ip=address))

    assert result["recommended_action"] == "escalate_to_tier2"


def test_response_plan_is_limited_to_twenty_highest_risk_alerts():
    alerts = [make_alert(risk=risk, src_ip=f"10.10.1.{risk}") for risk in range(1, 31)]

    plan = response.build_response_plan(alerts)

    assert len(plan["actions"]) == 20
    assert plan["actions"][0]["risk_score"] == 30
    assert "Do not execute" in plan["safety"]
