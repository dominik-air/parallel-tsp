import time

import pytest

from parallel_tsp.stop_condition import StopCondition


def test_stop_condition_max_generations():
    stop_condition = StopCondition(max_generations=10)
    stop = stop_condition.should_stop(generations_run=10, current_best_length=50)
    assert stop is True
    assert stop_condition.get_triggered_condition() == "max_generations"

    stop = stop_condition.should_stop(generations_run=5, current_best_length=50)
    assert stop is False


def test_stop_condition_improvement_percentage():
    stop_condition = StopCondition(improvement_percentage=10.0)
    stop_condition.update_initial_best_length(100.0)

    stop = stop_condition.should_stop(generations_run=5, current_best_length=95.0)
    assert stop is False

    stop = stop_condition.should_stop(generations_run=5, current_best_length=89.0)
    assert stop is True
    assert stop_condition.get_triggered_condition() == "improvement_percentage"


def test_stop_condition_max_time_seconds(monkeypatch):
    stop_condition = StopCondition(max_time_seconds=2)

    def mock_perf_counter():
        return stop_condition.start_time + 3

    monkeypatch.setattr(time, "perf_counter", mock_perf_counter)

    stop = stop_condition.should_stop(generations_run=5, current_best_length=50)
    assert stop is True
    assert stop_condition.get_triggered_condition() == "max_time_seconds"


def test_stop_condition_combined():
    stop_condition = StopCondition(
        max_generations=10, improvement_percentage=20.0, max_time_seconds=2
    )
    stop_condition.update_initial_best_length(100.0)

    stop = stop_condition.should_stop(generations_run=10, current_best_length=95.0)
    assert stop is True
    assert stop_condition.get_triggered_condition() == "max_generations"

    stop_condition = StopCondition(max_generations=10, improvement_percentage=20.0)
    stop_condition.update_initial_best_length(100.0)
    stop = stop_condition.should_stop(generations_run=5, current_best_length=79.0)
    assert stop is True
    assert stop_condition.get_triggered_condition() == "improvement_percentage"


def test_stop_condition_no_conditions_met():
    stop_condition = StopCondition(
        max_generations=10, improvement_percentage=10.0, max_time_seconds=5
    )
    stop_condition.update_initial_best_length(100.0)

    stop = stop_condition.should_stop(generations_run=5, current_best_length=95.0)
    assert stop is False
    assert stop_condition.get_triggered_condition() is None


def test_update_initial_best_length():
    stop_condition = StopCondition(improvement_percentage=10.0)
    stop_condition.update_initial_best_length(100.0)

    assert stop_condition.initial_best_length == 100.0

    stop_condition.update_initial_best_length(90.0)
    assert stop_condition.initial_best_length == 100.0


def test_error_without_initial_best_length():
    stop_condition = StopCondition(improvement_percentage=10.0)

    with pytest.raises(ValueError):
        stop_condition.should_stop(generations_run=5, current_best_length=90.0)


if __name__ == "__main__":
    pytest.main()
