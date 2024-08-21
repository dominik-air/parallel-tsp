import time

class StopCondition:
    def __init__(self, max_generations=None, improvement_percentage=None, max_time_seconds=None):
        self.max_generations = max_generations
        self.improvement_percentage = improvement_percentage
        self.max_time_seconds = max_time_seconds
        self.start_time = time.perf_counter() if max_time_seconds is not None else None
        self.initial_best_length = None
        self.triggered_condition = None

    def update_initial_best_length(self, length):
        if self.initial_best_length is None:
            self.initial_best_length = length

    def should_stop(self, generations_run, current_best_length):
        if self.max_generations is not None and generations_run >= self.max_generations:
            self.triggered_condition = 'max_generations'
            return True

        if self.improvement_percentage is not None:
            if self.initial_best_length is None:
                raise ValueError("Initial best length not set for percentage improvement condition.")
            improvement = (self.initial_best_length - current_best_length) / self.initial_best_length * 100
            if improvement >= self.improvement_percentage:
                self.triggered_condition = 'improvement_percentage'
                return True

        if self.max_time_seconds is not None:
            elapsed_time = time.perf_counter() - self.start_time
            if elapsed_time >= self.max_time_seconds:
                self.triggered_condition = 'max_time_seconds'
                return True

        return False

    def get_triggered_condition(self):
        return self.triggered_condition
