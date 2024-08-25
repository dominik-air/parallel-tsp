import time


class StopCondition:
    """Defines the stopping conditions for a genetic algorithm run.

    The StopCondition class allows a genetic algorithm to terminate based on one or more criteria:
    a maximum number of generations, a minimum percentage improvement in the best route length,
    or a maximum amount of time elapsed.

    Attributes:
        max_generations (int | None): The maximum number of generations to run the algorithm.
        improvement_percentage (float | None): The percentage improvement required to stop the algorithm.
        max_time_seconds (int | None): The maximum time allowed for the algorithm to run, in seconds.
        start_time (float | None): The start time of the algorithm, used for time-based stopping.
        initial_best_length (float | None): The length of the best route at the start of the algorithm,
            used for improvement-based stopping.
        triggered_condition (str | None): The condition that triggered the stopping of the algorithm.
    """

    def __init__(
        self,
        max_generations: int | None = None,
        improvement_percentage: float | None = None,
        max_time_seconds: int | None = None,
    ):
        """Initializes the StopCondition with the given parameters.

        Args:
            max_generations (int | None): The maximum number of generations to run the algorithm.
            improvement_percentage (float | None): The percentage improvement required to stop the algorithm.
            max_time_seconds (int | None): The maximum time allowed for the algorithm to run, in seconds.
        """
        self.max_generations = max_generations
        self.improvement_percentage = improvement_percentage
        self.max_time_seconds = max_time_seconds
        self.start_time = time.perf_counter() if max_time_seconds is not None else None
        self.initial_best_length = None
        self.triggered_condition = None

    def update_initial_best_length(self, length: float) -> None:
        """Updates the initial best route length used for improvement-based stopping.

        Args:
            length (float): The length of the best route at the start of the algorithm.
        """
        if self.initial_best_length is None:
            self.initial_best_length = length

    def should_stop(self, generations_run: int, current_best_length: float) -> bool:
        """Determines whether the genetic algorithm should stop.

        This method checks the current state of the algorithm against the defined stop conditions.
        It will return True if any of the conditions are met.

        Args:
            generations_run (int): The number of generations the algorithm has run so far.
            current_best_length (float): The length of the current best route.

        Returns:
            bool: True if the algorithm should stop, False otherwise.
        """
        if self.max_generations is not None and generations_run >= self.max_generations:
            self.triggered_condition = "max_generations"
            return True

        if self.improvement_percentage is not None:
            if self.initial_best_length is None:
                raise ValueError(
                    "Initial best length not set for percentage improvement condition."
                )
            improvement = (
                (self.initial_best_length - current_best_length)
                / self.initial_best_length
                * 100
            )
            if improvement >= self.improvement_percentage:
                self.triggered_condition = "improvement_percentage"
                return True

        if self.max_time_seconds is not None:
            elapsed_time = time.perf_counter() - self.start_time
            if elapsed_time >= self.max_time_seconds:
                self.triggered_condition = "max_time_seconds"
                return True

        return False

    def get_triggered_condition(self) -> str:
        """Returns the condition that triggered the stopping of the algorithm.

        Returns:
            str: The condition that triggered the stopping, such as 'max_generations',
            'improvement_percentage', or 'max_time_seconds'.
        """
        return self.triggered_condition
