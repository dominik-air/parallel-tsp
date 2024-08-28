import logging
import time
from enum import Enum

from mpi4py import MPI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class StopConditionType(str, Enum):
    """Enum for different types of stopping conditions."""

    MAX_GENERATIONS = "max_generations"
    IMPROVEMENT_PERCENTAGE = "improvement_percentage"
    MAX_TIME_SECONDS = "max_time_seconds"


class StopCondition:
    """Defines the stopping conditions for a genetic algorithm run.

    The `StopCondition` class allows a genetic algorithm to terminate based on one or more criteria:
    - A maximum number of generations.
    - A minimum percentage improvement in the best route length.
    - A maximum amount of time elapsed.

    The class supports both MPI-based timing and standard Python timing for time-based stopping conditions.
    """

    def __init__(
        self,
        max_generations: int | None = None,
        improvement_percentage: float | None = None,
        max_time_seconds: int | None = None,
        is_mpi: bool = False,
    ):
        """Initializes the StopCondition with the given parameters.

        Args:
            max_generations (int | None): The maximum number of generations to run the algorithm.
            improvement_percentage (float | None): The percentage improvement required to stop the algorithm.
            max_time_seconds (int | None): The maximum time allowed for the algorithm to run, in seconds.
            is_mpi (bool): Whether to use MPI for timing. If True, MPI's `Wtime` is used for time tracking.
                           If False, Python's `time.perf_counter` is used.
        """
        self.max_generations = max_generations
        self.improvement_percentage = improvement_percentage
        self.max_time_seconds = max_time_seconds
        self.is_mpi = is_mpi

        if self.is_mpi:
            self.timer = MPI.Wtime
        else:
            self.timer = time.perf_counter

        self.start_time = None
        self.initial_best_length = None
        self.triggered_condition = None

        logger.info(
            f"StopCondition initialized with: max_generations={self.max_generations}, "
            f"improvement_percentage={self.improvement_percentage}, max_time_seconds={self.max_time_seconds}, "
            f"is_mpi={self.is_mpi}"
        )

    def start_timer(self):
        """Starts the timer for time-based stopping condition.

        This method should be called before running the algorithm if a time-based stop condition is used.
        """
        self.start_time = self.timer()
        logger.info("Timer started for StopCondition.")

    def update_initial_best_length(self, length: float) -> None:
        """Updates the initial best route length used for improvement-based stopping.

        Args:
            length (float): The length of the best route at the start of the algorithm.
        """
        if self.initial_best_length is None:
            self.initial_best_length = length
            logger.info(f"Initial best route length set to: {length:.2f}")

    def should_stop(self, generations_run: int, current_best_length: float) -> bool:
        """Determines whether the genetic algorithm should stop based on the defined stop conditions.

        This method checks the current state of the algorithm against the stop conditions specified during initialization.
        It will return True if any of the conditions are met.

        Args:
            generations_run (int): The number of generations the algorithm has run so far.
            current_best_length (float): The length of the current best route.

        Returns:
            bool: True if the algorithm should stop, False otherwise.

        Raises:
            ValueError: If the timer was not started before running the algorithm and a time-based stop condition is used.
        """
        if self.max_generations is not None and generations_run >= self.max_generations:
            self.triggered_condition = StopConditionType.MAX_GENERATIONS
            logger.info(
                f"StopCondition triggered: {self.triggered_condition} after {generations_run} generations."
            )
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
                self.triggered_condition = StopConditionType.IMPROVEMENT_PERCENTAGE
                logger.info(
                    f"StopCondition triggered: {self.triggered_condition} with improvement of {improvement:.2f}%."
                )
                return True

        if self.max_time_seconds is not None:
            if self.start_time is None:
                raise ValueError(
                    "Timer was not started. Call 'start_timer()' before running the algorithm."
                )

            elapsed_time = self.timer() - self.start_time
            if elapsed_time >= self.max_time_seconds:
                self.triggered_condition = StopConditionType.MAX_TIME_SECONDS
                logger.info(
                    f"StopCondition triggered: {self.triggered_condition} after {elapsed_time:.2f} seconds."
                )
                return True

        return False

    def get_triggered_condition(self) -> StopConditionType | None:
        """Returns the condition that triggered the stopping of the algorithm.

        Returns:
            StopConditionType | None: The condition that triggered the stopping, such as 'MAX_GENERATIONS',
            'IMPROVEMENT_PERCENTAGE', or 'MAX_TIME_SECONDS'. Returns None if no condition has been triggered.
        """
        return self.triggered_condition
