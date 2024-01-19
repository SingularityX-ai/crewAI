import threading
import time
from typing import Union

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from crewai.utilities.logger import Logger


class RPMController(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    max_rpm: Union[int, None] = Field(default=None)
    logger: Logger = Field(default=None)
    _current_rpm: int = PrivateAttr(default=0)
    _timer: threading.Timer = PrivateAttr(default=None)
    _lock: threading.Lock = PrivateAttr(default=None)

    @model_validator(mode="after")
    def reset_counter(self):
        """
        Reset the counter and lock if max RPM is set.

        :return: None
        :raises: Any exceptions that may occur during the reset process.
        """


        if self.max_rpm:
            self._lock = threading.Lock()
            self._reset_request_count()
        return self

    def check_or_wait(self):
        """
        Check if the current RPM is less than the maximum RPM, and either increment the current RPM or wait for the next minute.

        :return: True if the current RPM is less than the maximum RPM or after waiting for the next minute.
        :raises: None
        """


        if not self.max_rpm:
            return True

        with self._lock:
            if self._current_rpm < self.max_rpm:
                self._current_rpm += 1
                return True
            else:
                self.logger.log(
                    "info", "Max RPM reached, waiting for next minute to start."
                )
                self._wait_for_next_minute()
                self._current_rpm = 1
                return True

    def stop_rpm_counter(self):
        """
        Stop the RPM counter.

        This method cancels the timer used for the RPM counter if it is running.

        Raises:
            None

        Returns:
            None
        """


        if self._timer:
            self._timer.cancel()
            self._timer = None

    def _wait_for_next_minute(self):
        """
        Wait for the next minute and reset the current RPM to 0.

        This method sleeps for 60 seconds and then resets the current RPM to 0 within a thread-safe context.

        Raises:
            Any exceptions raised by time.sleep() or by acquiring the lock.
        """


        time.sleep(60)
        with self._lock:
            self._current_rpm = 0

    def _reset_request_count(self):
        """
        Reset the request count and start a new timer for resetting the count after 60 seconds.

        This method resets the request count to 0 and starts a new timer to reset the count after 60 seconds.

        Raises:
            <Exception Type>: <Description of the exception raised>

        """

        
        with self._lock:
            self._current_rpm = 0
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(60.0, self._reset_request_count)
        self._timer.start()
