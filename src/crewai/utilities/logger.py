class Logger:
    def __init__(self, verbose_level=0):
        """
        Initialize the class with a specified verbose level.

        Args:
            verbose_level (int): The level of verbosity for logging messages.

        Raises:
            None

        Returns:
            None
        """

        
        verbose_level = (
            2 if isinstance(verbose_level, bool) and verbose_level else verbose_level
        )
        self.verbose_level = verbose_level

    def log(self, level, message):
        """
        Log a message with the specified level.

        Args:
        - level (str): The log level, should be one of "debug" or "info".
        - message (str): The message to be logged.

        Raises:
        - ValueError: If the specified log level is not supported.

        Returns:
        None
        """


        level_map = {"debug": 1, "info": 2}
        if self.verbose_level and level_map.get(level, 0) <= self.verbose_level:
            print(f"\n[{level.upper()}]: {message}")
