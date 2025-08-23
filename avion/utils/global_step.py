class GlobalStep:
    """Global step tracker using class-level state."""

    _step = 0

    @classmethod
    def increment(cls, value=1):
        """
        Increment the global step by a specified value.

        Args:
            value (int): The value to increment the step by. Defaults to 1.
        """
        cls._step += value

    @classmethod
    def get(cls):
        """
        Get the current value of the global step.

        Returns:
            int: The current global step value.
        """
        return cls._step

    @classmethod
    def set(cls, value):
        """
        Set the current value of the global step.

        Args:
            value (int): The value to set the global step to.
        """
        cls._step = value

    @classmethod
    def get_and_increment(cls):
        """
        Get the current value of the global step and increment it.

        Returns:
            int: The current global step value before incrementing.
        """
        current_step = cls._step
        cls.increment()
        return current_step
