import sys


class EthiopianCoffeeException(Exception):
    """
    Custom exception for the Ethiopian Coffee ML project.

    Preserves:
    - original exception message
    - script / module filename
    - line number
    """

    def __init__(
        self,
        error_message: Exception | str,
        error_details: sys,
    ) -> None:
        super().__init__(str(error_message))

        self.error_message = str(error_message)

        if error_details is not None and hasattr(error_details, "exc_info"):
            _, _, exc_tb = error_details.exc_info()
            if exc_tb is not None:
                self.lineno = exc_tb.tb_lineno
                self.file_name = exc_tb.tb_frame.f_code.co_filename
            else:
                self.lineno = -1
                self.file_name = "<unknown>"
        else:
            self.lineno = -1
            self.file_name = "<unknown>"

    def __str__(self) -> str:
        return (
            f"\n\nError occurred in script: [{self.file_name}]\n"
            f"Line number: [{self.lineno}]\n"
            f"Error message: [{self.error_message}]"
        )
