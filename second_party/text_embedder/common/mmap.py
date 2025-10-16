import os
import numpy as np
from typing import Tuple, Optional


class MemmapUtils:
    """
    Minimal utility to create and write to a numpy.memmap with periodic flushes.
    """

    def __init__(
        self,
        output_dir: str,
        filename: str,
        shape: Tuple[int, ...],
        dtype: np.dtype = np.float32,
        mode: str = "w+",
        flush_frequency: Optional[int] = None,
    ) -> None:
        self.output_dir = output_dir
        self.filename = filename
        self.shape = shape
        self.dtype = dtype
        self.mode = mode
        self.flush_frequency = flush_frequency

        os.makedirs(self.output_dir, exist_ok=True)
        self.path = os.path.join(self.output_dir, self.filename)
        self.memmap = np.memmap(
            self.path, dtype=self.dtype, mode=self.mode, shape=self.shape
        )

    def write_row(self, row_index: int, data: np.ndarray) -> None:
        self.memmap[row_index] = data
        if self.flush_frequency is not None and self.flush_frequency > 0:
            if (row_index + 1) % self.flush_frequency == 0:
                self.memmap.flush()

    def flush(self) -> None:
        self.memmap.flush()

    def estimated_megabytes(self) -> float:
        itemsize = np.dtype(self.dtype).itemsize
        total_bytes = int(np.prod(self.shape)) * itemsize
        return total_bytes / 1024.0 / 1024.0

    def read_row(self, row_index: int) -> np.ndarray:
        return self.memmap[row_index]


if __name__ == "__main__":
    mmap_utils = MemmapUtils(
        output_dir="test",
        filename="test.memmap",
        shape=(100, 100),
        dtype=np.float32,
        mode="w+",
        flush_frequency=10,
    )

    print(mmap_utils.read_row(0))
