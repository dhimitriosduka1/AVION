import os
import numpy as np

from tqdm import tqdm


def write_memmap(
    filename: str,
    data: np.ndarray,
    start_index: int = 0,
    end_index: int = None,
    dtype=None,
    mode="w+",
):
    """
    Writes a NumPy array to a memory-mapped file.

    Parameters:
        filename (str): Path to the file to create or overwrite.
        data (np.ndarray): Array data to write.
        start_index (int): Index to start writing from.
        end_index (int): Index to end writing at.
        dtype: Optional. Data type to use. Defaults to data.dtype.
        mode (str): File mode. 'w+' creates or overwrites the file.

    Returns:
        np.memmap: Memory-mapped object pointing to the written data.
    """
    dtype = dtype or data.dtype
    shape = data.shape

    # Create memmap file
    memmap = np.memmap(filename, dtype=dtype, mode=mode, shape=shape)

    for i in tqdm(range(start_index, end_index), desc="Writing memmap"):
        memmap[i] = data[i]
        if i % 10000 == 0:
            memmap.flush()

    memmap.flush()
    return memmap


def read_memmap(filename: str, dtype, shape, mode="r"):
    """
    Reads data from a memory-mapped file.

    Parameters:
        filename (str): Path to the file.
        dtype: Data type of stored array.
        shape (tuple): Shape of the stored array.
        mode (str): File mode. 'r' for read-only, 'r+' for read/write.

    Returns:
        np.memmap: Memory-mapped object for reading or writing.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    return np.memmap(filename, dtype=dtype, mode=mode, shape=shape)
