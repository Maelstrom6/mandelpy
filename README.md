# mandelpy
A Mandelbrot and Buddhabrot viewer with GPU acceleration using NVIDIA's CUDA toolkit.

# Requirements
- Python 64bit

- NumPy

  Standard I assume.

- Pillow

  The main image manipulation library.
  
- Numba

  Requires the Python 64bit interpreter. Chances are, if you managed to install TensorFlow for
  GPU, you would simply be able to type `$ pip install numba` with no problems and no extra steps. 
  Otherwise, according to
  [their documentation](https://numba.pydata.org/numba-doc/latest/cuda/overview.html#requirements), 
  one needs the CUDA Toolkit 8.0 or later. Please run the `numba_cuda_test.py` to check if
  everything is working properly.
