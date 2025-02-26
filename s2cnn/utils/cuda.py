# pylint: disable=R,C,E1101
from collections import namedtuple
from cupy.cuda import function  # pylint: disable=E0401
# from pynvrtc.compiler import Program  # pylint: disable=E0401
from cuda.core.experimental import Program


CUDA_NUM_THREADS = 1024
CUDA_MAX_GRID_DIM = 2**16 - 1


def get_blocks(n, num_threads):
    n_per_instance = (n + num_threads * CUDA_MAX_GRID_DIM - 1) // (num_threads * CUDA_MAX_GRID_DIM)
    return (n + num_threads * n_per_instance - 1) // (num_threads * n_per_instance)


Stream = namedtuple('Stream', ['ptr'])


def compile_kernel(kernel, filename, functioname):
    # program = Program(kernel, filename)
    program = Program(kernel, code_type="c++")
    ptx = program.compile("cubin", name_expressions=[functioname])

    m = function.Module()
    # m.load(bytes(ptx.code()))
    m.load(ptx.code)
    f = m.get_function(functioname)
    return f
