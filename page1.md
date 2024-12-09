<script type="text/javascript">
MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
  }
};
</script>
<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.js">
</script>

Test

Inline formula $E=mc^2$.

Test

$$\int_a^b f(x) \, dx = F(b) - F(a)$$

PyCUDA
PyCUDA is a very useful tool to embed low level programming on Graphics Processing Units (GPUs) with CUDA in a higher level programming framework provided by Python. It makes available a whole bunch of facilities to perform a step-by-step code debugging by checking intermediate variable values using breakpoints, simple prints, plots of vectors or images of matrices. In its interactive shell version, for example using Jupyter notebook, PyCUDA coding is even simpler. Jupyter is however just a possibility to be exploited locally, but free on line services also exist, like Kaggle or Google Colaboratory.

Coding with PyCUDA occurs without significant loss of performance due to the use of a high lever language. Indeed, acceleration using GPU parallel programming occurs by splitting the code into a sequential or mildly parallelizable part to be executed on a (possibly multicore) CPU and into a massively parallelizable part to be executed on GPU. Typically, the CPU is just a controller which schedules the GPU executions so that there is no significant penalty when using Python. Opposite to that, GPU coding can be worked out directly using CUDA.

PyCUDA is documented at the PyCUDA homepage which contains also a tutorial, it has a GitHub page where the source code is available and issues are discussed and some useful examples are available . This notwithstanding, a didactic getting started guide is missing. Therefore, I decided to contribute to the topic with this post having the aim of providing a smooth introduction to coding in PyCUDA. To this end, I will discuss five different ways to implement the classical example by which parallel programming on GPU is taught, namely, the elementwise sum of two vectors, an operation very common in scientific computing.

Basics of CUDA programming and of Python coding will be assumed. CUDA basics prerequisites can be reached by the classical CUDA By Example book.

To form the five examples, I will consider different possibilities offered by PyCUDA, namely, using:

the SourceModule module;
the ElementwiseKernel module;
the elementwise sum of two gpuarray’s.
Different possibilities may have different performance. For this reason, I will assess the performance of each version by the execution times on a Maxwell GeForce GTX 960 GPU.


Version 1: using SourceModule
The module SourceModule enables coding GPU processing directly using CUDA __global__ functions and to execute the kernels by specifying the launch grid.

In the code below, SourceModule is imported at line 6.

Lines 11 and 12 define the iDivUp function which is the analogous of the iDivUp function typically used in CUDA/C/C++ codes (see High Performance Programming for Soft Computing, page 103). It is used to define the number of blocks in the launch grid.

Lines 18 and 19 define CUDA events (see CUDA By Example) which will be subsequently used, on lines 56 and 58–61, to evaluate the execution times.

Later on, line 21 defines the number of vector elements and line 23 the size (BLOCKSIZE) of each execution block.

Lines 25–31 define, through the numpy library, the two random CPU vectors (h_a and h_b) to be transferred to GPU and summed thereon.

On the GPU, the space for these random vectors is allocated by the mem_alloc method of the cuda.driver at rows 34–36. Note that line 36 also allocates the global memory space to contain the results of the computations. The CPU-to-GPU memory transfers are executed at rows 39–40 by memcpy_htod. There also exist other possibilities to implement allocations and copies. One of these is offered by the gpuArray class and an example will be illustrated next, while another possibility is to link the CUDA runtime library (cudart.dll) and directly use its unwrapped functions, but this latter option is off topic for this post.

Rows 42–50 define the deviceAdd __global__ function appointed to perform the elementwise sum, row 53 defines a reference to deviceAdd, rows 54–55 define the launch grid while line 57 invokes the relevant __global__ function. Lines 64–65 allow the allocation of CPU memory space to store the results and the GPU-to-CPU memory transfers.

Finally, rows 67–70 check whether the GPU computation is correct by comparing the results with an analogous CPU computation.

Finally, line 73 has no effect in this code, but is kept for convenience. Whenever one decides to test the code into an interactive python shell and to use printf() within the __global__ function, such instructions would enable the flush of the printf() buffer. Without those, into an interactive, the printf() whould have no effect into an interactive python shell.

The processing time of the elementwise sum has been 0.0014ms.


Version 1 using SourceModule
Version 2: using SourceModule and copying data from host to device on-the-fly
The second version is the same as the foregoing one with the only exception that the copies from host to the device and viceversa are not performed explicitly before the kernel launch, but rather implicitly. Implicit copies are executed on-the-fly by applying cuda.In to the host input arrays and cuda.Out to the output host array (see line 50).

The code is now shorter, but simplicity is paid with execution times. Indeed, memory transfers now affect the computation time which becomes 0.957ms.


Version 2 using SourceModule and copying data from host to device on-the-fly
Version 3: using gpuArrays
In the third version, GPU arrays are dealt with by the gpuarray class. The elementwise sum is then performed by using the possibility offered by such a class of expressing array operations on the GPU with the classical numpy array syntax without explicitly coding a __global__ function and using SourceModule.

As compared to the first version, we have now a timing penalty since the elementwise execution requires 1.014ms.


Version 3 using gpuarrays
Version 4: using ElementwiseKernel
The PyCUDA ElementwiseKernel class allows to define snippets of C code to be executed elementwise. Since the __global__ deviceAdd function contains operations to be executed elementwise on the involved vectors, we are suggested to replace the use of SourceModule with ElementwiseKernel.

The code below reported conceptually represents version 1 with SourceModule replaced with ElementwiseKernel. Actually, now a linear combination of the involved vectors instead of a simple elementwise sum is performed. Lines 30–33 define the elementwise linear combination function lin_comb while line 36 calls it. In this way, it is also possible to illustrate how passing constant values.

The computation time is 0.1034ms so that, as compared to version 1, ElementwiseKernel seems to give rise to a loss of performance as compared to SourceModule.

It is nevertheless possible to check that the additional overhead brought by the linear combination is irrelevant. Actually, as compared to version 1, in version 4 the vectors are dealt with by the gpuarray class, while in version 1 the global memory allocation was performed by mem_alloc. gpuarray may than be suspected to be the responsible of the additional overhead. Actually, this is not the case as shown by the code in version 5.


Version 4 using ElementwiseKernel
Version 5: using SourceModule while handling vectors by gpuArray
With the aim of verifying whether gpuarray is responsible of the increase of the execution times of the previous version, the code below reconsiders version 1 while dealing now the vectors by gpuarray instead of mem_alloc.

The execution time keeps 0.0013ms, just like in version 1, so that the use of the gpuarray class is not responsible of any execution overhead.


Version 5 using SourceModule while handling vectors by gpuArray
