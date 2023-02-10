# **Parallel Programming Examples**

Usage of parallel programming tools such as `CUDA`, `MPI` and `OpenMP` with minimalist code examples, e.g. Game of Life, Matrix Multiplication and $\pi$ calculation.

_**NOTE: The last time I created and tested the following guide was in 2021. For at this moment, some things may have changed.**_

---

## **Table of Contents**

* [**OpenMP**](#openmp)
* [**CUDA**](#cuda)
* [**MPI**](#mpi)

---

## **1. OpenMP** <a id="openmp"></a>

`OpenMP` is a multi-platform programming interface that enables multiprocessing programming. `OpenMP` can be used in C++, C and Fortran languages, including different architectures like Windows and Unix. It consists of compilator directives that have an impact on code execution.

The `OpenMP` interface is a component of the GNU Compiler Collection (`GCC`), a set of open-source compilers developed by the GNU Project. `GCC` compiler is therefore highly recommended for use with `OpenMP`, although it is not required (there is an Intel compiler that also support `OpenMP`).

**INSTALLATION AND CONFIGURATION ON LINUX SYSTEMS:**

Start the terminal and update the repository:

```bash
>>> sudo apt-get update
```

Then install the `build-essential` package, including `gcc`, `g++` and `make`:

```bash
>>> sudo apt-get install build-essential
```

We can also install the manual pages on using GNU/Linux for programming, but it is not necessary:

```bash
>>> sudo apt-get install manpages-dev
```

To check `GCC` version, type:

```bash
>>> gcc --version
```

**INSTALLATION AND CONFIGURATION ON WINDOWS 10:**

On Windows, we need `MinGW`, a port of `GCC` providing a free, open environment
and tools that allow us to compile native executables for the Windows platform.
To do this, we go to: [https://sourceforge.net/projects/mingw/](https://sourceforge.net/projects/mingw/) and download `MinGW` - Minimalist GNU for Windows. Once installed, we check the compiler in the command line:

```bash
>>> gcc -v
```

Make sure that you've installed the `GCC` with the Posix thread model. If you get a message that the command is not recognised, add the appropriate environment variable to your system - "`../MinGW/bin`".

The use of `OpenMP` requires including a following library in the C/C++ code:

```c++
#include <omp.h>
```

It is also required to specify the appropriate flag during compilation:

```bash
>>> gcc -fopenmp -pedantic -pipe -O3 -march=native main.cpp -o main
```

The above flags mean:

- `-fopenmp` enables the execution of `OpenMP` directives,
- `-pedantic` is a standard error warning flag,
- `-pipe` causes that temporary files will be avoided, which speeds up build,
- `-O3` imposes a high degree of optimisation (be careful with this),
- `-march=native` generates code dedicated to the system on which it is compiled.

Only the `-fopenmp` flag is required for `OpenMP` to work. The others are optional flags which are worth using to optimise the code. We can read more about `GNU GCC` here: [https://gcc.gnu.org/](https://gcc.gnu.org/).

---

## **2. CUDA:** <a id="cuda"></a>

`CUDA` is Nvidia's universal architecture for multi-core processors (mainly graphics cards), allowing GPU to solve general numerical problems much more efficiently than traditional sequential general-purpose processors.

Working with Nvidia `CUDA` requires a dedicated graphics card from Nvidia that supports `CUDA` technology. If you have one, you can go to [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads) to download the `CUDA` Toolkit (select the appropriate operating system, architecture, version, etc.). After choosing the suitable options, you will also be shown a simple guide on what to do next to install the `CUDA` Toolkit.

Next, make sure that the Nvidia compiler works:

```bash
>>> nvcc
```

And check the current version:

```bash
>>> nvidia-smi
```

The compilation of a program using the `CUDA` architecture is performed as follows:

```bash
>>> nvcc main.cu -o main
```

You can also provide information for the compiler about the computing capability of your graphics card:

```bash
>>> nvcc -arch=sm_75 main.cu -o main
```

The `-arch=sm_75` flag informs compiler that you are equipped with graphics card with computing capability equal to $7.5$.

The `CUDA` Toolkit also allows the `nvprof` utility to view the operations performed on the graphics card and their execution time. To obtain such statistics, run the toolkit in the following way on Windows:

```bash
>>> nvprof main.exe
```

We can find a lot of useful information about `CUDA` in the official documentation:
[https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html). I strongly encourage you to familiarise yourself with this.

---

### **3. MPI** <a id="mpi"></a>

Message Passing Interface (`MPI`) is a communication protocol standard for transferring messages between parallel program processes on one or more computers. `MPI` is currently the most widely used communication model in clusters of computers and supercomputers.

There are several implementations of `MPI`, including `OpenMPI`, `MPICH` and `MSMPI`. On Linux, we can choose from `OpenMPI` and `MPICH`, while `MSMPI` is a Windows implementation. Before going any further, we should ensure that we have the `GCC` compiler installed.

**INSTALLATION AND CONFIGURATION OF `MPICH` ON LINUX SYSTEMS:**

Start the terminal and update the repository:

```bash
>>> sudo apt-get update
```

We then install the `mpich` package:

```bash
>>> sudo apt-get install mpich
```

We can now check the version of the installed `MPI` (this will actually be the `GCC` version):

```bash
>>> mpic++ --version
```

Here you can find out more about `MPICH`: [https://www.mpich.org/](https://www.mpich.org/).

**THE INSTALLATION PROCESS UNDER WINDOWS IS COMPLEX, AND I DO NOT RECOMMEND USING MPI WITH THE WINDOWS PLATFORM..., BUT IF YOU WANT TO HAVE FUN YOU HAVE TO CHOOSE `MSMPI`.**

More about `MSMPI`: [https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi).

We can use the `MPI` protocol to communicate between machines via the local network. In this way, you can run a program that will be executed in parallel by the available allocated threads on more than one machine. To carry out such a task, we need a minimum of two machines connected via the local network. The machines will communicate via the `SSH` protocol and exchange data via the `NFS` protocol. Step-by-step instructions on how this can be implemented on Linux systems is available at: [https://mpitutorial.com/tutorials/running-an-mpi-cluster-within-a-lan/.](https://mpitutorial.com/tutorials/running-an-mpi-cluster-within-a-lan/.)
