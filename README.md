# Efficient Parallel Gradient Domain Image Fusion Systems

Teammates: Lichen Jin, Jiuzhi Yu

URL: https://arceus48.github.io/15618-Proposal/

## Summary: 
For this project, we plan to implement the conjugate gradient image integration algorithm for  Image Gradient Fusion on GPU and multi-core CPU platforms. We plan to perform system benchmarks and observe the performance and efficiency of various parallel programming systems (OpenMP, ISPC, CUDA) on homogeneous and heterogeneous architectures.

## Background:
Gradient Image Fusion is a general and useful task we use in image editing. Though conjugate gradient descent proves to be a good solver to the problem, the algorithm can take a great number of iterations to converge and hence is very slow if executed sequentially. The algorithm is as the pseudo-code below:

We find that most of the algorithms are per-pixel operations or Laplacian convolution. So the computation-intensive algorithm can be well parallelized if divided by pixel data. Specifically, all per-pixel arithmetics and convolutions can be well parallelized except for the serialized part of dot product and start/end of each iteration. Therefore, it is worth exploring the performance of various parallel frameworks on various architectures including GPU and multi-core CPU.

## Challenge:
The whole image will be broken down into 2D blocks; most operations like arithmetics and convolutions have good data locality; however, data communication and explicit synchronizations may happen at the edge of convolutions and the reduction sum of dot products. To tackle that challenge we look into the literature and follow the suggestions to make non-trivial operations faster in the parallel systems.

Specifically, there are two non-trivial specific operators that need acceleration: Laplacian Convolution & Dot Product. We plan on following the suggestion in the paper Sparse matrix solvers on the GPU: conjugate gradients and multigrid.

## Resources:

Codebase: 
We plan to start from scratch without using any publicly available code base.

Dataset: 
We plan to use the demo datasets for a few different gradient fusion tasks, such as the poisson image pairs provided in the Poisson image editing paper [1].

Computing Resource: 
We will use the GHC machines for GPU and the PSC machines for a multi-core CPU environment.

Papers:
1. Georg Petschnigg, Richard Szeliski, Maneesh Agrawala, Michael Cohen, Hugues Hoppe, and Kentaro Toyama. 2004. Digital photography with flash and no-flash image pairs. ACM Trans. Graph. 23, 3 (August 2004), 664–672. https://doi.org/10.1145/1015706.1015777
2. Jeff Bolz, Ian Farmer, Eitan Grinspun, and Peter Schröder. 2003. Sparse matrix solvers on the GPU: conjugate gradients and multigrid. ACM Trans. Graph. 22, 3 (July 2003), 917–924. https://doi.org/10.1145/882262.882364
3. James McCann and Nancy S. Pollard. 2008. Real-time gradient-domain painting. ACM Trans. Graph. 27, 3 (August 2008), 1–7. https://doi.org/10.1145/1360612.1360692
4. Patrick Pérez, Michel Gangnet, and Andrew Blake. 2003. Poisson image editing. In ACM SIGGRAPH 2003 Papers (SIGGRAPH '03). Association for Computing Machinery, New York, NY, USA, 313–318. https://doi.org/10.1145/1201775.882269
5. Computational Photography (CMU 15-663). 2022. http://graphics.cs.cmu.edu/courses/15-463/.


## Goals and deliverables:

Goals we plan to achieve: Successfully build the algorithm for a GPU version and a multi-core CPU version (ISPC or OpenMP). After we finish the implementations, we will perform data analysis to reason about the performance differences on the two different platforms. For performance expectations, we hope to get a near-linear speedup.

Goals we hope to achieve: Besides the goals above, we hope to build a message passing version of the algorithm. Then we hope to analyze the performance differences between the three different parallel implementations and investigate the more suitable platform for parallelizing our problem.

Demo at the poster session: 
Image demo: We plan to show some images processed with the gradient image fusion algorithm and also showing how to run it through the command line in real-time.
Performance data: We plan to present the speedup graphs of the parallel implementations we make, and also provide time breakdown for the two or three versions of the implementations, including the serial portion and the parallel portion.

## Platform choice:
Image processing includes many similar small subtasks on pixels. The nature of the problem makes the GPU a good platform to perform parallel processing. We also want to compare the performance of the GPU with the multi-core CPU to gain a deeper understanding on how to choose between GPU and CPU for parallelizing an image processing task. 

## Schedule:
| Week          | Task                                                                                                                                      |
|---------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| 11.14 - 11.20 | Study about the papers and understand the algorithm.  Work on the implementations on the serial version.                                  |
| 11.21 - 11.27 | Work on CUDA implementations. Start gathering images with different characteristics.                                                      |
| 11.28 - 12.4  | Work on ISPC/OpenMP implementations. Continue gathering images.                                                                           |
| 12.5 - 12.11  | Perform experiments on the parallel implementations.  Analyze the performance data. If there is extra time, work on a MPI implementation. |
| 12.11 - 12.18 | Work on the final report. Work on the project poster.                                                                                     |