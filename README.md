# MaxNormRegCollaborativeFiltering_SGD

This is a simplified implementation of:

> LEE, Jason D., et al. Practical large-scale optimization for max-norm regularization. In: Advances in Neural Information Processing Systems. 2010. S. 1297-1305.

*"Simplified"* means:

- Positive:
  - it works (when a Movielens dataset is accessible)
  - it can provide very good results beating less complex models (SVD-based; Trace-norm and co.) with parameter-tuning (especially for large-scale datasets; e.g. Movielens 20M)
  - it's vectorized
- Negative:
  - the core-loop is a python-loop
    - should be cythonized for good speed
  - there are no potential improvements provided (they exist somewhere hidden on my HDD) like:
    - Simple momentum (easy)
    - Variance-reduction techniques like AdaGrad

A warning:

- Own research shows that naive minibatch-like processing of course improves performance, but convergence is hurt a lot!
  - See next paragraph for (scientifically) well-received alternative approaches!

For more parallelization look at the provided code by the authors:

- [Jellyfish](http://i.stanford.edu/hazy/victor/jellyfish/)
  - a bit of work; but not that hard to implement in python
- [HOGWILD!](http://i.stanford.edu/hazy/victor/Hogwild/)
  - much harder to implement in python as one needs a good understanding of lock-free parallelization (easier using C/C++ & OpenMP)

For more user-friendly variance-reduction techniques (less hyper-parameter tuning needed), see:
- [libmf](https://www.csie.ntu.edu.tw/~cjlin/libmf/)
  - high-quality / high-performance implementation
  - Scientific papers available
  - **But**: implementation does support *trace-norm*, but not *max-norm*!

For more theory about different regularizations including motivation, see:

> SREBRO, Nathan; SHRAIBMAN, Adi. Rank, trace-norm and max-norm. In: International Conference on Computational Learning Theory. Springer, Berlin, Heidelberg, 2005. S. 545-560.

This project is *somewhat* connected to the (non-decomposed) SDP-solver-based approach in [pyMMMF](https://github.com/sschnug/pyMMMF) (which only work for toy-datasets; e.g. 100x100).

A non-tuned output could look like:

    train size
    (800000, 2)
    (800000,)
    test size
    (200209, 2)
    (200209,)
    epoch:  0 / 50
    epoch:  1 / 50
     calc train-error:
     ->  0.8842490091736446
     calc test-error:
     ->  0.916555252398467
    epoch:  2 / 50
    epoch:  3 / 50
     calc train-error:
     ->  0.8041708542450389
     calc test-error:
     ->  0.896428026026099
    epoch:  4 / 50
    epoch:  5 / 50
     calc train-error:
     ->  0.7841972473358629
     calc test-error:
     ->  0.8927339518280402
    epoch:  6 / 50
    epoch:  7 / 50
     calc train-error:
     ->  0.7740798205973093
     calc test-error:
     ->  0.8891171873066838
    epoch:  8 / 50
    epoch:  9 / 50
     calc train-error:
     ->  0.7648603771886379
     calc test-error:
     ->  0.8842822451534099
