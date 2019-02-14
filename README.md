# GeoCert

Geometric-inspired algorithm for solving the general problem of finding the largest $$\ell_p$$ ball centered at a point $$x$$ for a union of polytopes which are *"perfectly-glued"*. Algorithm is provably correct for $$p\geq1$$. Primary application found in certifying the adversarial robustness of multilayer ReLu networks. Some example results:

Maximal $$\ell_2$$ projections           | Network Input Partioning
-----------------------------------------|-----------------------------------------
<img src="https://github.com/revbucket/geometric-certificates/blob/master/paper/Figures/%5B50%2C8%2C2%5D/non_robust_projections_l2.svg" alt="example" width="400"> | <img src="https://github.com/revbucket/geometric-certificates/blob/master/paper/Figures/%5B50%2C8%2C2%5D/locus_PG_plot_non_robust.svg" alt="mnist_reconstr" width="400">

---

# Primary Contents
This code base includes implemented methods for both the *"batched"* and *"incremental"* versions of the algorithm. See ipython notebooks for experiments and examples. 

### Functions:
* `batch_GeoCert`: finds largest $$\ell_p$$ ball centered at $$x$$ within a union of polytopes passed in as a list. 
* `incremental_GeoCert`: solves the same problem, but specifically for certifying the robustness of a multilayer ReLu network. 

### Scripts:
* __Experiment_1:__ demonstration to find the maximal $$\ell_p$$ ball at a point $$x_0$$, within which, the class label of a random classifier is equal to $$C(x_0)$$.
* __Experiment_2:__ similar to first experiment, but first the network is trained to classify random points in $$\mathbb{R}^2$$. The effects of $$\ell_1$$ regularization in training are demonstrated. 
* __Experiment_3:__ estimate the reduction in the number of encountered polytopes when $$\ell_1$$ regularizaiton is utilized. Again, classifier trained on random data in $$\mathbb{R}^2$$.
* __Experiment_4:__ estimate the number of encountered polytopes when varying the neural network architecture / capacity. Again, classifier trained on random data in $$\mathbb{R}^2$$ 

---

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

One can find the necessary prerequisite packages within the file: `requirements.txt` . 

### Installing

1. Clone the repository:
    ```shell
    $ git clone https://github.com/revbucket/geometric-certificates
    $ cd geometric-certificates
    ```
2. Install requirements:
    ```shell
    $ pip install -r requirements.txt
    ```
---

## Running the tests

With the codebase installed, run the ipython notebooks provided to get your hands on the algorithm and visualize its behaviour. As an example:

```shell
$ jupyter notebook Experiment_1.ipynb
```	

---

## Authors

* **Matt Jordan-** University of Texas at Austin 
* **Justin Lewis-** University of Texas at Austin 


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


