# Mini-Project for the Class Optimizations for Machine Learning thaught at EPFL, Spring Semester 2022
## Title: The Art of Hypertuning: A Comparative Analysis of Different Optimization Techniques
### Members:
- Ana Lucia Carrizo, Msc Data Science, EPFL
- Amine Tourki, Msc Robotics, EPFL
- Mortadha Abderrahim, Msc Data Science, EPFL

### Description:
Fine tuning hyperparameters for Deep Neural Networks is one of the most challenging tasks in deep learning. Learning rate tuning has been a major focus and methods like schedulers and learning rate optimizers have been a major success. Over recent years, researchers have shown interest over other parameters such as momentum with optimizers like ADAM, momentum SGD or Nesterov Accelerated Gradient (NAG). In this project we further explore momentum tuning techniques such as momentum decay with two State-Of-The-Art optimizers: Demon and YellowFin. We compare their performances with already established optimizers like Adam and SGD. We conclude on the performance of momentum tuning techniques and their viability.

### Implementation Details
To run this project we took advantage of Google Colab's GPUs. To implement the project, one needs to add all files to a Google Drive common folder, and mount said folder in the main notebook. 
#### Built with:
- PyTorch
- Torchvision
- Sklearn
- Numpy
- Time
- Itertools
- matplotlib.pyplot

### Project's Structure
- `main.ipynb`: notebook that runs each model with their respectives hyperparameters and plots the accuracies evolution per epoch. 
- `hypertuning.ipynb`: notebook that finds the best parameters per model. 
- `SGDhypertuning.ipynb`: notebook that finds the best parameters per model for SGD based models (SGD, SGDM, SGD-Nesterov).
- `model.py`: script containing the definition of the neural network, and the function to reset its weights. 
- `helpers.py`: script containing all the helper functions we will need to find the hyperparameters and run the K-Fold Cross-Validation.
- `averaging.py`: contains the resultS from doing 3 runs.
- YellowFin: folder containing the files to use the YellowFin optimizer.
  - `yellowfin.py`: script were we define the YFOptimizer.
  - `helpers_yellowfin.py`: contains the helper functions to run the K-Fold Cross-Validation and find the hyperparameters for this model.
  - `yellowfin_mnist.ipynb`: example of simple implementation of YellowFin.
- DemonRangerOptimizer: folder containing the files to use the ADAM Demon optimizer.
  - `optimizers.py`: script were we define the DemonRanger optimizer.
  - `helpers_demon.py`: contains the helper functions to run the K-Fold Cross-Validation and find the hyperparameters for this model.
  - `DemonAdam.ipynb`: example of simple implementation of ADAM Demon.


