# Finding the Higgs Boson - Machine Learning challenge

The Higgs boson is a particle that gives other particles their mass and its 
discovery is crucial, indicating the existence of new physics principles. 
Hence, the problem of classifying if a particle is a Higgs boson (signal) or 
some other process/particle (background) is undoubtedly significant. 
Improvements in the current solutions are still needed and highly valuable.

We explore various methods and propose a binary classifier based on logistic regression, 
which has achieved an accuracy of 79.9\% on the AICrowd platform.

### Model structure

- All the provided data from CERN is in folder `/data`, where we have 2 files:
<br />`train.csv` - Training set of 250000 events. The file starts with the ID column, then the label column, and finally 30 feature columns.
<br />`test.csv` - The test set of around 568238 events the same as above mentioned except the label is missing.
  > The dataset was downloaded from -> https://github.com/epfml/ML_course/tree/master/projects/project1/data.

- In the folder `/docs` you can find the description of the project (`project1_description.pdf`) and our report (`report.pdf`). 

- The folder `/pretrained_data` contains files with the weights obtained while training on the best set of parameters.

- The folder `/scripts` has the following files:
<br />`data_processor.py` - All the preprocessing and refining of the raw data. Here we have methods that standardize the data, scale, split data into different sets (based on jet numbers or to train and test set), feature expansion, etc. 
<br />`model.py` - Contains methods for training and validating the model. Also, the predictions, final evaluations, and creating submission is implemented here.
<br />`implementation.py` - Here are the 6 methods used for classification: linear regression using gradient descent, linear regression using stochastic
gradient descent, least squares regression using normal equations, ridge regression using normal equations, logistic regression using GD or SGD,
and regularized logistic regression using GD or SGD. Also, the file contains additional methods such as losses and gradients computations.
<br />`run.py` - The model runner, it runs the model with the best parameters.
<br />`plots.py` - Methods for plotting the data.
<br />`proj1_helpers.py` - Helpers received for the project.

- There is also `project1.ipybn` - a notebook that contains the whole code and can provide more insight into understanding our approach.

### Running the model

1. To run the code you first need to have *Python* installed. Also, the only external library for the model used in *numpy*, but if you want to generate plots you should also install *matplotlib*.
> The needed libraries are stated in `requirements.txt`, to install them run: `pip install -r requirements.txt` (Python 2), 
or `pip3 install -r requirements.txt` (Python 3).

2. To run the model with our best results simply run the python script `run.py`.

Another way of running the model is opening the `project1.ipybn` as *Jupyter notebook* or in 
*Google Colab*. There, you can run all the cells and experiment further.

### Authors

- Irina Bejan: irina.bejan@epfl.ch
- Nevena Drešević: nevena.dresevic@epfl.ch
- Marija Katanić: marija.katanic@epfl.ch
