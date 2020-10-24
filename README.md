# Finding the Higgs Boson - Machine Learning challenge

The Higgs boson is a particle that gives other particles their mass and its 
discovery is crucial, indicating the existence of new physics principles. 
Hence, the problem of classifying if a particle is a Higgs boson (signal) or 
some other process/particle (background) is undoubtedly significant. 
Improvements in the current solutions are still needed and highly valuable.

We explore various methods and propose a binary classifier based on logistic regression, 
which has achieved an accuracy of 79.9\% on the AICrowd platform.

### Model structure

The folder `/scripts` has the following files:
<br />`run.py` - The model runner, it runs the model with best parameters.
<br />`implementation.py` - Here are the 6 methods used for classification: least squares  and its helper functions
<br />`data_processor.py` - All the preprocessing and refining of the raw data. 
<br />`plots.py` - Methods for plotting the data.
<br />`proj1_helpers.py` - Helpers received for the project.

In the folder `/docs` you can find the description of the project (*project1_description.pdf*) and our report (*report.pdf*). 

There is also `project1.ipybn` - notebook that contains the whole code and can provide more insight into understanding our approach.

### Running the model

1. To run the code you first need to have *Python* installed. Also, the only external library for the model used in *numpy*, but if you want to generate plots you should also install *matplotlib*.

2. You can find the original dataset on the link -> 
https://github.com/epfml/ML_course/tree/master/projects/project1/data. The
following two files are important:
<br />`train.csv` - Training set of 250000 events. The file starts with the ID column, then the label column, and finally 30 feature columns.
<br />`test.csv` - The test set of around 568238 events the same as above mentioned except the label is missing.
<br /><br />Download the data, unzip it and place it into folder `/data`.

3. To run the model simply run the python script `run.py`.

Another way of running the model is opening the `project1.ipybn` as *Jupyter notebook* or in 
*Google Colab*. There, you can run all the cells and experiment further.

### Authors

- Irina Bejan: irina.bejan@epfl.ch
- Nevena Drešević: nevena.dresevic@epfl.ch
- Marija Katanić: marija.katanic@epfl.ch
