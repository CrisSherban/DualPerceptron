# DualPerceptron

### Disclaimer
This is a work in progress, coming soon: K-Fold Cross Validation

### Usage
*   ### Perceptron Class:  
    *   In this implementation the Perceptron offers several methods
        for different use cases.  
        The Perceptron constructor creates or loads a Gram Matrix according to 
        the given kernel.  
        Let's explore an example of usage: to train and test on a given dataset one must create 
        a Perceptron first, then use the method `fit()` and finally the method 
        `predict_set()` as follows:  
        
        ```python  
            clf = Perceptron.Perceptron(dataset_name, train_x, train_y, 
                                        kernel, epochs, sigma, dim)  
            clf.fit()  
            predicted = clf.predict_set(test_x, test_y)  
            accuracy = clf.accuracy(test_y, predicted)  
        ```
        
    *   It is also possible to predict a single element:  
        
        ```python
            clf.predict_element(an_element)
        ``` 

*   ### OptimizedTools.py
    This files is the heart of the perceptron, it can't be used as a standalone.
    A Perceptron object delegates the fitting, the prediction and the creation of a 
    Gram Matrix to this file.
    
*   ### DatasetTools.py
    This python file loads a dataset properly for a Perceptron,
    for example, it contains functions to normalize the dataset and 
    to convert string features to numerical features. 
    This is an example of a basic usage:  
    
    ```python
        train_set, test_set = load_dataset(dataset_name, split_train_percentage,
                                            normalize=False, standardize=False)
    ```  
    The file also includes a function that allows to test the 
    linear separability of a given dataset using scipy.optimize.linprog()  
    
    ```python
        print(verify_linear_separability(train_set, test_set))
    ```  
    
*   ### KernelValidation.py
    This .py script plots a gram matrix of some given dataset.
    The dataset is pre-ordered based on the labels so the gram 
    matrix can be used to visualize the classes of the dataset
    It is meant to be used as a standalone in this project.
    
    This is an example of the output:  
    <img src="/Pictures/gender_voice_gram_mat_3_ordered.png" width="400px">  
    
*   ### Plotter.py
    This file contains two functions to plot the hyperplane and the elements 
    in 2D or 3D. The datasets used have more than three features so the output 
    in this case is just for the scope of the illustration.
    Basic usage:
    ```python
        Plotter.plot_2d(clf, X, y)
    ```  
    <img src="/Pictures/2D Hyperplane Decision Boundary gender_voice.png" width="400px">
        
    The 3D plot function can be used as follows:
    ```python
        Plotter.plot_3d(clf, X, y)
    ```
    This saves 70 .png files of the 3d scatter plot and hyperplane that
    can be converted to a .gif file from a linux terminal as follows:
    ```
        convert -delay 10 plot_step*.png animated_plot.gif
    ```
    <img src="/3D_plots/animated_plot.gif" width="400px">  
    
*   ### NaiveCrossValidation.py
    This script can be used to tune the parameters of the Kernels.
    In the current version this is a naive implementation that just 
    chooses the best out of a series of
    pre-determined parameters and then prints them.
    Usage:  
    ```python
        dataset_name = "mushroom"
        train_set, test_set = dt.load_dataset(dataset_name, 70, standardize=True)

        cross_validate_parameters(dataset_name, train_set, test_set)
    ```  
    
*   ### Test.py
    This file can be used to test everything out, for example,it can be used to analyze results and accuracies
    with various kernels. The function that does that is called `analyze_accuracies(datasets_names)`
    which creates for each dataset three different classifiers with different kernels, then trains each
    classifier for 10 epochs and chooses the epoch that performs best. So for each epoch the classifier
    is trained and tested. Finally it prints a bar chart of the accuracies so the user can verify the results.  
    The user can choose to set the boolean variable test_on_train to True, this allows the function 
    to analyze the training set as a test set. 
    Basic usage:   
    
    ```python
    datasets_names = ["bank_marketing", "gender_voice", "mushroom"]
    analyze_accuracies(datasets_names, test_on_train=False)
    ```   
    
    
            
    
        
        
        
        
