# DualPerceptron


### Usage
*   #### Perceptron Class:  
    *   In this implementation the Perceptron offers several methods
        for different use cases.
        To train and test on a given dataset one must create 
        a Perceptron first, then use the method fit() and finally the method 
        predict_set() as follows:  
        
        ~~~python  
            clf = Perceptron.Perceptron(dataset_name, train_x, train_y, 
                                        kernel, epochs, sigma, dim)  
            clf.fit()  
            predicted = clf.predict_set(test_x, test_y)  
            accuracy = clf.accuracy(test_y, predicted)  
        ~~~
        
        
        
        
