import pandas as pd
import numpy as np

class GussianNB:
    def __init__(self, features, labels, x_test):
        self.features = features # out features to train the model
        self.labels = labels # labels to classify
        self.num_class = len(np.unique(labels)) # how many categories we have? 
        self.classes = np.unique(labels) # Store the names for each class
        self.x_test = x_test # test set
        self. prior = 1 / self.num_class # we are goint to use the perior to implement the probability part 
        
    def fit(self):
        features = pd.DataFrame(self.features) # convert the array to Data frame to ease the calculation process
        labels = pd.Series(self.labels)
        mean = features.groupby(by = labels).mean() # separate the features  into different categorical gropus then calculate the mean
        variance = features.groupby(by = labels).var() #calculate the variance for each groups
    
        return (mean.values, variance.values)
    
    def predict(self, mean, variance):
        mean_var = list() # recorde the mean and variance
        prediction_list = list() # predicted value
        final_probs_list = list() # the final calculation of the probability 
        
        for i in range(len(mean)): # according to each feature and groups we have different mean and variance. Loop through each of them
            m_row = mean[i]
            v_row = variance[i]
            
            for index, value in enumerate(m_row):
                mean_val = value
                var_val = v_row[index]
                mean_var.append([mean_val, var_val]) # now for each category we have their own mean and variance.
                
        mean_var_arr = np.array(mean_var) # convert mean_var list into array
        separated_mean_var = np.vsplit(mean_var_arr, self.num_class) # seperate the to different arrays according to number of the classes.
        
        for k in range(len(self.x_test)):
            prob_list = list()
            final_prob = list()
            
            for i in range(self.num_class):
                array_class = separated_mean_var[i] # Loop through each mean and variance for each class

                for j in range(len(array_class)): # seperate mean and variance. then, use test data to implement Guassian naive bayse
                    class_mean = array_class[j][0]
                    class_var = array_class[j][1]
                    x_values = self.x_test[k][j]

                    prob_list.append([self.gnb_equation(x_values, class_mean, class_var)]) # calcualte the probability for each test data then save the record.

            prob_array = np.array(prob_list)
            separated_prob = np.vsplit(prob_array, self.num_class) # sperate the calculated probability to the number of classes

            for i in separated_prob: # loop through each prob_values
                class_prop = np.prod(i) * self.prior
                final_prob.append(class_prop)

            maximum_prob = max(final_prob) # find the maximum probability for class in order to find the answer 
            final_probs_list.append(maximum_prob) # save the answer 
            prop_max_index = final_prob.index(maximum_prob) # find the index of the answer
            prediction = self.classes[prop_max_index] # which class has the highest prob_value
            prediction_list.append(prediction) # save the class name.
            
        return (prediction_list, final_probs_list)
    
    def gnb_equation(self, sample, mean, variance):
        # this function had exactly implemented like the Guassian equation.
        pi = np.pi
        
        equation_part_1 = 1 / np.sqrt(2 * pi * variance)
        equation_part_2 = np.exp(-((sample - mean))**2 / (2 * variance))
        final_equation = equation_part_1 * equation_part_2
        
        return final_equation