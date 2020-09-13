Table of Contents

# Installation
There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python. The code should run with no issues using Python versions 3.*.

# Project Motivation
For this project, I will use the CRISP-DM process to explore and ask questions of a free dataset about space rocket launch success.  

Business Understanding: Rockets appeal to our inner child curiosity, and a set of simple analyses about companies that develop and launch rockets may provide insight about determinants of launch success.

Data Understanding: The dataset used in these analyses contains information about each known space flight launch dating back to 1957. Each record represents a single launch or launch attempt, with summary information such as date, location, company name, rocket price, and whether or not the launch was a success.

Prepare Data: We walk through several steps to create an appropriate categorical response variable, as well as date formating and aggregation. 

Data Modeling: We use a simple Machine Learning linear regression method to predict launch success as a function of 1 or more predictor variables. Here, we use the sklearn library in Python.

Evaluate the Result: We will briefly evaluate the results (descriptive and inferential statistics) and comment on the utlility of these results for understanding space rocket launch success. 

# File Descriptions
There is a data file [datasets_828921_1417131_Space_Corrected.csv] obtained from https://www.kaggle.com/agirlcoding/all-space-missions-from-1957. 
There is a notbook availible here to showcase work related to the above questions. 
There is a folder 'Misc_files' that contains images used in Medium.com blog post, as well as a .py file used for data analyses.

# How to Interact with this Project
This project was created for a Udacity Data Science course and is intended to showcase some very simple python data manipulation and data analysis. Please look at the blog, notbook and .py file for details, or read the blog post on medium.com here:
https://medium.com/@turtlewheels/not-all-rocket-launches-are-successful-ad2634d4cf94


# Licensing, Authors, Acknowledgements
Please give any credit to Agirlcoding at Kaggle for compiling data. You can find the Licensing for the data and other descriptive information at the Kaggle link available here. Otherwise, feel free to use the code here as you would like!

