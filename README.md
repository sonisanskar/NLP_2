# NLP_2
In order to test the models ,you must give path to the train data and the test data which must have the columns as 'title' and 'tag' with 
values as 0(fake) and 1(real).You must have word 2 vec pretrained weights or can alternatively download from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
After running the py file ,
</br>python3 run_models.py</br>
Steps:</br>
1. First it will ask to input the path of training data file and subsequently the test file.</br>
2. It will show the data_train and data_test imported.</br>
3.Then it will ask us for the max length of para which can be selected from based on results which willl be shown on terminal above.</br>
4.It will ask for the first dimension in which para has to be broken and subsequently the second dimension.</br>
5.It will automatically run all the models and ouput the results on terminal and also a file would be created named results.csv</br>
The confusion matrix and the classifcation report will be there on terminal.
6.Also all the trained models will be saved respectively by mod1, mod2,and yoonkim</br>

For trying purpose ,I am uploading a trainn.csv and testt.csv file also .You may download these files and the py script and run on your system.
