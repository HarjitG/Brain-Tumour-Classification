# Brain-Tumour-Classification

In this project I will we using Convulational Neural Networks (CNNs) to classify different types of Brain Tumours

The dataset contains 4 different classes, 3 of which are different types of Tumors, and one of which is not a Tumour

I have taken 3 different approaches to this project. The first was running a model on all 4 classes, however due to the number of Tumors being classed as non-tumors, I am experimenting with a model with 2 parts:
1) Classsifing Tumour vs Non Tumour
2) Classifying the 3 differnet types of Tumours

Colab_Notebook_Binary contains the 4 classes merged into 2 to give us Tumour vs Non Tumour. Colab_Notebook_3d contains the 3 classes of different tumors

Colab_Notebook_4d Contains all 4 classes and is the main notebook used in this project. We can see that the other two were not performigng as well, and hence this notebook is the focus of the project. From the notebook we can see that the model is not picking up Glioma Tumours well, however is picking up the other 3 classes well. Even with the use of Data augmentiontation and regularisation techniques I was not able to get a model performing well across all 4 classes well, as shown below.

![Confusion-Matix](/images/Confusion_matrix.png)

I have also built a Flask web application which runs the model in an easy to use UI allowing users to upload an image via their file explorer.

To run the notebooks or Flask web app, open the terminal, navigate to where you want to save the folder, and run the folowing commands:

## Setting up

Ensure that you have docker desktop installed on your machine and running

Clone Repository in desired location
```shell
git clone https://github.com/HarjitG/Brain-Tumour-Classification.git
```
Navigate to the folder
```shell
cd Brain-Tumour-Classification
```
## 1) For those using VSCode


Launch VSCode and re-open the project in a Dev Container when prompted

Open a Terminal

Ensure Pipenv is installed
```shell
pip install pipenv
```

Initialise Virtual Enviornment
```shell
pipenv shell
```

Install required librarys for virtual enviornment
```shell
pipenv install
```

#### The Collab_Notebook_4d can now be ran.


Navigate to the Flask_app folder
```shell
cd Flask_app
```
Run the following code to start the application:
```shell
python Flask.py
```

## 2) For those using the Computer's Terminal

Run the following to launch VSCode
```shell
code . 
```
When propmpted open the folder in a container

Run the following and copy the container_id
```shell
docker ps
```

Access the docker image, this should take you into your root user
```shell
docker exec -it  <container_id> /bin/bash 
```

Ensure that Pipenv is installed
```shell
pip install pipenv
```

Access the directory in workspace folder
```shell
cd /workspaces/Brain-Tumour-Classification
```

Initialise Virtual Enviornment
```shell
pipenv shell
```

Install required librarys for virtual enviornment
```shell
pipenv install
```

Navigate to the Flask_app folder
```shell
cd Flask_app
```
Run the following code to start the application:
```shell
python Flask.py
```

## Running Flask Application




 This will create a Url which can be copied and will open up in a tab in your browser.

 

![Loading image](/images/loader.png)

Similarly, you can also enter the following url into your browser:
```shell
localhost:5000
```

This will open a webpage that looks like the following:

![Website](/images/website.png)

By clicking on Choose file users can upload a file from their local machine, and then by clicking the "Predict" button, this will predict the Tumor class using the model.


