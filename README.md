# How to run the project!

Make a new folder in your local and clone the repository with this link.

```
git clone https://github.com/JLBT10/books_project.git

```
Get in the books_project directory

```
cd books_project

```

**STARTING DOCKER**

### Method 1 (Iterative)

1. Open the Dockerfile and comment the CMD["python3","train.py"] then map with volume on your local pc

2. Build the docker images
```

docker build . -t books -f src/Dockerfile

```
3. Run the image in a container

```
docker run -it -v "/host/path":/app books

```

*/host/path : path is the books_project directory usually like this for a windows machine (c/users/.../books_project)<br>

4. Once you have started the container, run the train.py script like this and observe the outputs till the end, then exit the container.


```
python3 train.py > ./../project_report/results.txt

```
From that command, All outputs of the train.py will be directed to results.txt inside project_report folder.

With method 1 you have access to the container and the docker is mapped to the books_project directory on your local pc.<br>
**You can then easily check the results of training by opening the runs directory on your pc**.

### Method 2 (Declarative)
1. Open the Dockerfile and Uncomment the CMD["python3","train.py"] in line 19

2. Build the docker images

```
docker build . -t books -f src/Dockerfile

```
3. Run the image in a container by running the command below (model automatically start training 3 models decision tree, xgboost and random forest.)

```
docker run -it books 

```
# NB
The data analysis notebook is found in the project_report folder 