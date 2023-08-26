# How to run the project!

Clone the repository to your local

```
git clone https://github.com/JLBT10/books_project.git

```

**STARTING DOCKER**

### Method 1(Declarative)
1. Open the Dockerfile and Uncomment the CMD["python3","train.py"] in line 19

2. Build the docker images

```
docker build -t books .

```
3. Run the image in a container by running the command below (model automatically start training 3 models decision tree, xgboost and random forest.)

```
docker run -it books 

```

### Method 2 (Iterative)

1. Open the Dockerfile and comment the CMD["python3","train.py"] then map with volume on your local pc

2. Build the docker images

```

docker build -t books .

```

3. Run the image in a container

```
docker run -it -v /host/path:/app books

```

*/host/path : path is the books_project directory usually like this for a windows machine (c/users/.../books_project)<br>

4. Once you have started the container, run the train.py script like this and observe the outputs till the end, then exit the container.


```
python3 train.py

```

With method 2 you have access to the container and the docker is mapped to the books_project directory on your local pc.<br>
**You can then easily check the results of training by opening the runs directory on your pc**.
