README : How to run the project ?


git clone https://github.com/JLBT10/books_project.git

Go on ---> https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/books?authuser=0 

Download 'goodreads_book_authors.json.gz' and save it in the dataset folder in books_project

STARTING DOCKER

Method 1 # Uncomment the CMD["python3","train.py"]
1) Build the docker images
cd app
docker -t build books .

Run the docker
docker run -it books # model automatically start training 3 models decision tree, xgboost and random forest.



Method 2 : comment the CMD["python3","train.py"] & mapping with volume on your local pc

1) Build the docker images
cd ./app
docker -t build books .

Run the docker
docker run -it -v /host/path:/app books

*/host/path : path towards the books_project directory (.../books_project)

With that method you access the container and the docker is mapped on a books_project on your local pc.
You can then easily check the results of training by opening the runs directory on your pc.
