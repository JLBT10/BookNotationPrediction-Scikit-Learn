<p style="text-align: center;"><strong>Book Reviews Project Report.</strong></p>
<div style="text-align: center;">
<p style="text-align: center;"><strong>DATA SCIENCETECH INSTITUTE.</strong></p>
<img
src="images/logo_dsti.png"
width ='300'
height = '170'
>
</div>

# Authors
## [Data ScienceTech Institute - A22 Cohort Group:15]
- Jean-Luc Boa Thiemele
-  Deogratias Allakonon
-  Clement Adebisi

## <p style="text-align: center;">Understanding the data - DATA EXPLORATION.</p>

__Title__ <br>
The title columns contains text and has different entries , a simple encoding will not be of any use for training and prediction. so we will drop that columns for training. but before doing that we will check for doubloons so that they can be handled properly; moreover, the titles are written in different language which will render analysis complicated using nltk model. <br>


__Authors__ <br>
A book can be written by many authors. There are too many authors to do a One Hot Encoding, and if we isolate the authors(by exploding the columns) we will have  a lot of duplicated information which may not be good for the training. So in order to exploit that columns,we will add the average notation of authors as a new variable instead of authors name. The goodreads website has a json file __[goodreads_book_authors.json](https://drive.google.com/uc?id=19cdwyXwfXx_HDIgxXaHzH0mrx8nMyLvC)__ that contains the average notation of all authors that has ever been rated on goodreads and we will use this file to add information which we presume could be more beneficial in our model. <br>


__Average rating__ <br>
Average rating is the element to predict. Using that variable to engineer feature might lead to overfitting during training (we will have good result during training but bad  not as good result during testing). we made the choice not to use it. <br>


__Isbn,Isbn3,Publisher__ <br>
The publisher columns is structured in the same manner as the authors columns. Moreover we can see that Scholastic Inc. and Scholastic is the same publisher, but if we encode Scholastic Inc. will be considered different from Scholastic which is a huge problem, does the column contains other cases like this one ? probably yes but we will verify this in the Analysis Part.

We could use ISBN or ISBN13 in order to get the digit of the publisher as an ISBN.
For example : from ISBN13 : 978-0-545-01022-1 we can infer that 
"978" is the ISBN-13 prefix, indicating that it's an ISBN for a book.
"0" is the group identifier, representing the country or language area of the publisher.
"545" is the publisher identifier and it is unique for every publisher so we can take is as a form of ID to replace name of publishers which is a string data.
"01022" is the publication element identifies the specific publication
"1" is the check digit, calculated based on the preceding digits to help detect errors.

Unfortunately, the publisher identifier and the the publication element(title) can have varying length which makes it difficult to infer with confidence.
In our dataset that information is not explicitly given to us and we can not deduce it with confidence. for example is the publisher identifier composed of 1, 2, 3 , or 4 digits ? we can't be sure of that and this could be another layer of complexity for our model. <br>


__language_code, publication_date__ <br>
We will study the composition of the language_code
We will separate the month and the year in order to study their relation with average notation and authors notation <br>

__num_pages, ratings_count, text_reviews_count__ <br>
We will definitely use those for our model, one thing we will verify during analysis is the correlation between those variables. Do books get a better rating when there is huge ratings_count or/and text reviews or a lot of pages?
We will remove whitespace before a file label by renaming it as such ' num_pages' in  'num_pages'.

## <p style="text-align: center;">Cleaning the data</p>

From our trainining data, we could easily deduce the following by describing the dataset:
* The average rating ranges from 0 to 5
* We can see two issues with num_pages, the min is equal 0 and the maximum of pages is equal to 6576 (both are does not look normal for the common book)
* ratings_count : We can't have a rating_count of 0 and have an average_rating different from 0, there were some records of this kind of data in our dataset
* Over half of the books in our dataset have text_review_counts of less than 50, and even the highest rated books have far less significant number of text_review_counts<br>We may infer that this column might be less useful for our model.
* We can deduce from the title columns that those books with a lot of pages are actually collections or volumes.
* So it may be more logical to drop them since our concerns is to predict the average_rating of a single book, more over it could introduce redundancy in our training.
* Our box plot suggests that we have a lot of outliers that are mostly collections, novels and boxes. A good trade off could be for us to keep all books that have pages > 1000 
* There are no missing values(no Na) in our dataset
* The publication date is an object dtype but we will extract the year and the month then convert it to integer.
* page at 0 will be replaced by the mean of pages.



