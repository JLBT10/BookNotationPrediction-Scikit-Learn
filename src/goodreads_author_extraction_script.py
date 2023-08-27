import json

info_authors = pd.read_json('./datasets/goodreads_book_authors.json', lines=True)
lookup_authors = dict(zip(info_authors['name'], info_authors['average_rating']))
with open('./../datasets/dict_goodreads_book_authors.json', "w") as json_file:
    json.dump(lookup_authors, json_file)