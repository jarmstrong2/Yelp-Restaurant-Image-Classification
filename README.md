# Yelp Restaurant Image Classification
https://www.kaggle.com/c/yelp-restaurant-photo-classification
###### Note: Will not include data due to copyright, but will include model and training code. Also, code not fully functional without data.

Competition Description
-----------------------
In this competition, you are given photos that belong to a business and asked to predict the business attributes. There are 9 different attributes in this problem:

ID | Classification
--- | ---
0 | good_for_lunch
1 | good_for_dinner
2 | takes_reservations
3 | outdoor_seating
4 | restaurant_is_expensive
5 | has_alcohol
6 | has_table_service
7 | ambience_is_classy
8 | good_for_kids

These labels are annotated by the Yelp community. Your task is to predict these labels purely from the business photos uploaded by users. 

Competition Solution
-----------------------
As with many other submissions, my solution began with using a pre-trained image classification network. 
This network was the inception network: https://github.com/tensorflow/models/tree/master/inception. 
Using inception net and removing the last couple fully connected layers, I was able to collect vectors that 
would fairly accurately differentiate between features in restaurant images.

With the collection of image vectors, I decided to focus on building an RNN model using LSTMS and a limited number of
timesteps. For each timestep we would randomly choose, from usually several hundreds of images, a vector from a specific 
restaurant image, and use it as input. The rationale being, that instead of using the entire image set (or image vectors) from each restaurant, we use
a randomly selected handful to classify the type of dining experience.

Finally, I took all output vectors from each timestep of the RNN and averaged the entire set. This average vector is then used as input
to 10 different networks (9 classification IDs, and 1 ID for a restaurant with no ID). Each of these networks then decides the classification
of the restaurant based on the probability that this series of image vectors correlates with being good for lunch, good for dinner, etc.