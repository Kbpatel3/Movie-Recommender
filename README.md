Movie Recommender

Movie Recommender is a Python program that recommends movies to users based on their ratings history. The program uses collaborative filtering, a popular algorithm in recommendation systems, 
to find similar users and recommend movies that the user has not seen before. We are using Apache Spark MLlib and it's ALS model.

Installation

To use Movie Recommender, follow these steps:

    1) Clone this repository to your local machine.
    2) Install Apache Spark and set the path enviornment variables
      a) I recommend the following video tutorial - https://youtu.be/LcRxQjTdD1o
    3) Install the required packages by running the following:
      a) pip install pyspark
      b) pip install pyspark[sql]
    4) Unzip the zipped file (locacted in the root directory)
      a) This file contains a pretrained model and a small educational dataset from MovieLens
      b) You can replace the ratings and movies dataset, but in the users.csv, just add users you'd like to recommend for and assign the next available userId

Usage

Upon running the program, you will be prompted to enter a user ID. This ID corresponds to a user in the MovieLens dataset. After entering the ID, the program will recommend 30 movies to the user based on their ratings history.

You can also modify the number of movies to recommend by changing the num_recommendations variable in main.py.
Contributing

Contributions are welcome! If you notice any bugs or have ideas for improving the program, feel free to open an issue or submit a pull request.
License

This project is licensed under the MIT License. See the LICENSE file for more information.
