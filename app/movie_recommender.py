from pyspark.sql import SparkSession
from pyspark.sql.functions import lit



'''Combine two functions below'''
def load_movies_dataframes(spark):
    return spark.read.csv("../dataset/movies.csv",header=True)

def load_ratings_dataframes(spark):
    return spark.read.csv("../dataset/ratings.csv",header=True)

def combine_dataframes(movies, ratings):
    return ratings.join(movies, ['movieId'], 'left')
    
def calculate_sparsity(ratings):
    # Similar to SQL queries
    ratings_count = ratings.select("rating").count()

    distinct_users = ratings.select("userId").distinct().count()
    distinct_movies = ratings.select("movieId").distinct().count()

    total_users_movies = distinct_users * distinct_movies

    data_sparsity = (1.0 - (ratings_count * 1.0) / total_users_movies) * 100

    return data_sparsity


def ratings_to_binary(ratings):
    ratings = ratings.withColumn('has_watched', lit(1))

    userIds = ratings.select("userId").distinct()
    movieIds = ratings.select("movieId").distinct()

    new_ratings = userIds.crossJoin(movieIds).join(ratings, ['userId', 'movieId'], "left")
    new_ratings = new_ratings.select(['userId', 'movieId', 'has_watched']).fillna(0)

    return new_ratings


def main():
    # Instantiating a Spark session
    spark = SparkSession.builder.appName('Movie Recommender').getOrCreate()

    # Read the movies data into a dataframe called movies
    movies = load_movies_dataframes(spark)

    # Read the ratings data into a dataframe called ratings
    ratings = load_ratings_dataframes(spark)

    # Show the ratings dataframe
    ratings.show()

    # Show the movies dataframe
    movies.show()

    # Combine the dataframes
    combined_data = combine_dataframes(movies, ratings)

    # Show the combined dataframes
    combined_data.show()

    # Calculate and print sparsity
    data_sparsity = calculate_sparsity(ratings)
    print(f'The sparsity level for the dataframe is: {data_sparsity}')

    # Create our training and testing datasets
    (training_dataset, testing_dataset) = ratings.randomSplit([0.8, 0.2], seed=2023)

    # Convert ratings into binary format where 0 means not watched and 1 means watched
    new_ratings = ratings_to_binary(ratings)

    # Show the newly converted dataframe
    new_ratings.show()



if __name__ == "__main__":
    main()