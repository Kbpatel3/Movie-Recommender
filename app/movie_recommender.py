from pyspark.sql import SparkSession





def load_movies_dataframes(spark):
    return spark.read.csv("../dataset/movies.csv",header=True)

def load_ratings_dataframes(spark):
    return spark.read.csv("../dataset/ratings.csv",header=True)

def combine_dataframes(movies, ratings):
    return ratings.join(movies, ['movieId'], 'left')
    
def calculate_sparsity(ratings):
    ratings_count = ratings.select("rating").count()

    distinct_users = ratings.select("userId").distinct().count()
    distinct_movies = ratings.select("movieId").distinct().count()

    total_users_movies = distinct_users * distinct_movies

    data_sparsity = (1.0 - (ratings_count * 1.0) / total_users_movies) * 100

    print(f'The sparsity level for the dataframe is: {data_sparsity}')


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

    # Calculate sparsity
    calculate_sparsity(ratings)




if __name__ == "__main__":
    main()