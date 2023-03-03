"""
Kaushal Patel
Dylan Miller
Harrison Manka

03/03/2023

Movie Recommendations

In this program we use Apache Spark's Machine Learning Libary to generate movie recommendations
using the MovieLens dataset. The dataset includes a list of 100,000 ratings and 
3,600 tag applications applied to 9,000 movies by 626 users.
"""

# Import necessary classes and functions from PySpark SQL
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, FloatType
from pyspark.sql.functions import lit, col, explode, rand

# Import necessary classes and functions from PySpark Machine Learning
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.recommendation import ALSModel
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Import os to determine if a path exist
import os

# Variables for dividing
divider_top="==================================================================================="
divider_bottom="=================================================================================\n"


def load_movies_dataframes(spark):
    """
    Loads a CSV file containing movie data into a Spark DataFrame.

    Args:
    - spark : A SparkSession object, used to create and configure the Spark DataFrame.

    Returns:
    - Spark Dataframe for the movies dataset

    Raises:
    - None

    Example Usage:
    - load_movies_dataframes(SparkObject)
    """

    return spark.read.csv("./dataset/movies.csv",header=True)

def load_ratings_dataframes(spark):
    """
    Loads a CSV file containing movie rating data into a Spark DataFrame.

    Args:
    - spark : A SparkSession object, used to create and configure the Spark DataFrame.

    Returns:
    - Spark Dataframe for the ratings dataset

    Raises:
    - None

    Example Usage:
    - load_ratings_dataframes(SparkObject)
    """

    return spark.read.csv("./dataset/ratings.csv",header=True)

def combine_dataframes(movies, ratings):
    """
    Combines two DataFrames containing movie and rating data, respectively, into a single DataFrame.
    
    Args:
    - movies : A DataFrame containing movie data, where each row represents a movie and its features
    - ratings : A DataFrame containing rating data, where each row represents 
                a user's rating of a movie.

    Returns:
    - A combined dataframe consisting of the userId, Rating, MovieId, Title, etc

    Raises:
    - None

    Example Usage:
        combine_dataframes(movies_df, ratings_df)
    """

    return ratings.join(movies, ['movieId'], 'left')
    
def calculate_sparsity(ratings):
    """
    Calculates the data sparsity of a ratings dataset.

    Data sparsity is a measure of how much of a ratings dataset is empty or missing. 

    Args:
    - ratings : A DataFrame containing columns 'userId', 'movieId', and 'rating'.

    Returns:
    - The data sparsity of the ratings dataset as a percentage.

    Raises:
    - None

    Example Usage:
    calculate_sparsity(ratings_df)
    """

    # Similar to SQL queries

    # Selects the column labeled rating and returns the number of elements in it
    ratings_count = ratings.select("rating").count()

    # Selects the column labeled userId and returns the distinct number of users
    distinct_users = ratings.select("userId").distinct().count()

    # Selects the column labeled movieId and returns the distince number movies
    distinct_movies = ratings.select("movieId").distinct().count()

    # Calculates the total number of possible user-movie pairs in the ratings data
    total_users_movies = distinct_users * distinct_movies

    # Calculates the sparsity of the ratings data
    data_sparsity = (1.0 - (ratings_count * 1.0) / total_users_movies) * 100

    return data_sparsity


def create_als_model():
    """
    Returns an instance of the Alternating Least Squares (ALS) model for collaborative filtering.
    
    The ALS model is a matrix factorization algorithm that is commonly used for 
    recommendation systems. It factorizes the user-item rating matrix into two low-rank matrices, 
    representing user and item factors, and uses these matrices to predict missing ratings for 
    each user-item pair. 
    
    Args:
    - None
    
    Returns:
    - ALS Model

    Raises:
    - None

    Example Usage:
        create_als_model()
    """

    return ALS(
        # The following are hyperparameters
        userCol="userId", # column name for user IDs in the ratings DataFrame
        itemCol="movieId", # column name for movie IDs in the ratings DataFrame
        ratingCol="rating", # column name for ratings in the ratings DataFrame
        nonnegative=True, # Boolean for enforcing nonnegative constraints on the user and ratings
        implicitPrefs=False, # Boolean for using implicit feedback data instead of explicit ratings
        coldStartStrategy="drop" # Strategy for dealing with data not apart of the training data
    )


def build_param_grid(als_model):
    """
    This function builds a grid of hyperparameters for an ALS (Alternating Least Squares) model 
    based on the input ALS model.

    Args:
    - als_model - ALS model.

    Returns:
    A grid of hyperparameters that will be used to train the ALS model. 
    The hyperparameters are a combination of ranks and regularization parameters.

    Raises:
    - Node

    Example Usage:
        build_param_grid(als_model)
    """

    """
    - Rank: A value that represents the number of latent factors in the ALS model. 
            This parameter is used to determine the dimensionality of the user and item factors.
    - regParam: A regularization parameter that is used to prevent overfitting in the ALS model. 
                This parameter controls the amount of regularization applied to 
                the user and item factors.
    """
    return ParamGridBuilder().addGrid(als_model.rank, [10, 50, 100, 150, 200]) \
                             .addGrid(als_model.regParam, [0.001, 0.01, 0.1]).build()

def build_evaluator():
    """
    Returns a RegressionEvaluator object that evaluates the root mean squared error (rmse) of 
    a regression model's predictions.

    Args:
    - None

    Returns:
    - evaluator : A RegressionEvaluator object that evaluates the rmse of a 
                  regression model's predictions. The label column is set to 
                  "rating" and the prediction column is set to "prediction".

    Raises:
    - None

    Example Usage:
        build_evaluator()
    """

    return RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

def build_cross_validator(als_model, param_grid, evaluator):
    """
    Constructs a CrossValidator object for hyperparameter tuning of an ALS model.

    Args:
    - als_model : The ALS model to be tuned.
    - param_grid : A list of dictionaries, where each dictionary specifies a set of
                     hyperparameters to be tuned.
    - evaluator : The evaluation metric to be used for hyperparameter tuning.

    Returns:
    - (CrossValidator): A CrossValidator object configured for the given
        ALS model, hyperparameter grid, and evaluation metric.

    Raises:
    - None

    Example Usage:
        build_cross_validator(als_model, param_grid, evaluator)
    """

    return CrossValidator(estimator=als_model, estimatorParamMaps=param_grid, 
                          evaluator=evaluator, numFolds=10)

def train_models(cross_validation, evaluator, training_dataset, testing_dataset):
    """
    Trains a model using cross-validation on the training dataset and evaluates it 
    using the provided evaluator on the testing dataset.
    
    Args:
    - cross_validation : A cross-validator object that will be used to 
                         train the model on the training dataset.
    - evaluator : A regression evaluator object that will be used to 
                  evaluate the model on the testing dataset.
    - training_dataset : A Spark DataFrame containing the training dataset.
    - testing_dataset : A Spark DataFrame containing the testing dataset.
    
    Returns:
    - best_model (PipelineModel): A trained pipeline model representing the best 
                                model found by the cross-validator on the training dataset.

    Raises:
    - None

    Example Usage:
        train_models(CrossValidation, RegressionEvaluator, training_df, testing_df)
    """

    """
    Fits a cross-validator to the training dataset using the fit() method of the 
    cross_validation object. This creates a model that can be used to make predictions on new data.
    """
    model = cross_validation.fit(training_dataset)

    # Extracts the best model from the cross-validated model using the bestModel attribute
    best_model = model.bestModel
    
    # Uses the best model to make predictions on the testing dataset using the transform() method
    predictions_from_test = best_model.transform(testing_dataset)

    # Calculates the Root Mean Square Error which is an accuracy measure of regression model
    RMSE = evaluator.evaluate(predictions_from_test)

    print(divider_top)
    print(RMSE)
    print(divider_bottom)

    return best_model


def print_recommendations(movies, ratings, best_model, user, print_type, name):
    """
    Prints recommended movies for a given user

    This function takes in three DataFrames representing users, movies, and ratings respectively. 
    It uses the best_model to gather the recommendations. Then, using DataFrame tools, we combine
    and cross corelate the user we are generating recomendations for with their recommendations
    and the titles of the recommendations.

    Args:
    - movies : DataFrame containing movie data with columns 'movieId' and 'title'.
    - ratings : DataFrame containing user movie ratings data with columns 'userId', 
                    'movieId', and 'rating'.
    - best_model : Trained ALS model used to generate recommendations.
    - user : ID of the user to generate recommendations for.
    - print_type : Type of recommendations to print. "0" for top 30, "1" for top 30 shuffled.
    - name : Name of the user to address in the print statement.

    Returns:
    - None

    Raises:
    - None

    Example Usage:
        print_recommendations(movies_df, ratings_df, trained_model, user_df, print_type, user_name)
    """
    
    # Find the selected user from the ratings dataframe
    user_subset = ratings.filter(col("userId") == user)
    user_subset.show()

    # Generate 1000 recommendations for the selected use
    recommendations = best_model.recommendForUserSubset(user_subset, 1000)

    # Explode the recommendations column to create separate rows for each recommendation
    exploded_recommendations = (
        recommendations
        .select(col("userId"), explode(col("recommendations")).alias("recommendation"))
        .select(col("userId"), col("recommendation.movieId"), col("recommendation.rating"))
    )

    # Join the exploded recommendations with the movies DataFrame on the movieId column
    recommendations_with_titles = (
        exploded_recommendations
        .join(movies, exploded_recommendations.movieId == movies.movieId)
        .select(movies.title, exploded_recommendations.rating.alias("predicted_rating"))
    )

    # If the user presses 0, then we display the top 30 recommendations, else display 30 shuffled
    if print_type == "0":
        print(divider_top)
        print(f'{name} here are your top 30 recommendations\n')
        recommendations_with_titles.show(30, truncate=False)
        print(divider_bottom)
    elif print_type == "1":
        print(divider_top)
        print(f'{name} here are your top 30 shuffled recommendations\n')
        recommendations_with_titles = recommendations_with_titles.orderBy(rand())
        recommendations_with_titles.show(30, truncate=False)
        print(divider_bottom)

    

def user_recommendations(users, movies, ratings):
    """
    Recommend Movies to User

    This function takes in three DataFrames representing users, movies, and ratings respectively. 
    It uses a pre-trained ALS model to recommend movies to a user based on their name, and the 
    type of recommendation they prefer. The recommended movies can be in the form of either a 
    list of top recommendations or a shuffled list.

    Args:
    - users: DataFrame containing user data with columns 'name' and 'id'.
    - movies: DataFrame containing movie data with columns 'movieId' and 'title'.
    - ratings: DataFrame containing user movie ratings data with
               columns 'userId', 'movieId', and 'rating'.

    Returns:
    - None

    Raises:
    - None

    Example Usage:
        user_recommendations(users_df, movies_df, ratings_df)
    """

    # Load the trained model from the file directory
    best_model = ALSModel.load("./model/")

    # Get user input from the command-line to get recommendations for the user
    while True:
        name = input("Please enter your name (or 'q' to quit): ")
        full_or_shuffled = input("""Please enter 0 for your top recommendations or
                                 enter 1 for shuffled recommendations: """)
        
        if name == 'q':
            break

        # Find the user id of the user from their name
        user_id = users.filter(col("name") == name).select(col("id")).first()[0]

        # Function call to print the recommendations
        print_recommendations(movies, ratings, best_model, user_id, full_or_shuffled, name)


def main():
    '''
    This function is the main entry point of the program. It loads movie and rating data
    into dataframes and performs data cleaning operations such as converting data types.
    If no ML model is trained, it trains a model and saves it. If a model is already saved,
    it loads the model and uses it to make recommendations for users.
    '''

    # Create a SparkSession Object
    spark = SparkSession.builder.appName('Movie Recommender').getOrCreate()

    # Read the movies data into a dataframe called movies
    movies = load_movies_dataframes(spark)

    # Read the ratings data into a dataframe called ratings
    ratings = load_ratings_dataframes(spark)

    # Convert the userIds, movieId, and rating to interger types instead of strings
    ratings = ratings.withColumn("userId",ratings.userId.cast(IntegerType()))
    ratings = ratings.withColumn("movieId",ratings.movieId.cast(IntegerType()))
    ratings = ratings.withColumn("rating",ratings.rating.cast(FloatType()))

    # Convert the movieIds to integer types instead of strings
    movies = movies.withColumn("movieId",movies.movieId.cast(IntegerType()))

    '''
    If we do not have a ML model trained and generated, then we will proceed to train and make model
    Else we will use the pretrained model to start making recommendations
    '''
    if not os.listdir('./model/'):
        # Create our training and testing datasets
        (training_dataset, testing_dataset) = ratings.randomSplit([0.7, 0.3], seed=9119)

        # Create the ALS model
        als_model = create_als_model()

        # Add hyperparameters and their respective values to param_grid
        param_grid = build_param_grid(als_model)

        # Defining the evaluator for the model using RMSE (Root Mean Square Error)
        evaluator = build_evaluator()

        # Build cross validation using CrossValidator
        cross_validation = build_cross_validator(als_model, param_grid, evaluator)

        # Print the number of models that will tested based on the ParamBuilder's result
        print("Number of models to be tested: ", len(param_grid))

        # Fit the best model and evaluate predictions
        best_model = train_models(cross_validation, evaluator, training_dataset, testing_dataset)

        # Save the best model
        best_model.write().overwrite().save("./model/")
    else:
        print(divider_top)
        # Show the ratings dataframe
        ratings.show()
        print(divider_bottom)

        print(divider_top)
        # Show the movies dataframe
        movies.show()
        print(divider_bottom)

        print(divider_top)
        # Combine the dataframes
        combined_data = combine_dataframes(movies, ratings)

        # Show the combined dataframes
        combined_data.show()
        print(divider_bottom)


        print(divider_top)
        # Calculate and print sparsity
        data_sparsity = calculate_sparsity(ratings)
        print(f'The sparsity level for the dataframe is: {data_sparsity}')
        print(divider_bottom)


        # Read the users data into a dataframe called users
        users = spark.read.csv("./dataset/users.csv",header=True)

        # Convert the string id values to integers
        users = users.withColumn("id",users.id.cast(IntegerType()))
        
        # Generate recommendations for users using the best ML model trained
        user_recommendations(users, movies, ratings)





if __name__ == "__main__":
    main()