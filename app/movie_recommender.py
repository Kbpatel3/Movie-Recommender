from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, FloatType
from pyspark.sql.functions import lit, col, explode

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.recommendation import ALSModel
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator



'''Combine two functions below'''
def load_movies_dataframes(spark):
    return spark.read.csv("./dataset/movies.csv",header=True) # Was ../

def load_ratings_dataframes(spark):
    return spark.read.csv("./dataset/ratings.csv",header=True)

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


def create_als_model():
    return ALS(
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        nonnegative=True,
        implicitPrefs=False,
        coldStartStrategy="drop"
    )


def build_param_grid(als_model):
    return ParamGridBuilder().addGrid(als_model.rank, [10, 50, 100, 150]).addGrid(als_model.regParam, [0.01, 0.05, 0.1, 0.15]).build()

def build_evaluator():
    return RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

def build_cross_validator(als_model, param_grid, evaluator):
    return CrossValidator(estimator=als_model, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)

def train_models(cross_validation, evaluator, training_dataset, testing_dataset):
    #Fit cross validator to the 'train' dataset
    model = cross_validation.fit(training_dataset)#Extract best model from the cv model above
    best_model = model.bestModel # View the predictions
    predictions_from_test = best_model.transform(testing_dataset)
    RMSE = evaluator.evaluate(predictions_from_test)
    print(RMSE)

    return best_model

def make_recommendations():
    best_model = ALSModel.load("./model/")

    # Generate n Recommendations for all users
    recommendations = best_model.recommendForAllUsers(5)
    recommendations.show()
    return recommendations

def print_recommendations(movies, recommendations):
    recommendations = recommendations.withColumn("rec_exp", explode("recommendations")).select('userId', col("rec_exp.movieId"), col("rec_exp.rating"))
    
    recommendations.limit(10).show()

    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>")
    recommendations.join(movies, on='movieId').filter('userId = 611').show()

def main():
    # Instantiating a Spark session
    spark = SparkSession.builder.appName('Movie Recommender').getOrCreate()

    # Read the movies data into a dataframe called movies
    movies = load_movies_dataframes(spark)

    # Read the ratings data into a dataframe called ratings
    ratings = load_ratings_dataframes(spark)

    # TODO
    # Convert the userIds, movieId, and rating to interger types instead of strings
    ratings = ratings.withColumn("userId",ratings.userId.cast(IntegerType()))
    ratings = ratings.withColumn("movieId",ratings.movieId.cast(IntegerType()))
    ratings = ratings.withColumn("rating",ratings.rating.cast(FloatType()))

    # Convert the movieIds to integer types instead of strings
    movies = movies.withColumn("movieId",movies.movieId.cast(IntegerType()))

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

    # Create the ALS model
    als_model = create_als_model()

    '''
    Generate the 'Best' model
    '''

    # Add hyperparameters and their respective values to param_grid
    param_grid = build_param_grid(als_model)

    # Defining the evaluator for the model using RMSE (Root Mean Square Error)
    evaluator = build_evaluator()

    # Build cross validation using CrossValidator
    cross_validation = build_cross_validator(als_model, param_grid, evaluator)


    print("Number of models to be tested: ", len(param_grid))

    # Fit the best model and evaluate predictions
    best_model = train_models(cross_validation, evaluator, training_dataset, testing_dataset)

    # Save the best model
    best_model.write().overwrite().save("./model/")

    print("**Best Model**")# Print "Rank"
    print("  Rank:", best_model._java_obj.parent().getRank())# Print "MaxIter"
    print("  MaxIter:", best_model._java_obj.parent().getMaxIter())# Print "RegParam"
    print("  RegParam:", best_model._java_obj.parent().getRegParam())

    recommendations = make_recommendations()
    print_recommendations(movies, recommendations)




if __name__ == "__main__":
    main()