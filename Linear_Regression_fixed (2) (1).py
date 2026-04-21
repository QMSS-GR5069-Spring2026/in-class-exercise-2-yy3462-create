# Databricks notebook source
# MAGIC %md
# MAGIC # Regression: Predicting Rental Price
# MAGIC
# MAGIC In this notebook, we will use the dataset we cleansed in the previous lab to predict Airbnb rental prices in San Francisco.
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Use the SparkML API to build a linear regression model
# MAGIC  - Identify the differences between estimators and transformers

# COMMAND ----------

from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql.functions import col

# COMMAND ----------

airbnbDF = spark.read.csv('/Volumes/gr5069/raw/airbnb/airbnb-cleaned-mlflow.csv',header=True)
airbnbDF.count()

# COMMAND ----------

airbnbDF.write.mode('overwrite').parquet('/Volumes/gr5069/raw/airbnb/listings_clean')

# COMMAND ----------

display(airbnbDF)

# COMMAND ----------

filePath = "/Volumes/gr5069/raw/airbnb/listings_clean"
airbnbDF = spark.read.parquet(filePath)

airbnbDF = airbnbDF.withColumn("bedrooms", airbnbDF["bedrooms"].cast(DoubleType())).withColumn("price", airbnbDF["price"].cast(DoubleType()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train/Test Split
# MAGIC
# MAGIC When we are building ML models, we don't want to look at our test data (why is that?). 
# MAGIC
# MAGIC Let's keep 80% for the training set and set aside 20% of our data for the test set. We will use the `randomSplit` method [Python](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.randomSplit)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.sql.Dataset).
# MAGIC
# MAGIC **Question**: Why is it necessary to set a seed? What happens if I change my cluster configuration?

# COMMAND ----------

(trainDF, testDF) = airbnbDF.randomSplit([.8, .2], seed=42)
print(trainDF.count())

# COMMAND ----------

# MAGIC %md
# MAGIC Let's change the # of partitions (to simulate a different cluster configuration), and see if we get the same number of data points in our training set. 

# COMMAND ----------

(trainRepartitionDF, testRepartitionDF) = (airbnbDF
                                           .repartition(24)
                                           .randomSplit([.8, .2], seed=42))

print(trainRepartitionDF.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Linear Regression
# MAGIC
# MAGIC We are going to build a very simple model predicting `price` just given the number of `bedrooms`.
# MAGIC
# MAGIC **Question**: What are some assumptions of the linear regression model?

# COMMAND ----------

display(trainDF.select("price", "bedrooms"))

# COMMAND ----------

display(trainDF.select("price", "bedrooms").summary())

# COMMAND ----------

display(trainDF)

# COMMAND ----------

# MAGIC %md
# MAGIC There do appear some outliers in our dataset for the price ($9,999 a night??). Just keep this in mind when we are building our models :).
# MAGIC
# MAGIC We will use `LinearRegression` to build our first model [Python](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.regression.LinearRegression)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.regression.LinearRegression).

# COMMAND ----------

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
print("sklearn LinearRegression model created (features: bedrooms, label: price)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Explain Params
# MAGIC
# MAGIC When you are unsure of the defaults or what a parameter does, you can call `.explainParams()`.

# COMMAND ----------

# sklearn equivalent of explainParams()
params = lr.get_params()
for param, value in params.items():
    print(f"  {param}: {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Vector Assembler
# MAGIC
# MAGIC What went wrong? Turns out that the Linear Regression **estimator** (`.fit()`) expected a column of Vector type as input.
# MAGIC
# MAGIC We can easily get the values from the `bedrooms` column into a single vector using `VectorAssembler` [Python]((https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.VectorAssembler)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.feature.VectorAssembler). VectorAssembler is an example of a **transformer**. Transformers take in a DataFrame, and return a new DataFrame with one or more columns appended to it. They do not learn from your data, but apply rule based transformations.

# COMMAND ----------

# from pyspark.ml.feature import VectorAssembler

# vecAssembler = VectorAssembler(inputCols = ["bedrooms"], outputCol = "features")

# vecTrainDF = vecAssembler.transform(trainDF)

# lr = LinearRegression(featuresCol = "features", labelCol = "price")
# lrModel = lr.fit(vecTrainDF)

# COMMAND ----------

# Filter out rows with null or NaN values in the "price" column
trainDF = trainDF.filter(col("price").isNotNull())

# Convert to pandas for sklearn (only the columns we need for fitting)
trainPDF = trainDF.select("bedrooms", "price").toPandas().dropna()

# Fit the sklearn linear regression model
lr.fit(trainPDF[["bedrooms"]], trainPDF["price"])

print(f"Coefficient (bedrooms): {lr.coef_[0]:.4f}")
print(f"Intercept: {lr.intercept_:.4f}")

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Apply model to test set

# COMMAND ----------

# Convert test set to pandas, generate predictions, convert back to Spark DataFrame
testPDF = testDF.toPandas()
testPDF["prediction"] = lr.predict(testPDF[["bedrooms"]].fillna(0))

predDF = spark.createDataFrame(testPDF)

predDF.select("bedrooms", "price", "prediction").show()

# COMMAND ----------

predDF_final = predDF.select('host_total_listings_count','neighbourhood_cleansed','zipcode','latitude','longitude','property_type','room_type','accommodates','bathrooms','bedrooms','beds','bed_type','minimum_nights','number_of_reviews','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','price','prediction')

# COMMAND ----------

display(predDF_final)

# COMMAND ----------

predDF_final.write.mode('overwrite').csv('/Volumes/yy3462/raw/airbnb/predDF.csv')
