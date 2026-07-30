import urllib
import pandas as pd
import pyodbc
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sqlalchemy import create_engine

# Download the VADER lexicon for sentiment analysis if not already present.
nltk.download('vader_lexicon')


# Define a function to fetch data from a SQL database using a SQL query
def fetch_data_from_sql():
    # Define connection details
    conn_str = (
        "Driver={ODBC Driver 17 for SQL Server};"
        "Server=localhost;"
        "Database=PortfolioProject_MarketingAnalytics;"
        "Trusted_Connection=yes;"
    )

    # Wrap connection for SQLAlchemy (fixes the pandas UserWarning)
    params = urllib.parse.quote_plus(conn_str)
    engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

    # FIXED: Table name matches SSMS ('dbo.customer_reviews')
    query = "SELECT ReviewID, CustomerID, ProductID, ReviewDate, Rating, ReviewText FROM dbo.customer_reviews"

    # Execute the query and fetch data into DataFrame
    df = pd.read_sql(query, engine)

    return df


# Fetch the customer reviews data
customer_reviews_df = fetch_data_from_sql()

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()


def calculate_sentiment(review):
    # Ensure review is treated as string to avoid errors with empty text
    sentiment = sia.polarity_scores(str(review))
    return sentiment['compound']


def categorize_sentiment(score, rating):
    if score > 0.05:
        if rating >= 4:
            return 'Positive'
        elif rating == 3:
            return 'Mixed Positive'
        else:
            return 'Mixed Negative'
    elif score < -0.05:
        if rating <= 2:
            return 'Negative'
        elif rating == 3:
            return 'Mixed Negative'
        else:
            return 'Mixed Positive'
    else:
        if rating >= 4:
            return 'Positive'
        elif rating <= 2:
            return 'Negative'
        else:
            return 'Neutral'


def sentiment_bucket(score):
    if score >= 0.5:
        return '0.5 to 1.0'
    elif 0.0 <= score < 0.5:
        return '0.0 to 0.49'
    elif -0.5 <= score < 0.0:
        return '-0.49 to 0.0'
    else:
        return '-1.0 to -0.5'


# Apply sentiment calculations
customer_reviews_df['SentimentScore'] = customer_reviews_df['ReviewText'].apply(
    calculate_sentiment
)
customer_reviews_df['SentimentCategory'] = customer_reviews_df.apply(
    lambda row: categorize_sentiment(row['SentimentScore'], row['Rating']),
    axis=1,
)
customer_reviews_df['SentimentBucket'] = customer_reviews_df[
    'SentimentScore'
].apply(sentiment_bucket)

# Print first few rows
print(customer_reviews_df.head())

# Save output CSV
customer_reviews_df.to_csv(
    'fact_customer_reviews_with_sentiment.csv', index=False
)