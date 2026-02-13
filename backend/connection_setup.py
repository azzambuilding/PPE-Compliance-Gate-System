import os
from dotenv import load_dotenv #Loads environment variables from .env file
import psycopg2 #direct postgreSQL connection
from sqalchemy import create_engine #allows user to write python classes and objects rather than
#raw SQL. TLDR: Translates python code to SQL


load_dotenv()

db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_name = os.getenv('DB_NAME')
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')


conn = psycopg2.connect(
    database = db_name,
    user = db_user,
    password = db_password,
    host = db_host,
    port = db_port,
)