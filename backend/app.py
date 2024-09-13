from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from bson.json_util import dumps
from datetime import datetime
import logging
from urllib.parse import quote_plus

app = Flask(__name__)
CORS(app)

# MongoDB Connection
username = quote_plus('thissya129')
password = quote_plus('Thissya129')
uri = f'mongodb+srv://{username}:{password}@energymanagement.zn8ue.mongodb.net/?retryWrites=true&w=majority&appName=EnergyManagement'

try:
    client = MongoClient(uri)
    db = client['Energy_Management']
    logging.info("Connected to MongoDB")
except Exception as e:
    logging.error(f"Error connecting to MongoDB: {str(e)}")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Mapping of department names to collection field keys
department_map = {
    "east campus": "eastCampus",
    "mba": "mbaMca",
    "mca": "mbaMca",
    "civil": "civil",
    "mechanical": "mech",
    "automobile": "auto"
}

@app.route('/form', methods=['POST'])
def chatbot():
    data = request.get_json()
    logger.debug(f"Received data: {data}")
    user_query = data.get('query', '').lower()

    collection_name = get_collection_name(user_query)
    if not collection_name:
        return jsonify({"response": "I can provide data on solar power and electricity. Please ask about either!"})

    date_query = extract_date(user_query)
    logger.debug(f"Extracted date: {date_query}")
    department_query = extract_department(user_query)
    logger.debug(f"Extracted department: {department_query}")

    try:
        query = build_query(date_query, department_query)
        logger.debug(f"MongoDB Query: {query}")

        response = db[collection_name].find_one(query, {"_id": 0})
        logger.debug(f"MongoDB Response: {response}")

        if not response:
            return jsonify({"response": "No data found for the specified criteria."})

        if department_query:
            block_data = {department_query: response.get(department_query, "No data for this block")}
        else:
            block_data = {key: value for key, value in response.items() if key != "date"}

        logger.debug(f"Final Response Data: {block_data}")
        return jsonify({"response": block_data})

    except Exception as e:
        logger.error(f"Error fetching data from the database: {str(e)}")
        return jsonify({"response": f"Error fetching data from the database: {str(e)}"})

def get_collection_name(query):
    if "solar" in query:
        return 'solar_energy'
    elif "electricity" in query:
        return 'electrical_energy'
    return None

def extract_date(query):
    if "on" in query:
        date_part = query.split("on")[1].strip()
        try:
            date_object = datetime.strptime(date_part, "%d.%m.%Y")
            return date_object
        except ValueError:
            logger.error(f"Invalid date format: {date_part}")
            return None
    return None

def extract_department(query):
    for key, value in department_map.items():
        if key in query:
            return value
    return None

def build_query(date_query, department_query):
    query = {}

    if date_query:
        start_date = datetime.combine(date_query, datetime.min.time())
        end_date = datetime.combine(date_query, datetime.max.time())
        query["Date"] = {"$gte": start_date, "$lte": end_date}
        logger.debug(f"Date Query: {query['Date']}")

    if department_query:
        query[department_query] = {"$exists": True}

    return query

if __name__ == '__main__':
    app.run(debug=True)
