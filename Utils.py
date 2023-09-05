

import datetime
import json


def get_time_based_greeting():
    current_time = datetime.datetime.now().time()
    
    if current_time < datetime.time(12, 0):
        greeting = "Good morning!"
    elif current_time < datetime.time(18, 0):
        greeting = "Good afternoon!"
    else:
        greeting = "Good evening!"
    
    return greeting

def extract_text_values(json_string):
    try:
        data = json.loads(json_string)
        text_values = [item for item in data if isinstance(item, str)]
        return text_values
    except json.JSONDecodeError:
        return []

def get_parse_jd_details(json_string):
    table_rows = []
    for key, values in json_string.items():
        column_header = key.replace('_', ' ').title()  # Capitalize words and replace underscores
        value = values[0]
        value = test = ' - '.join(str(x) for x in value.values())
        table_rows.append([column_header, value])
    return table_rows

