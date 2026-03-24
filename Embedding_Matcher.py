import pandas as pd
import numpy as np
import Mark_Attendance_Camera
from datetime import datetime
import csv_storage
import os
import json

# NEW FUNCTION

def load_database_embeddings():
    df2 = pd.read_csv("Data/employees_embeddings.csv")

    database_embeddings = {}

    for index, row in df2.iterrows():
        emp_id = row['Employee_ID']
        emb_list = list(map(float, row['Embedding'].split(",")))
        emb_array = np.array(emb_list)

        database_embeddings[emp_id] = emb_array

    return database_embeddings

# EXISTING CODE

df1 = pd.read_csv("Data/employees_info.csv")

attendance_mean = Mark_Attendance_Camera.attendance_mean_embedding
attendance_mean_embedding = np.array(attendance_mean)

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm1 = np.linalg.norm(a)
    norm2 = np.linalg.norm(b)
    return dot_product/(norm1*norm2)

def match_face(attendance_mean_embedding, registered_embedding, threshold=0.7):
    best_score = -1
    best_id = None

    for emp_id, stored_embedding in registered_embedding.items():

        score = cosine_similarity(attendance_mean_embedding, stored_embedding)

        if score > best_score:
            best_score = score
            best_id = emp_id

    if best_score >= threshold:
        return best_id, best_score
    else:
        return None, best_score

# ATTENDANCE LOGIC

def mark_attendance_logic(emp_id, emp_name):

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    day_str = now.strftime("%A")
    current_time = now.time()
    time_str = now.strftime("%H:%M:%S")

    in_status = half_day_status = out_status = ""
    in_time = half_day_time = out_time = ""

    # LOAD TIME CONFIG

    time_file = "Data/time_config.json"

    if os.path.exists(time_file):
        with open(time_file, "r") as f:
            time_data = json.load(f)
    else:

        # default fallback

        time_data = {
            "in_start": "08:50",
            "in_end": "10:00",
            "half_start": "13:30",
            "half_end": "14:50",
            "out_start": "16:59",
            "out_end": "18:00"
        }

    in_start = datetime.strptime(time_data["in_start"], "%H:%M").time()
    in_end   = datetime.strptime(time_data["in_end"], "%H:%M").time()

    half_start = datetime.strptime(time_data["half_start"], "%H:%M").time()
    half_end   = datetime.strptime(time_data["half_end"], "%H:%M").time()

    out_start = datetime.strptime(time_data["out_start"], "%H:%M").time()
    out_end   = datetime.strptime(time_data["out_end"], "%H:%M").time()

    # Load attendance file
    
    if os.path.exists(csv_storage.ATTENDANCE_FILE):
        df = pd.read_csv(csv_storage.ATTENDANCE_FILE)
    else:
        df = pd.DataFrame()

    # HANDLE EMPTY FILE SAFELY

    if df.empty:
        df = pd.DataFrame(columns=[
            "Employee_ID", "Employee_Name", "Date", "Day",
            "In", "Half_Day", "Out",
            "In_Time", "Half_Day_Time", "Out_Time"
        ])

    mask = (df["Employee_ID"].astype(str) == str(emp_id)) & (df["Date"] == date_str)

    # IN ATTENDANCE

    if in_start <= current_time <= in_end:

        if mask.any() and df.loc[mask, "In"].values[0] == "Present":
            return False, "⚠ IN attendance already marked today"

        in_status = "Present"
        in_time = time_str

    # HALF DAY
    
    elif half_start <= current_time <= half_end:

        if mask.any() and df.loc[mask, "Half_Day"].values[0] == "Halfday Taken":
            return False, "⚠ Half Day already marked today"

        half_day_status = "Halfday Taken"
        half_day_time = time_str

    # OUT ATTENDANCE
    
    elif out_start <= current_time <= out_end:

        if mask.any():

            row = df.loc[mask].iloc[0]

            if row["Half_Day"] == "Halfday Taken":
                return False, "⚠ OUT attendance not allowed after Half Day"

            if row["Out"] == "Out_At":
                return False, "⚠ OUT already marked today"

        out_status = "Out_At"
        out_time = time_str

    else:
        return False, "Attendance time not valid"

    # Save attendance

    csv_storage.save_attendance(
        emp_id,
        emp_name,
        date_str,
        day_str,
        in_status,
        half_day_status,
        out_status,
        in_time,
        half_day_time,
        out_time
    )

    return True, "Attendance marked successfully"