import csv
import os
import pandas as pd

# FILE PATHS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "Data")
EMP_FILE = os.path.join(DATA_DIR, "employees_info.csv")
EMB_FILE = os.path.join(DATA_DIR, "employees_embeddings.csv")
ATTENDANCE_FILE = os.path.join(DATA_DIR, "attendance.csv")

os.makedirs(DATA_DIR, exist_ok=True)

# SAVE EMPLOYEE INFO

def save_employee(emp_id, name, phone, email, address, date):
    file_exists = os.path.exists(EMP_FILE)

    with open(EMP_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "Employee_ID",
                "Employee_Name",
                "Phone_Number",
                "Email",
                "Address",
                "Joining_Date"
            ])

        writer.writerow([emp_id, name, phone, email, address, date])

# SAVE EMBEDDINGS

def save_embedding(emp_id, embedding):
    file_exists = os.path.exists(EMB_FILE)
    embedding_str = ",".join(map(lambda x: str(round(float(x), 6)), embedding.tolist()))

    with open(EMB_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["Employee_ID", "Embedding"])

        writer.writerow([emp_id, embedding_str])

# UPDATE EMPLOYEE INFO

def update_employee(old_emp_id, new_emp_id, name, phone, email, address, date):
    if not os.path.exists(EMP_FILE):
        raise Exception("Employee info file not found.")

    df = pd.read_csv(EMP_FILE)

    if old_emp_id not in df["Employee_ID"].values:
        raise Exception("Employee ID not found.")

    # Update employee info
    
    df.loc[df["Employee_ID"] == old_emp_id, [
        "Employee_ID",
        "Employee_Name",
        "Phone_Number",
        "Email",
        "Address",
        "Joining_Date"
    ]] = [new_emp_id, name, phone, email, address, date]

    df.to_csv(EMP_FILE, index=False)

    # Update ID in embeddings file

    if os.path.exists(EMB_FILE):
        df_emb = pd.read_csv(EMB_FILE)
        if old_emp_id in df_emb["Employee_ID"].values:
            df_emb.loc[df_emb["Employee_ID"] == old_emp_id, "Employee_ID"] = new_emp_id
            df_emb.to_csv(EMB_FILE, index=False)

# SAVE ATTENDANCE

def save_attendance(emp_id, emp_name, date_str, day_str,
                    in_status="", half_day_status="", out_status="",
                    in_time="", half_day_time="", out_time=""):

    # Load CSV or create empty dataframe

    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
    else:
        df = pd.DataFrame(columns=[
            "Employee_ID","Employee_Name","Date","Day",
            "In","In_Time","Half_Day","Half_Day_Time",
            "Out","Out_Time"
        ])

    # Ensure Employee_ID is treated as string

    df["Employee_ID"] = df["Employee_ID"].astype(str)
    emp_id = str(emp_id)

    # Find existing record for same employee + date

    mask = (df["Employee_ID"] == emp_id) & (df["Date"] == date_str)

    if mask.any():

        idx = df[mask].index[0]

        # Update only the required columns

        if in_status:
            df.at[idx, "In"] = in_status
            df.at[idx, "In_Time"] = in_time

        if half_day_status:
            df.at[idx, "Half_Day"] = half_day_status
            df.at[idx, "Half_Day_Time"] = half_day_time

        if out_status:
            df.at[idx, "Out"] = out_status
            df.at[idx, "Out_Time"] = out_time

    else:

        # Create new row only if no record exists

        new_row = {
            "Employee_ID": emp_id,
            "Employee_Name": emp_name,
            "Date": date_str,
            "Day": day_str,
            "In": in_status,
            "In_Time": in_time,
            "Half_Day": half_day_status,
            "Half_Day_Time": half_day_time,
            "Out": out_status,
            "Out_Time": out_time
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Save updated CSV
    df.to_csv(ATTENDANCE_FILE, index=False)