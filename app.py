import streamlit as st 
from datetime import date
import pandas as pd
import os
import time
import json

from csv_storage import (
    save_employee,
    update_employee,
    save_embedding,
    EMP_FILE
)

from Register_Camera import capture_faces_streamlit
from Mark_Attendance_Camera import capture_attendance_face_streamlit
from Embedding_Matcher import match_face, mark_attendance_logic, load_database_embeddings

# ADMIN FILE

ADMIN_FILE = "Data/admin_credentials.json"

def load_admin():
    if os.path.exists(ADMIN_FILE):
        with open(ADMIN_FILE, "r") as f:
            return json.load(f)
    return {"username": "admin", "password": "admin123", "email": "admin@gmail.com"}

def save_admin(username, password):
    with open(ADMIN_FILE, "w") as f:
        json.dump({
            "username": username,
            "password": password,
        }, f)

admin_data = load_admin()

# PAGE CONFIG

st.set_page_config(page_title="Face Attendance System", layout="wide")

st.title("📋 Facial Recognition Attendance System")

# LOGIN SYSTEM

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "role" not in st.session_state:
    st.session_state.role = None

if not st.session_state.logged_in:

    st.subheader("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Login as Admin"):
            if username == admin_data["username"] and password == admin_data["password"]:
                st.session_state.logged_in = True
                st.session_state.role = "admin"
                st.success("Admin Login Successful ✅")
                st.rerun()
            else:
                st.error("Invalid Admin Credentials ❌")

    with col2:
        if st.button("Continue as User"):
            st.session_state.logged_in = True
            st.session_state.role = "user"
            st.rerun()

    st.stop()

# LOGOUT

if st.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.session_state.role = None
    st.rerun()

# MENU

if st.session_state.role == "admin":
    mode = st.radio(
    "Admin Panel",
    ["Register Employee", "View Attendance", "Update Employee Data", "Admin Settings", "Update Attendance Time"],
    horizontal=True
    )
else:
    mode = st.radio("User Panel", ["Mark Attendance"], horizontal=True)

st.markdown("---")

# REGISTER EMPLOYEE

if mode == "Register Employee":

    st.subheader("Employee Registration")

    # 📷 CAPTURE FACE

    if st.button("📷 Capture Face"):

        frame_placeholder = st.empty()
        final_embedding = None

        for frame, embeddings, mean_embedding in capture_faces_streamlit(max_faces=10):
            if frame is not None:
                frame_placeholder.image(frame)

            if mean_embedding is not None:
                final_embedding = mean_embedding
                break

        if final_embedding is not None:
            st.session_state.embedding = final_embedding
            st.success("Face Captured ✅")

    # 📝 FORM

    form_key = "emp_form_" + str(st.session_state.get("form_reset", 0))

    with st.form(form_key):

        col1, col2 = st.columns(2)

        with col1:
            emp_id = st.text_input("Employee ID")
            emp_phone = st.text_input("Phone Number")

        with col2:
            emp_name = st.text_input("Employee Name")
            emp_date = st.date_input("Date", value=date.today())

        emp_email = st.text_input("Email")
        emp_address = st.text_area("Address")

        st.markdown("---")

        submitted = st.form_submit_button("💾 Save Employee")

        # 💾 SAVE EMPLOYEE
        
        if submitted:

            if "embedding" not in st.session_state:
                st.error("Please capture face first ❌")

            else:
                save_employee(
                    emp_id,
                    emp_name,
                    emp_phone,
                    emp_email,
                    emp_address,
                    str(emp_date)
                )

                save_embedding(emp_id, st.session_state.embedding)

                # REMOVE EMBEDDING AFTER SAVING
                del st.session_state.embedding

                st.success("Employee Registered Successfully ✅")

                # RESET FORM (NO ERROR)
                st.session_state.form_reset = st.session_state.get("form_reset", 0) + 1
                st.rerun()

# VIEW ATTENDANCE

elif mode == "View Attendance":

    file = os.path.join("Data", "attendance.csv")

    if os.path.exists(file):
        df = pd.read_csv(file)
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("No attendance records")

# UPDATE EMPLOYEE

elif mode == "Update Employee Data":

    df = pd.read_csv(EMP_FILE)
    edited = st.data_editor(df, use_container_width=True)

    if st.button("Update Changes"):
        edited.to_csv(EMP_FILE, index=False)
        st.success("Updated Successfully ✅")

# ADMIN SETTINGS

elif mode == "Admin Settings":

    st.subheader("⚙️ Change Credentials")

    form_key = "admin_form_" + str(st.session_state.get("admin_reset", 0))

    with st.form(form_key):

        current_user = st.text_input("Current Username")
        current_pass = st.text_input("Current Password", type="password")

        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")

        submitted = st.form_submit_button("Update Credentials")

        if submitted:

            if (
                current_user == admin_data["username"] and
                current_pass == admin_data["password"]
            ):
                save_admin(new_user, new_pass)

                st.success("Credentials Updated Successfully ✅")

                # 🔥 RESET FORM
                st.session_state.admin_reset = st.session_state.get("admin_reset", 0) + 1
                st.rerun()

            else:
                st.error("Invalid current username or password ❌")

# UPDATE ATTENDANCE TIME

elif mode == "Update Attendance Time":

    st.subheader("⏰ Update Attendance Timing")

    time_file = "Data/time_config.json"

    if os.path.exists(time_file):
        with open(time_file, "r") as f:
            time_data = json.load(f)
    else:
        time_data = {
            "in_start": "05:50",
            "in_end": "05:52",
            "half_start": "05:54",
            "half_end": "05:56",
            "out_start": "05:58",
            "out_end": "19:29"
        }

    in_start = st.text_input("IN Start", value=time_data["in_start"])
    in_end   = st.text_input("IN End", value=time_data["in_end"])
    half_start = st.text_input("Half Start", value=time_data["half_start"])
    half_end   = st.text_input("Half End", value=time_data["half_end"])
    out_start  = st.text_input("OUT Start", value=time_data["out_start"])
    out_end    = st.text_input("OUT End", value=time_data["out_end"])

    if st.button("Save Timing"):

        new_data = {
            "in_start": in_start,
            "in_end": in_end,
            "half_start": half_start,
            "half_end": half_end,
            "out_start": out_start,
            "out_end": out_end
        }

        with open(time_file, "w") as f:
            json.dump(new_data, f)

        st.success("Timing Updated ✅")
            
# MARK ATTENDANCE

elif mode == "Mark Attendance":

    st.subheader("📸 Mark Attendance")

    frame_placeholder = st.empty()
    message_placeholder = st.empty()

    generator = capture_attendance_face_streamlit(max_faces=5)

    while st.session_state.logged_in and st.session_state.role == "user":

        frame, embeddings, keypoints, mean_embedding = next(generator)

        if frame is not None:
            frame_placeholder.image(frame)

        if mean_embedding is not None:
            database_embeddings = load_database_embeddings()
            emp_id, score = match_face(mean_embedding, database_embeddings)

            if emp_id:

                try:
                    df = pd.read_csv("Data/employees_info.csv")

                    emp_name = df[df["Employee_ID"].astype(str) == str(emp_id)]["Employee_Name"].values

                    if len(emp_name) > 0:
                        emp_name = emp_name[0]
                    else:
                        emp_name = "Unknown"

                except Exception as e:
                    emp_name = "Unknown"

                success, message = mark_attendance_logic(emp_id, emp_name)

                # CLEAR PREVIOUS MESSAGE
                message_placeholder.empty()

                # SHOW NEW MESSAGE
                if success:
                    message_placeholder.success(f"{emp_name}: {message} ✅")
                else:
                    message_placeholder.warning(f"{emp_name}: {message}")

                time.sleep(2)

            else:
                message_placeholder.empty()
                message_placeholder.error("Face Not Recognized ❌")