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
from Embedding_Matcher import match_face, database_embeddings, mark_attendance_logic

# -------------------------------------------------
# ADMIN FILE
# -------------------------------------------------
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

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Face Attendance System", layout="wide")

st.title("📋 Facial Recognition Attendance System")

# -------------------------------------------------
# LOGIN SYSTEM
# -------------------------------------------------
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

# -------------------------------------------------
# LOGOUT
# -------------------------------------------------
if st.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.session_state.role = None
    st.rerun()

# -------------------------------------------------
# MENU
# -------------------------------------------------
if st.session_state.role == "admin":
    mode = st.radio(
        "Admin Panel",
        ["Register Employee", "View Attendance", "Update Employee Data", "Admin Settings"],
        horizontal=True
    )
else:
    mode = st.radio("User Panel", ["Mark Attendance"], horizontal=True)

st.markdown("---")

# =================================================
# REGISTER EMPLOYEE (UNCHANGED)
# =================================================
if mode == "Register Employee":

    st.subheader("Employee Registration")

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

    if "embedding" in st.session_state:
        if st.button("💾 Save Employee"):

            save_employee(emp_id, emp_name, emp_phone, emp_email, emp_address, str(emp_date))
            save_embedding(emp_id, st.session_state.embedding)

            del st.session_state.embedding

            st.success("Employee Registered Successfully ✅")

# =================================================
# VIEW ATTENDANCE
# =================================================
elif mode == "View Attendance":

    file = os.path.join("Data", "attendance.csv")

    if os.path.exists(file):
        df = pd.read_csv(file)
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("No attendance records")

# =================================================
# UPDATE EMPLOYEE
# =================================================
elif mode == "Update Employee Data":

    df = pd.read_csv(EMP_FILE)
    edited = st.data_editor(df, use_container_width=True)

    if st.button("Update Changes"):
        edited.to_csv(EMP_FILE, index=False)
        st.success("Updated Successfully ✅")

# =================================================
# ADMIN SETTINGS (SIMPLIFIED)
# =================================================
elif mode == "Admin Settings":

    st.subheader("⚙️ Change Credentials")

    current_user = st.text_input("Current Username", key="cu_user")
    current_pass = st.text_input("Current Password", type="password", key="cu_pass")

    new_user = st.text_input("New Username", key="new_user")
    new_pass = st.text_input("New Password", type="password", key="new_pass")

    if st.button("Update Credentials"):

        if (
            current_user == admin_data["username"] and
            current_pass == admin_data["password"]
        ):
            save_admin(new_user, new_pass)
            st.success("Credentials Updated Successfully ✅")
        else:
            st.error("Invalid current username or password ❌")

# =================================================
# MARK ATTENDANCE (FIXED - NO CONTINUOUS MARKING)
# =================================================
elif mode == "Mark Attendance":

    st.subheader("📸 Mark Attendance")

    frame_placeholder = st.empty()

    marked_ids = set()  # ✅ prevents multiple marking

    for frame, embeddings, keypoints, mean_embedding in capture_attendance_face_streamlit(max_faces=5):

        if frame is not None:
            frame_placeholder.image(frame)

        if mean_embedding is not None:

            emp_id, score = match_face(mean_embedding, database_embeddings)

            if emp_id and emp_id not in marked_ids:

                marked_ids.add(emp_id)  # ✅ mark once

                mark_attendance_logic(emp_id)  # your original function

                st.success(f"Attendance Marked for {emp_id} ✅")

                time.sleep(2)  # small delay to avoid multiple triggers

            elif not emp_id:
                st.error("Face Not Recognized ❌")