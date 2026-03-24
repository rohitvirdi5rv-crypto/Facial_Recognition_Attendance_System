from cx_Freeze import setup, Executable

include_files = [
    ("Data", "Data"),
    ("app.py", "app.py"),
    ("csv_storage.py", "csv_storage.py"),
    ("Register_Camera.py", "Register_Camera.py"),
    ("Mark_Attendance_Camera.py", "Mark_Attendance_Camera.py"),
    ("Embedding_Matcher.py", "Embedding_Matcher.py"),
    ("venv/Lib/site-packages/cv2", "cv2")
]

build_exe_options = {
    "packages": [
        "streamlit",
        "pandas",
        "numpy",
        "cv2",
        "mtcnn",
        "keras_facenet",
        "tensorflow",
        "sklearn",
        "pkg_resources",
    ],
    "includes": [
        "csv_storage",
        "Register_Camera",
        "Mark_Attendance_Camera",
        "Embedding_Matcher",
        "cv2"   
    ],
    "include_files": include_files
}

setup(
    name="Face Attendance System",
    version="1.0",
    description="Face Recognition Attendance",
    options={"build_exe": build_exe_options},
    executables=[
        Executable("run_app.py", icon="verified-user.ico")  
    ]
)