#############################################
# IMPORTING LIBRARIES
#############################################
import tkinter as tk
from tkinter import ttk, messagebox as mess, simpledialog as tsd
import cv2
import os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time

#############################################
# APPLICATION CLASS
#############################################

class FaceAttendanceApp:
    """
    A desktop application for facial recognition-based attendance,
    built with tkinter.
    """
    def __init__(self, window):
        self.window = window
        self.window.geometry("1280x720")
        self.window.resizable(True, False)
        self.window.title("Face Recognition Attendance System")
        self.window.configure(background='#2d3436')

        # --- Constants ---
        self.HAAR_CASCADE_PATH = "haarcascade_frontalface_default.xml"
        self.TRAINING_IMAGE_PATH = "TrainingImage/"
        self.TRAINING_LABEL_PATH = "TrainingImageLabel/"
        self.STUDENT_DETAILS_PATH = "StudentDetails/StudentDetails.csv"
        self.ATTENDANCE_PATH = "Attendance/"
        self.PASSWORD_FILE = os.path.join(self.TRAINING_LABEL_PATH, "psd.txt")
        self.TRAINER_FILE = os.path.join(self.TRAINING_LABEL_PATH, "Trainner.yml")

        # --- Setup ---
        self.assure_paths_exist()
        self.check_haarcascade_file()
        self.setup_gui()
        self.tick()
        self.update_registration_count()

    def assure_paths_exist(self):
        """Ensures all necessary directories exist."""
        os.makedirs(self.TRAINING_IMAGE_PATH, exist_ok=True)
        os.makedirs(self.TRAINING_LABEL_PATH, exist_ok=True)
        os.makedirs(os.path.dirname(self.STUDENT_DETAILS_PATH), exist_ok=True)
        os.makedirs(self.ATTENDANCE_PATH, exist_ok=True)

    def check_haarcascade_file(self):
        """Checks for the Haar cascade file and shows an error if it's missing."""
        if not os.path.isfile(self.HAAR_CASCADE_PATH):
            mess.showerror('File Missing', f'{self.HAAR_CASCADE_PATH} is missing.')
            self.window.destroy()

    def tick(self):
        """Updates the clock display every 200 milliseconds."""
        time_string = time.strftime('%H:%M:%S')
        self.clock.config(text=time_string)
        self.clock.after(200, self.tick)

    def setup_gui(self):
        """Initializes all the GUI components."""
        # Frames
        frame1 = tk.Frame(self.window, bg="#3498db") # Attendance
        frame1.place(relx=0.11, rely=0.17, relwidth=0.39, relheight=0.80)

        frame2 = tk.Frame(self.window, bg="#3498db") # Registration
        frame2.place(relx=0.51, rely=0.17, relwidth=0.38, relheight=0.80)

        # Header
        tk.Label(self.window, text="Face Recognition Based Attendance System", fg="white", bg="#2d3436",
                 width=55, height=1, font=('Helvetica', 29, 'bold')).place(x=10, y=10)

        # Date and Time Display
        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
        tk.Label(self.window, text=f"Date: {date}", fg="white", bg="#2d3436", font=('Helvetica', 16)).place(x=10, y=80)
        
        self.clock = tk.Label(self.window, fg="white", bg="#2d3436", font=('Helvetica', 16))
        self.clock.place(x=1100, y=80)

        # Frame Headers
        tk.Label(frame1, text="Attendance Log", fg="white", bg="#2980b9", font=('Helvetica', 17, 'bold')).pack(side=tk.TOP, fill=tk.X)
        tk.Label(frame2, text="New User Registration", fg="white", bg="#2980b9", font=('Helvetica', 17, 'bold')).pack(side=tk.TOP, fill=tk.X)

        # --- Registration Section (Frame 2) ---
        tk.Label(frame2, text="Enter ID", width=20, height=1, fg="white", bg="#3498db", font=('Helvetica', 15, 'bold')).place(x=80, y=55)
        self.txt_id = tk.Entry(frame2, width=32, fg="black", font=('Helvetica', 15, 'bold'))
        self.txt_id.place(x=30, y=88)

        tk.Label(frame2, text="Enter Name", width=20, fg="white", bg="#3498db", font=('Helvetica', 15, 'bold')).place(x=80, y=140)
        self.txt_name = tk.Entry(frame2, width=32, fg="black", font=('Helvetica', 15, 'bold'))
        self.txt_name.place(x=30, y=173)

        self.message1 = tk.Label(frame2, text="1. Take Images  ->  2. Save Profile", bg="#3498db", fg="white", width=39, height=1, font=('Helvetica', 15, 'italic'))
        self.message1.place(x=7, y=230)

        self.lbl_reg_count = tk.Label(frame2, text="", bg="#3498db", fg="white", width=39, height=1, font=('Helvetica', 16, 'bold'))
        self.lbl_reg_count.place(x=7, y=500)

        # --- Attendance Section (Frame 1) ---
        # Treeview for Attendance Table
        self.tv = ttk.Treeview(frame1, height=15, columns=('name', 'date', 'time'))
        self.tv.column('#0', width=82)
        self.tv.column('name', width=130)
        self.tv.column('date', width=133)
        self.tv.column('time', width=133)
        self.tv.place(x=30, y=110)
        self.tv.heading('#0', text='ID')
        self.tv.heading('name', text='NAME')
        self.tv.heading('date', text='DATE')
        self.tv.heading('time', text='TIME')

        scroll = ttk.Scrollbar(frame1, orient='vertical', command=self.tv.yview)
        scroll.place(x=478, y=110, height=345)
        self.tv.configure(yscrollcommand=scroll.set)

        # --- Buttons ---
        tk.Button(frame2, text="Clear", command=self.clear_fields, fg="white", bg="#e74c3c", width=11, font=('Helvetica', 11, 'bold')).place(x=335, y=86)
        tk.Button(frame2, text="Take Images", command=self.take_images, fg="white", bg="#2c3e50", width=34, height=1, font=('Helvetica', 15, 'bold')).place(x=30, y=300)
        tk.Button(frame2, text="Save Profile", command=self.psw, fg="white", bg="#2c3e50", width=34, height=1, font=('Helvetica', 15, 'bold')).place(x=30, y=380)
        tk.Button(frame1, text="Take Attendance", command=self.track_images, fg="white", bg="#27ae60", width=35, height=1, font=('Helvetica', 15, 'bold')).place(x=30, y=50)
        tk.Button(frame1, text="Quit", command=self.window.destroy, fg="white", bg="#c0392b", width=35, height=1, font=('Helvetica', 15, 'bold')).place(x=30, y=500)

    def clear_fields(self):
        self.txt_id.delete(0, 'end')
        self.txt_name.delete(0, 'end')
        self.message1.configure(text="1. Take Images  ->  2. Save Profile")

    def take_images(self):
        Id = self.txt_id.get()
        name = self.txt_name.get()

        if not (Id.isdigit() and name.replace(' ', '').isalpha()):
            mess.showerror("Input Error", "ID must be a number and Name must be alphabetic.")
            return

        # Check if ID already exists
        if os.path.isfile(self.STUDENT_DETAILS_PATH):
            df_check = pd.read_csv(self.STUDENT_DETAILS_PATH)
            if not df_check[df_check['ID'] == int(Id)].empty:
                mess.showerror("Error", f"ID {Id} already exists.")
                return

        cam = cv2.VideoCapture(0)
        detector = cv2.CascadeClassifier(self.HAAR_CASCADE_PATH)
        sampleNum = 0
        
        while True:
            ret, img = cam.read()
            if not ret: break
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                sampleNum += 1
                img_path = os.path.join(self.TRAINING_IMAGE_PATH, f"{name}.{Id}.{sampleNum}.jpg")
                cv2.imwrite(img_path, gray[y:y + h, x:x + w])
                cv2.imshow('Taking Images', img)
            
            if cv2.waitKey(100) & 0xFF == ord('q') or sampleNum >= 100:
                break
                
        cam.release()
        cv2.destroyAllWindows()
        
        # Save details to CSV
        if os.path.isfile(self.STUDENT_DETAILS_PATH):
             df = pd.read_csv(self.STUDENT_DETAILS_PATH)
             serial = df['SERIAL NO.'].max() + 1 if not df.empty else 1
        else:
            serial = 1

        row = [serial, Id, name]
        header = ['SERIAL NO.', 'ID', 'NAME']
        mode = 'a+' if os.path.exists(self.STUDENT_DETAILS_PATH) else 'w'
        with open(self.STUDENT_DETAILS_PATH, mode, newline='') as csvFile:
            writer = csv.writer(csvFile)
            if mode == 'w': writer.writerow(header)
            writer.writerow(row)
        
        self.message1.configure(text=f"Images Taken for ID: {Id}")
        self.update_registration_count()

    def train_images(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        faces, Ids = self.get_images_and_labels(self.TRAINING_IMAGE_PATH)
        
        if not faces:
            mess.showerror('No Data', 'No images found to train. Please register someone first!')
            return
            
        try:
            recognizer.train(faces, np.array(Ids))
            recognizer.save(self.TRAINER_FILE)
            self.message1.configure(text="Profile Saved Successfully")
            self.update_registration_count()
        except cv2.error as e:
            mess.showerror("Training Error", f"An error occurred: {e}")

    def get_images_and_labels(self, path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faces, Ids = [], []
        
        for imagePath in imagePaths:
            try:
                pilImage = Image.open(imagePath).convert('L')
                imageNp = np.array(pilImage, 'uint8')
                ID = int(os.path.split(imagePath)[-1].split(".")[1])
                faces.append(imageNp)
                Ids.append(ID)
            except Exception as e:
                print(f"Skipping file {imagePath}: {e}")
                
        return faces, Ids

    def track_images(self):
        if not os.path.isfile(self.TRAINER_FILE):
            mess.showerror('Data Missing', 'Trainer data is missing. Please save a profile first.')
            return
        if not os.path.isfile(self.STUDENT_DETAILS_PATH):
            mess.showerror('Details Missing', 'Student details are missing.')
            return

        for k in self.tv.get_children(): self.tv.delete(k)

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(self.TRAINER_FILE)
        faceCascade = cv2.CascadeClassifier(self.HAAR_CASCADE_PATH)
        df = pd.read_csv(self.STUDENT_DETAILS_PATH)
        
        cam = cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        date_str = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d')
        attendance_file = os.path.join(self.ATTENDANCE_PATH, f"Attendance_{date_str}.csv")
        
        present_ids = set()
        if os.path.exists(attendance_file):
            df_attendance = pd.read_csv(attendance_file)
            present_ids.update(df_attendance['ID'])

        while True:
            ret, im = cam.read()
            if not ret: break
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.2, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
                Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
                
                if conf < 50:
                    student_info = df.loc[df['ID'] == Id]
                    if not student_info.empty:
                        name = student_info['NAME'].values[0]
                        if Id not in present_ids:
                            ts = time.time()
                            timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                            attendance = [Id, name, date_str, timeStamp]
                            
                            with open(attendance_file, 'a+', newline='') as csvFile:
                                writer = csv.writer(csvFile)
                                if os.path.getsize(attendance_file) == 0:
                                    writer.writerow(['ID', 'NAME', 'DATE', 'TIME'])
                                writer.writerow(attendance)
                            
                            present_ids.add(Id)
                            self.tv.insert('', 'end', text=str(Id), values=(name, date_str, timeStamp))
                    else:
                        name = 'Unknown'
                else:
                    name = 'Unknown'
                
                cv2.putText(im, str(name), (x, y + h), font, 1, (255, 255, 255), 2)
                
            cv2.imshow('Taking Attendance (Press "q" to exit)', im)
            if cv2.waitKey(1) == ord('q'): break
                
        cam.release()
        cv2.destroyAllWindows()

    def update_registration_count(self):
        res = 0
        if os.path.isfile(self.STUDENT_DETAILS_PATH):
            df = pd.read_csv(self.STUDENT_DETAILS_PATH)
            res = len(df)
        self.lbl_reg_count.configure(text=f'Total Registrations: {res}')

    def psw(self):
        """Handles password protection for training."""
        if not os.path.isfile(self.PASSWORD_FILE):
            new_pas = tsd.askstring('Password Not Set', 'Please enter a password to secure the system', show='*')
            if new_pas:
                with open(self.PASSWORD_FILE, "w") as tf: tf.write(new_pas)
                mess.showinfo('Password Registered', 'Password registered successfully!')
            else:
                mess.showerror('No Password Entered', 'Password not set! Please try again.')
            return

        with open(self.PASSWORD_FILE, "r") as tf: key = tf.read()
        password = tsd.askstring('Password', 'Enter Password', show='*')
        if password == key:
            self.train_images()
        elif password is not None:
            mess.showerror('Wrong Password', 'You have entered the wrong password.')

#############################################
# MAIN EXECUTION
#############################################
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceAttendanceApp(root)
    root.mainloop()
#############################################
# IMPORTING LIBRARIES
#############################################
import tkinter as tk
from tkinter import ttk, messagebox as mess, simpledialog as tsd
import cv2
import os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time

#############################################
# APPLICATION CLASS
#############################################

class FaceAttendanceApp:
    """
    A desktop application for facial recognition-based attendance,
    built with tkinter.
    """
    def __init__(self, window):
        self.window = window
        self.window.geometry("1280x720")
        self.window.resizable(True, False)
        self.window.title("Face Recognition Attendance System")
        self.window.configure(background='#2d3436')

        # --- Constants ---
        self.HAAR_CASCADE_PATH = "haarcascade_frontalface_default.xml"
        self.TRAINING_IMAGE_PATH = "TrainingImage/"
        self.TRAINING_LABEL_PATH = "TrainingImageLabel/"
        self.STUDENT_DETAILS_PATH = "StudentDetails/StudentDetails.csv"
        self.ATTENDANCE_PATH = "Attendance/"
        self.PASSWORD_FILE = os.path.join(self.TRAINING_LABEL_PATH, "psd.txt")
        self.TRAINER_FILE = os.path.join(self.TRAINING_LABEL_PATH, "Trainner.yml")

        # --- Setup ---
        self.assure_paths_exist()
        self.check_haarcascade_file()
        self.setup_gui()
        self.tick()
        self.update_registration_count()

    def assure_paths_exist(self):
        """Ensures all necessary directories exist."""
        os.makedirs(self.TRAINING_IMAGE_PATH, exist_ok=True)
        os.makedirs(self.TRAINING_LABEL_PATH, exist_ok=True)
        os.makedirs(os.path.dirname(self.STUDENT_DETAILS_PATH), exist_ok=True)
        os.makedirs(self.ATTENDANCE_PATH, exist_ok=True)

    def check_haarcascade_file(self):
        """Checks for the Haar cascade file and shows an error if it's missing."""
        if not os.path.isfile(self.HAAR_CASCADE_PATH):
            mess.showerror('File Missing', f'{self.HAAR_CASCADE_PATH} is missing.')
            self.window.destroy()

    def tick(self):
        """Updates the clock display every 200 milliseconds."""
        time_string = time.strftime('%H:%M:%S')
        self.clock.config(text=time_string)
        self.clock.after(200, self.tick)

    def setup_gui(self):
        """Initializes all the GUI components."""
        # Frames
        frame1 = tk.Frame(self.window, bg="#3498db") # Attendance
        frame1.place(relx=0.11, rely=0.17, relwidth=0.39, relheight=0.80)

        frame2 = tk.Frame(self.window, bg="#3498db") # Registration
        frame2.place(relx=0.51, rely=0.17, relwidth=0.38, relheight=0.80)

        # Header
        tk.Label(self.window, text="Face Recognition Based Attendance System", fg="white", bg="#2d3436",
                 width=55, height=1, font=('Helvetica', 29, 'bold')).place(x=10, y=10)

        # Date and Time Display
        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
        tk.Label(self.window, text=f"Date: {date}", fg="white", bg="#2d3436", font=('Helvetica', 16)).place(x=10, y=80)
        
        self.clock = tk.Label(self.window, fg="white", bg="#2d3436", font=('Helvetica', 16))
        self.clock.place(x=1100, y=80)

        # Frame Headers
        tk.Label(frame1, text="Attendance Log", fg="white", bg="#2980b9", font=('Helvetica', 17, 'bold')).pack(side=tk.TOP, fill=tk.X)
        tk.Label(frame2, text="New User Registration", fg="white", bg="#2980b9", font=('Helvetica', 17, 'bold')).pack(side=tk.TOP, fill=tk.X)

        # --- Registration Section (Frame 2) ---
        tk.Label(frame2, text="Enter ID", width=20, height=1, fg="white", bg="#3498db", font=('Helvetica', 15, 'bold')).place(x=80, y=55)
        self.txt_id = tk.Entry(frame2, width=32, fg="black", font=('Helvetica', 15, 'bold'))
        self.txt_id.place(x=30, y=88)

        tk.Label(frame2, text="Enter Name", width=20, fg="white", bg="#3498db", font=('Helvetica', 15, 'bold')).place(x=80, y=140)
        self.txt_name = tk.Entry(frame2, width=32, fg="black", font=('Helvetica', 15, 'bold'))
        self.txt_name.place(x=30, y=173)

        self.message1 = tk.Label(frame2, text="1. Take Images  ->  2. Save Profile", bg="#3498db", fg="white", width=39, height=1, font=('Helvetica', 15, 'italic'))
        self.message1.place(x=7, y=230)

        self.lbl_reg_count = tk.Label(frame2, text="", bg="#3498db", fg="white", width=39, height=1, font=('Helvetica', 16, 'bold'))
        self.lbl_reg_count.place(x=7, y=500)

        # --- Attendance Section (Frame 1) ---
        # Treeview for Attendance Table
        self.tv = ttk.Treeview(frame1, height=15, columns=('name', 'date', 'time'))
        self.tv.column('#0', width=82)
        self.tv.column('name', width=130)
        self.tv.column('date', width=133)
        self.tv.column('time', width=133)
        self.tv.place(x=30, y=110)
        self.tv.heading('#0', text='ID')
        self.tv.heading('name', text='NAME')
        self.tv.heading('date', text='DATE')
        self.tv.heading('time', text='TIME')

        scroll = ttk.Scrollbar(frame1, orient='vertical', command=self.tv.yview)
        scroll.place(x=478, y=110, height=345)
        self.tv.configure(yscrollcommand=scroll.set)

        # --- Buttons ---
        tk.Button(frame2, text="Clear", command=self.clear_fields, fg="white", bg="#e74c3c", width=11, font=('Helvetica', 11, 'bold')).place(x=335, y=86)
        tk.Button(frame2, text="Take Images", command=self.take_images, fg="white", bg="#2c3e50", width=34, height=1, font=('Helvetica', 15, 'bold')).place(x=30, y=300)
        tk.Button(frame2, text="Save Profile", command=self.psw, fg="white", bg="#2c3e50", width=34, height=1, font=('Helvetica', 15, 'bold')).place(x=30, y=380)
        tk.Button(frame1, text="Take Attendance", command=self.track_images, fg="white", bg="#27ae60", width=35, height=1, font=('Helvetica', 15, 'bold')).place(x=30, y=50)
        tk.Button(frame1, text="Quit", command=self.window.destroy, fg="white", bg="#c0392b", width=35, height=1, font=('Helvetica', 15, 'bold')).place(x=30, y=500)

    def clear_fields(self):
        self.txt_id.delete(0, 'end')
        self.txt_name.delete(0, 'end')
        self.message1.configure(text="1. Take Images  ->  2. Save Profile")

    def take_images(self):
        Id = self.txt_id.get()
        name = self.txt_name.get()

        if not (Id.isdigit() and name.replace(' ', '').isalpha()):
            mess.showerror("Input Error", "ID must be a number and Name must be alphabetic.")
            return

        # Check if ID already exists
        if os.path.isfile(self.STUDENT_DETAILS_PATH):
            df_check = pd.read_csv(self.STUDENT_DETAILS_PATH)
            if not df_check[df_check['ID'] == int(Id)].empty:
                mess.showerror("Error", f"ID {Id} already exists.")
                return

        cam = cv2.VideoCapture(0)
        detector = cv2.CascadeClassifier(self.HAAR_CASCADE_PATH)
        sampleNum = 0
        
        while True:
            ret, img = cam.read()
            if not ret: break
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                sampleNum += 1
                img_path = os.path.join(self.TRAINING_IMAGE_PATH, f"{name}.{Id}.{sampleNum}.jpg")
                cv2.imwrite(img_path, gray[y:y + h, x:x + w])
                cv2.imshow('Taking Images', img)
            
            if cv2.waitKey(100) & 0xFF == ord('q') or sampleNum >= 100:
                break
                
        cam.release()
        cv2.destroyAllWindows()
        
        # Save details to CSV
        if os.path.isfile(self.STUDENT_DETAILS_PATH):
             df = pd.read_csv(self.STUDENT_DETAILS_PATH)
             serial = df['SERIAL NO.'].max() + 1 if not df.empty else 1
        else:
            serial = 1

        row = [serial, Id, name]
        header = ['SERIAL NO.', 'ID', 'NAME']
        mode = 'a+' if os.path.exists(self.STUDENT_DETAILS_PATH) else 'w'
        with open(self.STUDENT_DETAILS_PATH, mode, newline='') as csvFile:
            writer = csv.writer(csvFile)
            if mode == 'w': writer.writerow(header)
            writer.writerow(row)
        
        self.message1.configure(text=f"Images Taken for ID: {Id}")
        self.update_registration_count()

    def train_images(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        faces, Ids = self.get_images_and_labels(self.TRAINING_IMAGE_PATH)
        
        if not faces:
            mess.showerror('No Data', 'No images found to train. Please register someone first!')
            return
            
        try:
            recognizer.train(faces, np.array(Ids))
            recognizer.save(self.TRAINER_FILE)
            self.message1.configure(text="Profile Saved Successfully")
            self.update_registration_count()
        except cv2.error as e:
            mess.showerror("Training Error", f"An error occurred: {e}")

    def get_images_and_labels(self, path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faces, Ids = [], []
        
        for imagePath in imagePaths:
            try:
                pilImage = Image.open(imagePath).convert('L')
                imageNp = np.array(pilImage, 'uint8')
                ID = int(os.path.split(imagePath)[-1].split(".")[1])
                faces.append(imageNp)
                Ids.append(ID)
            except Exception as e:
                print(f"Skipping file {imagePath}: {e}")
                
        return faces, Ids

    def track_images(self):
        if not os.path.isfile(self.TRAINER_FILE):
            mess.showerror('Data Missing', 'Trainer data is missing. Please save a profile first.')
            return
        if not os.path.isfile(self.STUDENT_DETAILS_PATH):
            mess.showerror('Details Missing', 'Student details are missing.')
            return

        for k in self.tv.get_children(): self.tv.delete(k)

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(self.TRAINER_FILE)
        faceCascade = cv2.CascadeClassifier(self.HAAR_CASCADE_PATH)
        df = pd.read_csv(self.STUDENT_DETAILS_PATH)
        
        cam = cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        date_str = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d')
        attendance_file = os.path.join(self.ATTENDANCE_PATH, f"Attendance_{date_str}.csv")
        
        present_ids = set()
        if os.path.exists(attendance_file):
            df_attendance = pd.read_csv(attendance_file)
            present_ids.update(df_attendance['ID'])

        while True:
            ret, im = cam.read()
            if not ret: break
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.2, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
                Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
                
                if conf < 50:
                    student_info = df.loc[df['ID'] == Id]
                    if not student_info.empty:
                        name = student_info['NAME'].values[0]
                        if Id not in present_ids:
                            ts = time.time()
                            timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                            attendance = [Id, name, date_str, timeStamp]
                            
                            with open(attendance_file, 'a+', newline='') as csvFile:
                                writer = csv.writer(csvFile)
                                if os.path.getsize(attendance_file) == 0:
                                    writer.writerow(['ID', 'NAME', 'DATE', 'TIME'])
                                writer.writerow(attendance)
                            
                            present_ids.add(Id)
                            self.tv.insert('', 'end', text=str(Id), values=(name, date_str, timeStamp))
                    else:
                        name = 'Unknown'
                else:
                    name = 'Unknown'
                
                cv2.putText(im, str(name), (x, y + h), font, 1, (255, 255, 255), 2)
                
            cv2.imshow('Taking Attendance (Press "q" to exit)', im)
            if cv2.waitKey(1) == ord('q'): break
                
        cam.release()
        cv2.destroyAllWindows()

    def update_registration_count(self):
        res = 0
        if os.path.isfile(self.STUDENT_DETAILS_PATH):
            df = pd.read_csv(self.STUDENT_DETAILS_PATH)
            res = len(df)
        self.lbl_reg_count.configure(text=f'Total Registrations: {res}')

    def psw(self):
        """Handles password protection for training."""
        if not os.path.isfile(self.PASSWORD_FILE):
            new_pas = tsd.askstring('Password Not Set', 'Please enter a password to secure the system', show='*')
            if new_pas:
                with open(self.PASSWORD_FILE, "w") as tf: tf.write(new_pas)
                mess.showinfo('Password Registered', 'Password registered successfully!')
            else:
                mess.showerror('No Password Entered', 'Password not set! Please try again.')
            return

        with open(self.PASSWORD_FILE, "r") as tf: key = tf.read()
        password = tsd.askstring('Password', 'Enter Password', show='*')
        if password == key:
            self.train_images()
        elif password is not None:
            mess.showerror('Wrong Password', 'You have entered the wrong password.')

#############################################
# MAIN EXECUTION
#############################################
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceAttendanceApp(root)
    root.mainloop()
