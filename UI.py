
# same as the UI module

from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import Face_Recognition 
import cv2
import numpy as np
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine
import time
import queue
from datetime import datetime
import threading

TF_ENABLE_ONEDNN_OPTS=0
#original
class UI:
    def __init__(self):
        self.root = Tk()
        self.root.title('NDSpectra')
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.resizable(True, True)
        self.root.minsize(600, 500)
        self.root.maxsize(screen_width, screen_height)
        self.frame = Frame(self.root)
        self.frame.pack(fill='both', expand=True)
        self.admin_win = None
        self.canvas = None
        self.scrollbar = None
        self.scrollable_frame = None
        self.search_entry = None
        self.images_data = [] #image path , name and unique id
        self.known_embeddings = [] #for storing the embeddings of the known faces
        self.known_labels=None
        self.face_recognition=Face_Recognition.Face_Recognition()
        self.attendance_queue = queue.Queue()
        self.present_names = set() # TO AVOID DUPLICATES IN THE PRESENT LIST


    def get_embeddings(self):
        #gettig the embeddings of the faces added 
        #getting the image path from the source directory 
        image_path=[self.folder_path+'/'+i for i in os.listdir(self.folder_path)]
        for i in image_path:
            print(i)
        # preparing labels
        image_labels=[ i.split('/')[0] for i in os.listdir(self.folder_path)]
        self.known_labels=image_labels
        print('\n'.join([str(i) for  i in image_labels]))
        
        self.known_embeddings=[]
        for face_path in image_path:
            face_img = np.asarray(Image.open(face_path).convert('RGB'))
            face_array, _ = self.face_recognition.extract_face(face_img)
            if face_array is not None:
                face_embedding = self.face_recognition.embedder.embeddings(np.expand_dims(face_array, axis=0))[0]
                self.known_embeddings.append(normalize([face_embedding])[0])
        


    def extract_embeddings_thread(self, known_embeddings, known_labels):
        present = set()
        while True:
            if self.face_recognition.frame is not None:
                frame_copy = self.face_recognition.frame.copy()
                detected_names = set()
                for face_array, (x1, y1, x2, y2) in self.face_recognition.faces_data:
                    face_embedding = self.face_recognition.embedder.embeddings(np.expand_dims(face_array, axis=0))[0]
                    face_embedding = normalize([face_embedding])[0]

                    best_match_label, best_similarity = self.face_recognition.find_closest_match(face_embedding, known_embeddings, known_labels)
                    if best_similarity > 0.5:
                        detected_names.add(best_match_label)
                        label = f'{best_match_label}: {best_similarity:.2f}'
                        color = (0, 255, 0)
                    else:
                        label = 'Not Matched'
                        color = (0, 0, 255)

                    current_frame = frame_copy[y1:y2, x1:x2]
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.attendance_queue.put((best_match_label, current_frame, timestamp, best_similarity > 0.5))

                present.update(detected_names)
                absent = set(known_labels) - present

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def start_processing(self, cap, known_embeddings, known_labels):
        threading.Thread(target=self.face_recognition.detect_faces_thread, args=(cap, known_embeddings, known_labels), daemon=True).start()
        threading.Thread(target=self.extract_embeddings_thread, args=(known_embeddings, known_labels), daemon=True).start()

        

    def add_file(self):
        self.folder_path = filedialog.askdirectory(title='Choose the image folder')
        if not self.folder_path:
            messagebox.showerror("Error", 'No folder selected') 
            return

        try:
            self.images_data = []
            file_names = os.listdir(self.folder_path)
            for i, file_name in enumerate(file_names):
                image_path = os.path.join(self.folder_path, file_name)
                image_name = os.path.splitext(file_name)[0]
                self.images_data.append((image_path, image_name, i + 1))

            self.display_images(self.images_data) #function to dispay images in the admin window
            
            # Run get_embeddings in a separate thread
            threading.Thread(target=self.get_embeddings, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", str(e))
 

    def clear_admin_window(self):
        if self.canvas:
            self.canvas.destroy()
            self.scrollbar.destroy()
            self.canvas = None
            self.scrollbar = None
            self.scrollable_frame = None

    def display_images(self, images_data):
        # Clear previous content in admin window
        self.clear_admin_window()

        # Create canvas and scrollbar if not already created7
        self.canvas = Canvas(self.admin_win, bg="white")
        self.canvas.pack(side=LEFT, fill=BOTH, expand=True)
        self.scrollable_frame = Frame(self.canvas, bg="white")
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.scrollbar = Scrollbar(self.admin_win, orient=VERTICAL, command=self.canvas.yview)
        self.scrollbar.pack(side=RIGHT, fill=Y)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Add mouse wheel scrolling
        def _on_mouse_wheel(event):
            self.canvas.yview_scroll(-1 * int((event.delta / 120)), "units")

        self.canvas.bind_all("<MouseWheel>", _on_mouse_wheel)
        self.canvas.bind_all("<Button-4>", _on_mouse_wheel)
        self.canvas.bind_all("<Button-5>", _on_mouse_wheel)

        columns = 5  # Number of images in a row3
        for i, (path, name, idx) in enumerate(images_data):
            row = i // columns
            col = i % columns
            
            # Load the image
            img = Image.open(path)
            img = img.resize((220, 220))  # Resize the image to fit in the grid3
            img = ImageTk.PhotoImage(img)
            
            # Create a frame for each image and its details
            item_frame = Frame(self.scrollable_frame, bg="lightgray", bd=1, relief="solid")
            item_frame.grid(row=row, column=col, padx=20, pady=20, sticky="nsew")

            # Create a label for the image1
            img_label = Label(item_frame, image=img, padx=20, pady=20, width=240, height=240)
            img_label.image = img  # Keep a reference to avoid garbage collection
            img_label.pack()

            # Create a label for the name and unique ID
            name_id_label = Label(item_frame, text=f"{name}\nID: {idx}", bg='lightblue', padx=20, pady=20, width=30, wraplength=100, justify="center")
            name_id_label.pack()

        # Update scroll region of canvas
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
    def search_images(self):
        query = self.search_entry.get().lower()
        #searching for name and unique identifier. images_data (holds image path,name and index)
        filtered_images = [img for img in self.images_data if query in img[1].lower() or query in str(img[2])]
        self.display_images(filtered_images)
    
    # file menu in admin window 
    def edit_file(self):
        print(self.folder_path)
        

    def clear_attendance_list(self):
        for label in self.present_labels:
            label.destroy()
        for label in self.absent_labels:
            label.destroy()
        self.present_labels = []
        self.absent_labels = []

    def check_queue(self):
        try:
            while True:
                name, image, timestamp, is_present = self.attendance_queue.get_nowait()
                self.update_attendance_list(name, image, timestamp, is_present)
        except queue.Empty:
            pass
        self.root.after(50, self.check_queue)
            
    def update_attendance_list(self, name, image, timestamp, is_present):
        if self.attendance_window is not None and self.attendance_window.winfo_exists():
            # Check if the person is already in the present list to avoid duplicates
            if is_present and name not in self.present_names:
                frame = self.present_frame
                bg_color = 'lightgreen'
                # Remove from absent frame if present
                if name in self.absent_frames:
                    self.absent_frames[name].destroy()
                    del self.absent_frames[name]
                # Add to present names set to keep track
                self.present_names.add(name)
            else:
                # If the person is not present, do not add to the present list
                return

            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            img = img.resize((100, 100))
            img = ImageTk.PhotoImage(img)

            item_frame = Frame(frame, bg=bg_color, padx=10, pady=10)
            item_frame.pack(pady=5)

            img_label = Label(item_frame, image=img, padx=10, pady=10)
            img_label.image = img
            img_label.pack(side=LEFT)

            info_label = Label(item_frame, text=f"{name}\n{timestamp}", bg=bg_color, padx=10, pady=10)
            info_label.pack(side=RIGHT)

                    
    def attendance_window(self):
        self.attendance_window = Toplevel(self.root)
        self.attendance_window.title('Attendance')
        self.attendance_window.geometry('800x600')

        self.present_frame = LabelFrame(self.attendance_window, text="Present", bg='white', padx=10, pady=10)
        self.present_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=10, pady=10)

        self.absent_frame = LabelFrame(self.attendance_window, text="Absent", bg='white', padx=10, pady=10)
        self.absent_frame.pack(side=RIGHT, fill=BOTH, expand=True, padx=10, pady=10)

        # Initially display all known labels in the absent list
        for name, img_path in zip(self.known_labels, [img[0] for img in self.images_data]):
            image = np.asarray(Image.open(img_path).convert('RGB'))
            img = Image.fromarray(image)
            img = img.resize((100, 100))
            img = ImageTk.PhotoImage(img)

            item_frame = Frame(self.absent_frame, bg='lightcoral', padx=10, pady=10)
            item_frame.pack(pady=5)

            img_label = Label(item_frame, image=img, padx=10, pady=10)
            img_label.image = img
            img_label.pack(side=LEFT)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            info_label = Label(item_frame, text=f"{name}\n{timestamp}", bg='lightcoral', padx=10, pady=10)
            info_label.pack(side=RIGHT)
    
        # Initialize a dictionary to keep track of the frames
        self.absent_frames = {name: item_frame for name, item_frame in zip(self.known_labels, self.absent_frame.winfo_children())}

        # Start the webcam and processing threads after the window is created
        threading.Thread(target=self.start_camera_and_processing, daemon=True).start()

        self.check_queue()

    
    def start_camera_and_processing(self):
        cap = cv2.VideoCapture(0)
        threading.Thread(target=self.face_recognition.detect_faces_thread, args=(cap, self.known_embeddings, self.known_labels), daemon=True).start()
        threading.Thread(target=self.extract_embeddings_thread, args=(self.known_embeddings, self.known_labels), daemon=True).start()


    def admin_window(self):
        if self.admin_win is not None and self.admin_win.winfo_exists():
            self.admin_win.destroy()

        self.admin_win = Toplevel(self.root)
        self.admin_win.title('ADMIN')
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.admin_win.resizable(True, True)
        self.admin_win.minsize(600, 500)
        self.admin_win.maxsize(screen_width, screen_height)

        top_frame = Frame(self.admin_win)
        top_frame.pack(side=TOP, fill=X)

        # Create search bar
        self.search_entry = Entry(top_frame, width=30)
        self.search_entry.pack(side=LEFT, padx=10, pady=10)

        search_button = Button(top_frame, text="Search", command=self.search_images)
        search_button.pack(side=LEFT, padx=10, pady=10)

        # Create canvas and scrollbar
        self.canvas = Canvas(self.admin_win, bg="white")
        self.canvas.pack(side=LEFT, fill=BOTH, expand=True)
        self.scrollable_frame = Frame(self.canvas, bg="white")
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.scrollbar = Scrollbar(self.admin_win, orient=VERTICAL, command=self.canvas.yview)
        self.scrollbar.pack(side=RIGHT, fill=Y)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        my_menu = Menu(self.admin_win)
        self.admin_win.config(menu=my_menu)

        file_menu = Menu(my_menu)
        my_menu.add_cascade(label='File', menu=file_menu)
        file_menu.add_command(label="Add", command=self.add_file)
        file_menu.add_separator()
        file_menu.add_command(label="Edit", command=self.edit_file)

    def main_window(self):
        admin_button = Button(self.frame, text='Admin', command=self.admin_window)
        admin_button.place(relx=0.5, rely=0.3, anchor=CENTER)

        attendance_button = Button(self.frame, text='Attendance', command=self.attendance_window)
        attendance_button.place(relx=0.5, rely=0.5, anchor=CENTER)

if __name__ == '__main__':
    ui = UI()
    ui.main_window()
    ui.root.mainloop()
    

