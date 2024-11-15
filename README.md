# Face Recognition App for Autonomous Attendance System

This project is an autonomous attendance system built using face recognition technology. It leverages MTCNN for face detection and FaceNet for face recognition. The app is designed with a Tkinter-based GUI for a seamless user experience and was developed in Visual Studio. It operates without a database, making it lightweight and suitable for small-scale applications such as classrooms or small offices.

## 🚀 Features

- **Face Detection**: Detect faces in real-time using MTCNN.
- **Face Recognition**: Identify registered faces with high accuracy using FaceNet embeddings.
- **Attendance Logging**: Automatically recognize faces and mark attendance.
- **User-Friendly Interface**: Built with Tkinter, offering a simple and intuitive GUI.
- **Standalone Solution**: Operates without a database, ensuring simplicity and ease of setup.

## 🛠 Technologies Used

- **Programming Language**: Python
- **GUI Framework**: Tkinter
- **Machine Learning Models**:
  - MTCNN for face detection
  - FaceNet for face recognition
- **Development Environment**: Visual Studio

## 📚 How It Works

### Face Detection:
- Uses MTCNN to locate faces in images.
- Crops and aligns the detected faces for recognition.

### Face Recognition:
- Generates embeddings for each face using the FaceNet model.
- Matches embeddings with known faces to identify individuals.

### Attendance Logging:
- Once a face is recognized, the app automatically logs the attendance.

### User Interaction:
- The Tkinter-based GUI allows users to register new faces and view attendance logs.

## 🔧 Challenges Faced

- **Model Integration**:
  - Choosing and integrating appropriate models like **FaceNet** and **MTCNN**.
- **Database-Free Design**:
  - Creating an effective system without relying on a database for storage.
- **User Interface**:
  - Building a smooth and intuitive GUI using Tkinter to ensure ease of use.
- **Environment Compatibility**:
  - Ensuring seamless operation within Visual Studio.

## 🚀 Future Improvements

- **Add database integration** for scalable attendance management.
- **Enhance face recognition accuracy** in challenging environments like low light.
- **Expand functionality** for multi-camera support and live video streams.

