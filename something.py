import cv2  
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image: Image.Image) -> Image.Image:
    """Rotates image to the orientation with the most detected faces using OpenCV."""
    rotations = [0, 90, 180, 270]
    max_faces = 0
    best_rotation = 0
    best_image = image

    for angle in rotations:
        # Rotate image
        rotated_img = image.rotate(-angle, expand=True)

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(np.array(rotated_img), cv2.COLOR_RGB2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        face_count = len(faces)

        if face_count > max_faces:
            max_faces = face_count
            best_rotation = angle
            best_image = rotated_img

    print(f"Best rotation based on face detection: {best_rotation}° ({max_faces} face(s) detected)")
    return best_image

image = detect_faces(image)