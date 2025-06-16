#face_recognition code
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

mtcnn = MTCNN(image_size=160, margin=0)
model = InceptionResnetV1(pretrained='vggface2').eval()

def capture_face(frame_input=None):
    if frame_input is None:
        cap = cv2.VideoCapture(0)
        photos = []
        embeddings = []
        count = 0
        while count < 3:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                cap.release()
                break
            cv2.imshow("Press 'c' to capture photo, 'q' to quit", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('c'):
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                photos.append(img)
                print(f"Photo {count+1} captured")
                count += 1

            elif key & 0xFF == ord('q'):
                print("Capture cancelled.")
                break

        cap.release()
        cv2.destroyAllWindows()
        #return photos
        if(len(photos)!=3):
            print("Please capture all three photos")
            return
        for photo in photos:      
            face = mtcnn(photo)
            if face is not None:
                embedding = model(face.unsqueeze(0)).detach().numpy()
                embeddings.append(embedding)
                return embeddings
            else:
                print("Face not found in one of the photos, registration failed")            
                return
    else:
        img = Image.fromarray(cv2.cvtColor(frame_input, cv2.COLOR_BGR2RGB))
        face = mtcnn(img)
        if face is not None:
            return model(face.unsqueeze(0)).detach().numpy()[0].astype('float32')
        else:
            return None





