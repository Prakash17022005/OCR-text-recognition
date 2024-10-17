import easyocr
import cv2
import matplotlib.pyplot as plt
from tkinter import Tk,Text,Scrollbar,END
from tkinter import filedialog
from PIL import Image,ImageTk


# Step 1: Initialize the EasyOCR reader (once at the beginning)
reader = easyocr.Reader(['en'])  # You can set gpu=True if your system supports GPU

# Function to load, resize, and process the image
def process_image(image_path):
    # Step 2: Load the image
    image = cv2.imread(image_path)

    # Step 3: Optionally resize the image to make it smaller (reduce processing time)
    # Resizing to 50% of the original size
    scale_percent = 50  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    # Step 4: Convert the image to RGB format (necessary for EasyOCR)
    image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    

    # Step 5: Perform text recognition
    result = reader.readtext(image_rgb)
    
    #TKinter window
    root=Tk()
    root.title("OCR Text Recognition")
    
    text_box=Text(root,wrap='word',font=("Helvetica",14))
    text_box.pack(expand=True,fill='both')
    
    scrollbar=Scrollbar(root)
    scrollbar.pack(side='right',fill='y')
    text_box.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=text_box.yview)
    
    for (_,text,prob) in result:
        text_box.insert(END,f"{text}\n")
    
    text_box.config(state='normal')
    

    # Step 6: Display the recognized text and confidence
    for (bbox, text, prob) in result:
        print(f"Text: {text}, Confidence: {prob}")

    # Optionally: Draw bounding boxes on the image (only if you want to display it)
    for (bbox, text, prob) in result:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple([int(val) for val in top_left])
        bottom_right = tuple([int(val) for val in bottom_right])
        cv2.rectangle(image_rgb, top_left, bottom_right, (0, 255, 0), 2)

    # Display the processed image with bounding boxes (optional)
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()
    
    root.mainloop()
    
 
# Preprocessing the Image
# Preprocessing the image before feeding it to the OCR model can significantly improve accuracy by making the text clearer.
def preprocess_image(image_path):
        
    image=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    
    image=cv2.GaussianBlur(image,(5,5),0)
    
    image=cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    
    return image

# Example usage (replace 'path_to_image.jpg' with your image path)
image_path=filedialog.askopenfilename(title="Select an image",filetype=[("Image files","*.jpg *.png *.jpeg")])

if image_path:
    
    preprocessed_image = preprocess_image('image_path')
    
    process_image(preprocessed_image)
