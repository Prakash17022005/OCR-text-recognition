import easyocr
import cv2
import matplotlib.pyplot as plt
from tkinter import Tk, Text, Scrollbar, END
from tkinter import filedialog
from PIL import Image, ImageTk


# Step 1: Initialize the EasyOCR reader (once at the beginning)
reader = easyocr.Reader(['en'])  # You can set gpu=True if your system supports GPU

# Function to load, resize, and process the image
def process_image(image_rgb):  # Take preprocessed image directly
    # Step 5: Perform text recognition on the preprocessed image
    result = reader.readtext(image_rgb)

    # TKinter window setup
    root = Tk()
    root.title("OCR Text Recognition")
    
    # Set up the text box for displaying OCR results
    text_box = Text(root, wrap='word', font=("Helvetica", 14))
    text_box.pack(expand=True, fill='both')
    
    # Add a scrollbar to the text box
    scrollbar = Scrollbar(root)
    scrollbar.pack(side='right', fill='y')
    text_box.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=text_box.yview)
    
    # Insert recognized text into the text box
    for (_, text, prob) in result:
        text_box.insert(END, f"{text}\n")
    
    text_box.config(state='normal')
    
    # Step 6: Display the recognized text and confidence
    for (bbox, text, prob) in result:
        print(f"Text: {text}, Confidence: {prob}")
    
    # Optionally: Draw bounding boxes on the image (if you want to display it)
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

# Preprocessing the Image for better OCR results
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    
    # Apply Gaussian Blur to reduce noise
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Apply Adaptive Thresholding to binarize the image for better OCR clarity
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Optionally resize the image for faster processing
    scale_percent = 50  # Resize to 50% of the original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    # Convert to RGB for EasyOCR
    image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)  # Convert from grayscale to RGB
    
    return image_rgb

# Select the image file and process it
image_path = filedialog.askopenfilename(title="Select an image", filetype=[("Image files", "*.jpg *.png *.jpeg")])

if image_path:
    # Preprocess the selected image
    preprocessed_image = preprocess_image(image_path)
    
    # Perform OCR on the preprocessed image
    process_image(preprocessed_image)
