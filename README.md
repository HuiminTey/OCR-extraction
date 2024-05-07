# OCR Image to Text Extraction using Tesseract and Google Cloud Vision API

## Project Breakdown

1. Page Segmentation by lines
   - Use OpenCV to find lines between sections
   - Identify coordinates for the lines
2. Segmenting the Image
   - Use the coordinates of the lines to break the image into segments
3. Extracting Text (Printed and Handwritten)
   - Extracting image to text by segments
   - Clean up text and remove unwanted characters
   - Convert extracted values to dataframe
  
## Technical Stack

1. Tesseract
   - Tesseract is useful for converting image to text
   - Free open source python library
   - Cons : unable to correctly detect handwritten text
2. Google Cloud Vision API
   - Performs way better for detecting handwritten text from image
   - Cons : API cost

## Error Handling

1. Extracting text after specific keywords
   - Assumption made that the "aircraft journey summary" follows a fixed tabular key + value format
   - All value that appears after the pre-specified substring(s) is returned and stored

## Deployment

1. EKS
   - Suggested deployment using Docker for containerisation and deploy using Kurbenetes
   - Use the endpoint to receive any incoming input (images) from users
   - Automate the process of image to text extraction and upload to data warehouse like Redshift
2. Monitoring
   - Real time visual monitoring by connecting data warehouse to visualisation playform like Tableu/Quicksight


     
### Example Image
![alt text](https://github.com/HuiminTey/huimintey/blob/main/image/img.png)

### Step 1 - Identifying line coordinates
```
image_path = 'img.png' 

def findHorizontalLines(img):
    img = cv2.imread(img) 
    
    #convert image to greyscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # set threshold to remove background noise
    thresh = cv2.threshold(gray,30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    
    # define rectangle structure (line) to look for: width 100, hight 1. This is a 
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (200,1))
    
    # Find horizontal lines
    lineLocations = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    
    return lineLocations

lineLocations = findHorizontalLines(image_path)
plt.figure(figsize=(24,24))
plt.imshow(lineLocations, cmap='Greys')
```
![alt text](https://github.com/HuiminTey/huimintey/blob/main/image/line_break.png)

### Step 2 - Image segmentation
```
def pageSegmentation1(img, w, df_SegmentLocations):
    img = cv2.imread(img) 
    im2 = img.copy()
    segments = []

    for i in range(len(df_SegmentLocations)):
        y = df_SegmentLocations['SegmentStart'][i]
        h = df_SegmentLocations['Height'][i]

        cropped = im2[y:y + h, 0:w] 
        segments.append(cropped)
        plt.figure(figsize=(8,8))
        plt.imshow(cropped)
        plt.title(str(i+1))        

    return segments

img = image_path
w = lineLocations.shape[1]
segments = pageSegmentation1(img, w, df_SegmentLocations)
```

![alt text](https://github.com/HuiminTey/huimintey/blob/main/image/segment_image.png)

### Step 3 - Extracting Text
```
# Define substring labels
substrings = {
    "Aircraft Model": "Aircraft Model",
    "Registration Number": "Registration Number",
    "Departure Airport": "Departure Airport",
    "Arrival Airport": "Arrival Airport",
    "Crew": "Crew",
    "Fuel": "Fuel",
    "Load": "Load"
}

# Loop through the coordinates
columns = []
values = []

# Only limiting to the tabular section
for i in range(1, len(df_SegmentLocations) - 1):
    y = df_SegmentLocations['SegmentStart'][i]
    h = df_SegmentLocations['Height'][i]
    
    w = lineLocations.shape[1]

    try:
        cropped_image = image[y:y + h, 0:w] 
        response = CloudVisionTextExtractor(cropped_image)
        result = getTextFromVisionResponse(response)
        
        # Remove unwanted text
        cleaned_text = result.replace("☐", "")
        
        # Loop through substrings to find matching substring label
        for substring_label, substring in substrings.items():
            if substring in cleaned_text:
                # Extract text after the substring
                text = cleaned_text.split(substring)[-1].strip()
                columns.append(substring_label)
                values.append(text)
                break  # Break loop once substring is found

    except IndexError:
        print(f"Empty text from Image - skipping")

df = pd.DataFrame([values], columns=columns)
df
```
![alt text](https://github.com/HuiminTey/huimintey/blob/main/image/output.png)

## Repository

Refer to [notebook.md](https://github.com/HuiminTey/huimintey/blob/main/ADE_Assessment.ipynb)



