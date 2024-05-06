# OCR Image to Text Extraction using Tesseract and Google Cloud Vision API

<details>
    <summary>
        <b>
            <font size="+2">
                Overview
            </font>
        </b>
    </summary>

1. Page Segmentation by lines
   - Use OpenCV to find lines between sections
   - Identify coordinates for the lines
2. Segmenting the Image
   - Use the coordinates of the lines to break the image into segments
3. Extracting Text (Printed and Handwritten)
   - Extracting image to text by segments
   - Clean up text and remove unwanted characters
   - Convert extracted values to dataframe


