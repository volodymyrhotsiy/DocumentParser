import pytesseract
from PIL import Image
import cv2
import numpy as np
import os
from pdf2image import convert_from_path
from pypdf import PdfReader
import openai
import requests

def chatGPT(user_input,text):
    openai.api_key = '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'

    URL = "https://api.openai.com/v1/chat/completions"

    payload = {
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": user_input + text}],
    "temperature" : 1.0,
    "top_p":1.0,
    "n" : 1,
    "stream": False,
    "presence_penalty":0,
    "frequency_penalty":0,
    }

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {openai.api_key}"
    }

    response = requests.post(URL, headers=headers, json=payload, stream=False)
    return response.content

def extract_text_from_pdf(file_path):
    output_text = ''
    reader = PdfReader(file_path)
    for page in reader.pages:
        output_text += page.extract_text()
    return output_text   

def pdf_to_png(input_path, output_dir=r'C:\Users\HOME1\DocumentParser\tempoPNG'):
    """Converts a PDF file to a series of PNG images. 

    Args:
        input_path (str): The path to the input PDF file.
        output_dir (str): The path to the output directory to save the PNG images.

    Returns:
        List[str]: A list of paths to the PNG images.

    Raises:
        Exception: If there is an error during the conversion process.

    """
    try:
        # Use pdf2image to convert the PDF to a series of PIL images
        images = convert_from_path(input_path, poppler_path=r'C:\Users\HOME1\OneDrive\Desktop\poppler-23.01.0\Library\bin')

        # Save each image as a PNG file in the output directory
        png_paths = []
        for i, image in enumerate(images):
            png_path = os.path.join(output_dir, f"page_{i}.png")
            image.save(png_path, "PNG")
            png_paths.append(png_path)

        return png_paths

    except Exception as e:
        raise Exception(f"Error converting PDF to PNG: {e}")

def extract_text_from_png(png_path):
    # Load the PNG image using PIL
    image = Image.open(png_path)
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    # Preprocess the image
    image = image.resize((image.width * 3, image.height * 3))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    image = cv2.medianBlur(image, 3)
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Use pytesseract to extract text from the image
    text = pytesseract.image_to_string(
        image, lang='eng', config='--psm 6'
    )

    return text

def extract_text_from_all_pngs(directory=r'C:\Users\HOME1\DocumentParser\tempoPNG'):
    """Extracts text from all PNG images in a specified directory and appends the text to a single output string.

    Args:
        directory (str): The path to the directory containing the PNG images.

    Returns:
        A string containing the extracted text from all the PNG files in the directory.

    """
    output_text = ""

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is a PNG
        if filename.endswith(".png"):
            # Extract text from the PNG using the extract_text_from_png function
            png_path = os.path.join(directory, filename)
            text = extract_text_from_png(png_path)

            # Append the extracted text to the output string
            output_text += text

    return output_text

def extraxt_text_from_pdf_scan(pdf_path):
    pdf_to_png(pdf_path)
    return extract_text_from_all_pngs()

def process_file(file_path):
    file_type = file_path.split(".")[-1]
    if file_type == "pdf":
        return extract_text_from_pdf(file_path)
    elif file_type == "png":
        return extract_text_from_png(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    

    

