!pip install pdf2image
!pip install PyPDF2 pyautogui pytesseract pillow

from PyPDF2 import PdfFileReader, PdfFileWriter
from pdf2image import convert_from_path
from PIL import Image
import pytesseract



def extract_first_page(filename):
    pdf = PdfFileReader(filename)
    pdf_writer = PdfFileWriter()
    pdf_writer.addPage(pdf.getPage(0))

    with open('first_page.pdf', 'wb') as out:
        pdf_writer.write(out)
extract_first_page('your_file.pdf')


def convert_pdf_to_image(pdf_path, output_path, first_page=1, last_page=1):
    images = convert_from_path(pdf_path, first_page=first_page, last_page=last_page)
    for i, image in enumerate(images):
        image.save(f'{output_path}_{i}.png', 'PNG')
convert_pdf_to_image('your_file.pdf', 'output_image')

def ocr_core(filename):
    text = pytesseract.image_to_string(Image.open(filename))
    return text
print(ocr_core('screenshot.png'))

def write_to_txt(text, filename):
    with open(filename, 'w') as f:
        f.write(text)
write_to_txt('your extracted text', 'output.txt')


with open('cleaned_text.txt', mode='w', encoding='utf-8') as file_insert:

