import os
from markitdown import MarkItDown

# standard markitdown usage
def convert_pdfs():
    input_dir = './textbook'
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    md_converter = MarkItDown()

    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(input_dir, filename)
            with open(pdf_path, 'rb') as f:
                result = md_converter.convert_stream(f, file_extension=".pdf")
            markdown_text = result.text_content

            output_filename = os.path.splitext(filename)[0] + '.md'
            output_path = os.path.join(output_dir, output_filename)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_text)

            print(f"Saved Markdown: {output_path}")

if __name__ == '__main__':
    convert_pdfs()