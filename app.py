from flask import Flask, render_template, request
import gradio as gr
from threading import Thread
from PIL import Image
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define the Gradio interface
def process_file(file):
    try:
        image_path = os.path.join(UPLOAD_FOLDER, 'image.png')

        image = Image.open(file.name)
        image.save(image_path)
        
        return image_path  # Or you can return the image object if needed

    except Exception as e:
        # If it's not an image, return an error message
        return f"Error: Could not process the file as an image. {str(e)}"


# Use Gradio's Blocks for more flexibility
def create_gradio_app():
    with gr.Blocks() as demo:
        gr.Markdown("## Upload ton poster de film !")
        file_input = gr.File(label="Upload")
        output = gr.Image(label="Preview")
        file_input.change(fn=process_file, inputs=file_input, outputs=output)
    return demo

# Create Gradio app
gradio_app = create_gradio_app()

# Set Gradio port
gradio_app_port = 7861

@app.route("/", methods=['GET', 'POST'])
def home():
    # If the form is submitted, process the image path here (if necessary)
    if request.method == 'POST':
        # File is uploaded and processed by Gradio, get the image URL for further use
        uploaded_file = request.files.get("file")  # Assuming Flask handles a file
        if uploaded_file:
            image_url = process_file(uploaded_file)
            # Here, you can use the `image_url` for further operations
            return f"Image processed. The image is saved at: {image_url}"

    # Pass Gradio iframe URL to the HTML template
    return render_template("index.html", gradio_url=f"http://127.0.0.1:{gradio_app_port}/")

if __name__ == "__main__":
    # Run Gradio app in a separate thread
    def run_gradio():
        gradio_app.launch(server_name="127.0.0.1", server_port=gradio_app_port, share=False)

    gradio_thread = Thread(target=run_gradio)
    gradio_thread.start()
    
    # Run Flask app
    app.run(debug=True, port=5000)

