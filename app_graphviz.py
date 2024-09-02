import os
import pandas as pd
import cv2
from flask import Flask, render_template, request, jsonify, redirect, url_for , send_from_directory
from werkzeug.utils import secure_filename
from pdf2image import convert_from_path
from skimage.metrics import structural_similarity as compare_ssim
from dotenv import load_dotenv
import google.generativeai as genai
from IPython.display import Markdown
import textwrap
from graphviz import Digraph
import graphviz


load_dotenv()

# Configure the generative AI model
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')


app = Flask(__name__)
UPLOAD_FOLDER='uploads'
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['SAVE_FOLDER'] = 'static/'  # Folder to store uploaded files

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

# Check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Convert PDF to images (first page only for simplicity)
def convert_pdf_to_image(pdf_path):
    images = convert_from_path(pdf_path)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_image.png')
    images[0].save(image_path, 'PNG')
    return image_path

# Function to compare images for similarity and return detailed differences
def compare_images_detailed(image_path1, image_path2):
    # Load images
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    # Resize images to the same size for comparison
    image1 = cv2.resize(image1, (640, 480))
    image2 = cv2.resize(image2, (640, 480))

    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between two images
    (score, diff) = compare_ssim(gray1, gray2, full=True)
    diff = (diff * 255).astype("uint8")

    # Threshold the difference image
    _, thresh = cv2.threshold(diff, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours of the differences
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Highlight differences on the images
    image1_diff = image1.copy()
    image2_diff = image2.copy()

    for contour in contours:
        # Compute the bounding box for the contour
        (x, y, w, h) = cv2.boundingRect(contour)
        # Draw a rectangle around differences on both images
        cv2.rectangle(image1_diff, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(image2_diff, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return score, contours, image1_diff, image2_diff

# Route for the home page
@app.route('/pagefive')
def pagefive():
    return render_template('pagefive.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# Route for file upload and processing
@app.route('/result_images', methods=['GET' , 'POST'])
def result_images():
    try:
        if 'file1' not in request.files or 'file2' not in request.files:
            return redirect(request.url)

        file1 = request.files['file1']
        file2 = request.files['file2']

        if file1 and allowed_file(file1.filename) and file2 and allowed_file(file2.filename):
            filename1 = secure_filename(file1.filename)
            filename2 = secure_filename(file2.filename)

            file_path1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
            file_path2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)

            file1.save(file_path1)
            file2.save(file_path2)

            # Convert PDFs to images if necessary
            if filename1.endswith('.pdf'):
                file_path1 = convert_pdf_to_image(file_path1)
            if filename2.endswith('.pdf'):
                file_path2 = convert_pdf_to_image(file_path2)

            # Compare images
            similarity_score, contours, image1_diff, image2_diff = compare_images_detailed(file_path1, file_path2)

            # Save the images with highlighted differences
            diff_image1_filename = 'diff1.png'
            diff_image2_filename = 'diff2.png'
            diff_image1_path = os.path.join(app.config['SAVE_FOLDER'], diff_image1_filename)
            diff_image2_path = os.path.join(app.config['SAVE_FOLDER'], diff_image2_filename)
            cv2.imwrite(diff_image1_path, image1_diff)
            cv2.imwrite(diff_image2_path, image2_diff)

            # Render the results
            return render_template('pagefive.html',
                                   similarity_score=similarity_score,
                                   num_differences=len(contours),
                                   diff_image1=diff_image1_filename,
                                   diff_image2=diff_image2_filename,
                                   differences=[cv2.boundingRect(contour) for contour in contours])

        return redirect(request.url)

    except Exception as e:
        print(f"Error in result_images route: {e}")
        return render_template('pagefive.html', error_message="An error occurred while processing the images."), 500

@app.route('/model_route', methods=['GET', 'POST'])
def model_route():
    results = []
    if request.method == 'POST':
        # Get the uploaded files (multiple)
        uploaded_files = request.files.getlist('data-file')
        
        for uploaded_file in uploaded_files:
            if uploaded_file:
                # Save the file to the upload folder
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
                uploaded_file.save(file_path)
                
                # Process the file
                result = process_file(file_path)
                results.append(result)
    
    return render_template('pagetwo.html', results=results)

def generate_er_diagram():
    try:
        csv_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.csv')]
        dataframes = {}
        primary_keys = {}

        # Read each CSV file into a DataFrame and find potential primary keys
        for file in csv_files:
            df = pd.read_csv(os.path.join(UPLOAD_FOLDER, file))
            dataframes[file] = df
            primary_keys[file] = find_potential_primary_keys(df)

        # Function to check if a DataFrame has any date columns
        def has_date_columns(df):
            date_columns = df.select_dtypes(include=['datetime', 'datetime64', 'timedelta']).columns
            return len(date_columns) > 0

        # Create a Digraph object from graphviz
        dot = Digraph(comment='ER Diagram')

        # Add nodes (tables) and edges (relationships)
        for file in csv_files:
            keys_with_id_suffix = [key for key in primary_keys[file] if key.endswith('id')]
            label = f"{file}\n({', '.join(keys_with_id_suffix)})"
            dot.node(file, label=label, shape='box')

        foreign_keys = {}

        # Find relationships (edges) between tables based on primary keys
        for file, keys in primary_keys.items():
            for key in keys:
                if not key.endswith('id'):
                    continue
                for other_file, other_df in dataframes.items():
                    if file != other_file and key in other_df.columns:
                        dot.edge(file, other_file)
                        if key not in foreign_keys:
                            foreign_keys[key] = []
                        foreign_keys[key].append(other_file)
        print(foreign_keys)
         
        # Dictionary to store table types
        table_types = {}

        # Determine table types
        for file in csv_files:
            if has_date_columns(dataframes[file]):
                table_types[file] = 'Fact'
            else:
                is_fact = True
                for key in primary_keys[file]:
                    if key in foreign_keys:
                        is_fact = False
                        break
                table_types[file] = 'Fact' if is_fact else 'Dimension'

        # Print table types and columns for dimension tables
        dimension_tables = {}
        for file, table_type in table_types.items():
            print(f"Table: {file}")
            print(f"Type: {table_type}")
            if table_type == 'Dimension':
                columns = dataframes[file].columns.tolist()
                dimension_tables[file] = columns
                print(f"Columns: {', '.join(columns)}")
            print("\n")

        fact_tables = {}
        for file, table_type in table_types.items():
            print(f"Table: {file}")
            print(f"Type: {table_type}")
            if table_type == 'Fact':
                columns = dataframes[file].columns.tolist()
                fact_tables[file] = columns
                print(f"Columns: {', '.join(columns)}")
            print("\n")
       
        # Create a graph to represent relationships using Graphviz
        dot = graphviz.Digraph(comment='ER Diagram')

        # Add nodes for each table
        for file in csv_files:
            keys_with_id_suffix = [key for key in primary_keys[file] if key.endswith('id')]
            label = f"{file}\n({table_types[file]})"
            dot.node(file, label=label, shape='box', style='filled', color='lightgray')

        # Add edges for primary key to foreign key relationships
        for key, tables in foreign_keys.items():
            for table in tables:
                for file in primary_keys:
                    if key in primary_keys[file]:
                        dot.edge(file, table, label=key)

        output_path = 'output/er_diagram'
        dot.render(output_path, view=False)
        # Specify paths for saving .dot and .png files
        dot_path = os.path.join(UPLOAD_FOLDER, 'er_diagram')
        png_path = os.path.join(UPLOAD_FOLDER, 'er_diagram')

        # Save the diagram as .dot file
        with open(dot_path, 'w') as f:
            f.write(dot.source)

        # Render the diagram as .png
        dot.render(png_path, format='png')

        return primary_keys, dot_path, png_path

    except Exception as e:
        # Log the exception for debugging
        print(f"Error generating ER diagram: {str(e)}")
        return None, str(e), None



# Function to process the file and return the results
def process_file(file_path):
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            return {"error": "Unsupported file type"}

        primary_keys = find_potential_primary_keys(df)
        candidate_keys = find_candidate_keys(df, primary_keys)
        table_type = classify_table(df, {}, primary_keys)

        return {
            "file": os.path.basename(file_path),
            "primary_keys": primary_keys,
            "candidate_keys": candidate_keys,
            "table_type": table_type
        }

    except Exception as e:
        return {"error": str(e)}

# Reuse the provided functions for data validation and analysis
def validate_dataframe(df, file):
    if df.empty:
        raise ValueError(f"{file} is empty.")
    if df.isnull().values.any():
        raise ValueError(f"{file} contains null values.")
    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]) and not pd.api.types.is_string_dtype(df[column]):
            raise ValueError(f"{file} contains non-numeric and non-string data in column {column}.")
    return df

def find_potential_primary_keys(df):
    primary_keys = [column for column in df.columns if column.lower().endswith('id')]
    if primary_keys:
        return primary_keys
    else:
        return [column for column in df.columns if df[column].nunique() == len(df)]

def find_candidate_keys(df, primary_keys):
    return [column for column in df.columns if df[column].nunique() == len(df) and column not in primary_keys]

def classify_table(df, foreign_keys, primary_keys):
    if foreign_keys:
        return "Fact"

    num_columns = len(df.columns)
    num_records = len(df)
    
    if num_records > 1000 and df.select_dtypes(include=['number']).shape[1] > num_columns / 2:
        return "Fact"

    if not foreign_keys and df.select_dtypes(include=['object', 'category']).shape[1] > num_columns / 2:
        return "Dimension"

    return "Dimension" if primary_keys else "Fact"

@app.route('/start_chat', methods=['POST'])
def start_chat():
    try:
        chat = model.start_chat(history=[])
        return jsonify({'chat_id': chat.id}), 200
    except Exception as e:
        return jsonify({'message': f'Failed to start chat: {str(e)}'}), 500

def to_markdown(text):
    text = text.replace('.', ' *')
    return Markdown(textwrap.indent(text, '--', predicate=lambda _: True))

# Route for converting expressions
@app.route('/convert_expression', methods=['POST'])
def convert_expression():
    data = request.json
    chat_id = data.get("chat_id", "")
    message = data.get("message", "")
    conversion_type = data.get("conversion_type", "")
    try:
        if conversion_type == "DAX to Tableau":
            prompt = f"Convert this DAX expression into Tableau expression. Provide only the Tableau query as output: {message}"
        elif conversion_type == "Tableau to PowerBI":
            prompt = f"Convert this Tableau expression into DAX expression. Provide only the DAX query as output: {message}"
        else:
            return jsonify({'message': 'Invalid conversion type'}), 400

        # Assuming 'model' is already initialized and can call generate_content
        response = model.generate_content(prompt)

        if response.candidates:
            # Extract the DAX expression text from the response
            dax_expression = response.candidates[0].content.parts[0].text.strip()

            # Log the extracted expression
            print(f"Extracted Expression: {dax_expression}")

            # Format it if needed
            formatted_dax = f"\n{dax_expression}\n"

            # Return the formatted expression in the JSON response
            return jsonify({'dax_expression': formatted_dax}), 200
        else:
            return jsonify({'message': 'No valid expression found in response'}), 500

    except Exception as e:
        return jsonify({'message': f'Failed to generate expression: {str(e)}'}), 500

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for the Model Generator page
@app.route('/pagetwo')
def pagetwo():
    return render_template('pagetwo.html')

# Route for the DAX Generator page
@app.route('/pagethree')
def pagethree():
    return render_template('pagethree.html')

# Route for the Report Generator page
@app.route('/pagefour')
def pagefour():
    return render_template('pagefour.html')

if __name__ == '__main__':
    app.run(debug=True)

