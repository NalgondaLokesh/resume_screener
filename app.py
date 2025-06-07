import nltk
import re
import pickle
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import PyPDF2
import docx

nltk.download('punkt')
nltk.download('stopwords')
clf = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('le.pkl', 'rb'))

def clean_text(text):
  text = re.sub('http\S+\s*', ' ', text)  # Remove URLs
  text = re.sub('RT|cc', ' ', text)  # Remove RT and cc
  text = re.sub('#\S+', '', text)  # Remove hashtags
  text = re.sub('@\S+', '  ', text)  # Remove mentions
  text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)  # Remove punctuations
  text = re.sub(r'[^\x00-\x7f]',r' ', text) # Remove non-ASCII characters
  text = re.sub('\s+', ' ', text)  # Remove extra whitespace
  return text

#Function to predict the category of the resume
def pred(input_resume):
  cleaned_text = clean_text(input_resume) # Changed cleanResume to clean_text
  vectorized_text = tfidf.transform([cleaned_text])
  vectorized_text = vectorized_text.toarray()
  predicted_category = clf.predict(vectorized_text)
  prediction = le.inverse_transform(predicted_category)
  return prediction[0]

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

# Function to extract text from TXT with explicit encoding handling
def extract_text_from_txt(file):
    # Try using utf-8 encoding for reading the text file
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        # In case utf-8 fails, try 'latin-1' encoding as a fallback
        text = file.read().decode('latin-1')
    return text

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
    return text



#web app
def main():
    st.set_page_config(page_title="Resume Screening App", page_icon="📝", layout="centered")
    st.markdown(
        """
        <style>
        .main-header {
            text-align: center;
            color: #fff;
            background: linear-gradient(90deg, #4F8BF9 0%, #1CB5E0 100%);
            padding: 30px 0 10px 0;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        .footer {
            text-align: center;
            color: #888;
            margin-top: 40px;
        }
        .category-box {
            background: linear-gradient(90deg, #D6EAF8 0%, #AED6F1 100%);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            margin-top: 20px;
            box-shadow: 0 2px 8px rgba(79,139,249,0.08);
        }
        .expander-header {
            color: #154360;
        }
        </style>
        <div class='main-header'>
            <h1>📝 Resume Screening App</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/resume.png", width=100)
        st.markdown("## <span style='color:#4F8BF9;'>About</span>", unsafe_allow_html=True)
        st.info(
            "Upload a resume in **PDF**, **TXT**, or **DOCX** format and get the predicted job category instantly!"
        )
        st.markdown("---")
        st.markdown("**A project by <span style='color:#1CB5E0;'>Nalgonda Lokesh</span>**", unsafe_allow_html=True)

    st.markdown(
        "<p style='text-align: center; color: #1CB5E0; font-size: 18px;'>Select your resume file below and click to see the prediction.</p>",
        unsafe_allow_html=True,
    )

    upload_file = st.file_uploader(
        "📄 Upload your resume", type=["pdf", "docx", "txt"], help="Supported formats: PDF, DOCX, TXT"
    )

    if upload_file is not None:
        with st.spinner("Extracting and analyzing your resume..."):
            try:
                resume_text = handle_file_upload(upload_file)
                st.success("✅ Successfully extracted the text from the uploaded resume.")

                with st.expander("🔍 View Extracted Resume Text", expanded=True):
                    st.markdown(resume_text[:2000] + ("..." if len(resume_text) > 2000 else ""), unsafe_allow_html=True)
                st.markdown("<h3 style='color:#4F8BF9;'>🎯 Predicted Category</h3>", unsafe_allow_html=True)
                category = pred(resume_text)
                st.markdown(
                    f"<div class='category-box'><h2 style='color: #154360;'>{category}</h2></div>",
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"❌ An error occurred while processing the file: {e}")
                st.warning("Please ensure the file is a valid PDF, DOCX, or TXT file and try again.")

    st.markdown(
        "<div class='footer'><hr><p>© 2025 <span style='color:#4F8BF9;'>Resume Screener</span> | Powered by <span style='color:#1CB5E0;'>Streamlit</span></p></div>",
        unsafe_allow_html=True,
    )

if __name__ == '__main__':
    main()
