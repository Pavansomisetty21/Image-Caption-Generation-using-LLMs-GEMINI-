import streamlit as st
import warnings
import google.generativeai as genai
from PIL import Image

# Constants for text styling
BOLD_BEGIN = "\033[1m"
BOLD_END = "\033[0m"

# System content for the generative model
system_content = """You are an expert analyzing images and provide accurate descriptions.
You do not make descriptions."""

# Set the API key and Gemini model name directly in the code
api_key = "your api key"  # Replace with your actual API key
model_name = "gemini-1.5-flash"

# Ensure the API key is set
if not api_key:
    raise ValueError("API_KEY must be set.")

# Configure the generative AI client
genai.configure(api_key=api_key)

# ClientFactory class to manage API clients
class ClientFactory:
    def __init__(self):
        self.clients = {}
    
    def register_client(self, name, client_class):
        self.clients[name] = client_class
    
    def create_client(self, name, **kwargs):
        client_class = self.clients.get(name)
        if client_class:
            return client_class(**kwargs)
        raise ValueError(f"Client '{name}' is not registered.")

# Register and create the Google generative AI client
client_factory = ClientFactory()
client_factory.register_client('google', genai.GenerativeModel)

client_kwargs = {
    "model_name": model_name,
    "generation_config": {"temperature": 0.8},
    "system_instruction": None,
}

client = client_factory.create_client('google', **client_kwargs)

# User content for image description
user_content = """Describe this picture, landscape, buildings, country, settings, and art style if any dictated. 
                  Identify any signs and indicate what they may suggest."""

# Streamlit app
st.title("Image Description with Google Gemini")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # Generate image description
    st.write("Generating description...")
    response = client.generate_content([user_content, image], stream=True)
    response.resolve()
    st.write(f"Image description: {response.text}")
