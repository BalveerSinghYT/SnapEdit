import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2

# Set the page configuration
st.set_page_config(
    page_title="SnapEdit",
    page_icon=":camera:",
    layout="wide",
)

# Columns for displaying original and processed images
col1, col2 = st.columns(2)

def display_images(original_img, processed_img):
    """
    Function to display original and processed images side-by-side.
    """
    with col1:
        st.text("Original Image")
        st.image(original_img)

    with col2:
        st.text("Processed Image")
        st.image(processed_img)


def process_image(image, enhance_type, value=None, edge_enhance=False):
    """
    Function to process the image based on the enhancement type and value.
    """
    if enhance_type == 'Gray-Scale':
        img_array = np.array(image.convert('RGB'))
        gray_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        return gray_img

    elif enhance_type == 'Contrast':
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(value)

    elif enhance_type == 'Brightness':
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(value)

    elif enhance_type == 'Blurring':
        img_array = np.array(image.convert('RGB'))
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        blurred_img = cv2.GaussianBlur(img_bgr, (11, 11), value)
        return Image.fromarray(cv2.cvtColor(blurred_img, cv2.COLOR_BGR2RGB))

    elif enhance_type == 'Sharpness':
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(value)

    elif enhance_type == 'Edge Enhance' or edge_enhance:
        return image.filter(ImageFilter.EDGE_ENHANCE)

    return image


def main():
    """
    Main function for the Image Processing Operations web app.
    """
    st.title("Python Image Processing with Streamlit")
    modes = ["Noob Mode", "Pro Mode", "About"]
    selected_mode = st.sidebar.selectbox("Select the Mode", modes)

    st.subheader("Upload Image")
    image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

    if selected_mode == 'Noob Mode' and image_file is not None:
        enhance_type = st.sidebar.radio("Enhance Type", ["Original", "Gray-Scale", "Contrast", "Brightness", "Blurring", "Sharpness", "Edge Enhance"])
        image = Image.open(image_file)
        
        if enhance_type == 'Original':
            processed_img = image
        elif enhance_type in ['Contrast', 'Brightness', 'Blurring', 'Sharpness']:
            value = st.sidebar.slider(f"{enhance_type}", 0.5, 3.5)
            processed_img = process_image(image, enhance_type, value)
        else:
            processed_img = process_image(image, enhance_type)

        display_images(image, processed_img)

    elif selected_mode == 'Pro Mode' and image_file is not None:
        image = Image.open(image_file)

        brightness = st.sidebar.slider("Brightness", 0.5, 3.5, 1.0)
        contrast = st.sidebar.slider("Contrast", 0.5, 3.5, 1.0)
        sharpness = st.sidebar.slider("Sharpness", 0.5, 3.5, 1.0)
        blurring = st.sidebar.slider("Blurring", 0.5, 3.5, 1.0)
        edge_enhance = st.sidebar.checkbox("Edge Enhance")

        # Apply enhancements
        processed_img = process_image(image, 'Brightness', brightness)
        processed_img = process_image(processed_img, 'Contrast', contrast)
        processed_img = process_image(processed_img, 'Sharpness', sharpness)
        processed_img = process_image(processed_img, 'Blurring', blurring)
        if edge_enhance:
            processed_img = process_image(processed_img, 'Edge Enhance', edge_enhance=True)

        display_images(image, processed_img)

    elif selected_mode == 'About':
        st.subheader("About")
        st.markdown("This is a simple Image Processing Streamlit web app. Built with OpenCV and PIL.")
        st.text("Built by Balveer Singh")
        st.markdown("[Visit my Website](https://balveersingh.in)")

if __name__ == '__main__':
    main()
