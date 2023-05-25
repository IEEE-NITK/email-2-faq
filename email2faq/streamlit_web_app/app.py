from pathlib import Path
import base64
import re
import os
import streamlit as st
from PIL import Image
import requests
import random
import pickle
import numpy as np
import io
# from google_images_search import GoogleImagesSearch
import warnings
warnings.filterwarnings('ignore')


# set page layout
st.set_page_config(
    page_title="Email-2-FAQ",
    page_icon=":email:",
    layout="centered",
    initial_sidebar_state="expanded",

)


# Markdown cannot be directly rendered from local machine using st.markdown , we need to use st.image for that and these funcitons below
# parse the markdown file and find the markdown way of mentioning images ![]<> and convert it into html and then it is seen.
def markdown_images(markdown):
    # example image markdown:
    # ![Test image](images/test.png "Alternate text")
    images = re.findall(
        r'(!\[(?P<image_title>[^\]]+)\]\((?P<image_path>[^\)"\s]+)\s*([^\)]*)\))', markdown)
    return images


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def img_to_html(img_path, img_alt):
    img_format = img_path.split(".")[-1]
    img_html = f'<img src="data:image/{img_format.lower()};base64,{img_to_bytes(img_path)}" alt="{img_alt}" style="max-width: 100%;">'

    return img_html


def markdown_insert_images(markdown):
    images = markdown_images(markdown)

    for image in images:
        image_markdown = image[0]
        image_alt = image[1]
        image_path = image[2]
        if os.path.exists(image_path):
            markdown = markdown.replace(
                image_markdown, img_to_html(image_path, image_alt))
    return markdown


def get_file_content_as_string(path):
    with open(path, 'r') as f:
        readme_text = f.read()
    readme_text = markdown_insert_images(readme_text)
    return readme_text


# function to show the developer information
def about():
    st.sidebar.markdown("# A B O U T")
    # st.sidebar.image('profile.png',width=180)
    # st.sidebar.markdown("## Meher")
    # st.sidebar.markdown('* ####  Connect via [LinkedIn]()')
    # st.sidebar.markdown('* ####  Connect via [Github]()')
    # st.sidebar.markdown('* ####  xxxxxxx@gmail.com')
    st.sidebar.markdown("""

### Mentors

- Aryan Amit Barsainyan
- Ashish Bharath
- Amandeep Singh

### Members

- Bharadwaja M Chittrapragada
- Karan Kumar Bhagat
- Mohammad Aadil Shabier
- Tejashree Chaudhari

### [GitHub Repo](https://github.com/IEEE-NITK/email-2-faq)

## Acknowledgements

Special thanks to Pranav DV, Rakshit P, Nishant Nayak and the seniors for guiding and reviewing us during the project.
    """)



# This will be run the app page
def run_app():

    instruction = st.sidebar.markdown('''
##### How to Use:question:

1. Upload the Emails to the upload section in the page.

##### Results (__FGen Framework__):email:

1. __QC__(_Query Classfier_)  : The First text box preprocesses the emails and outputs the queries from all the emails uploaded.
2. __FGG__ (_FAQ Group Generator_) : The second text box clusters similar quiries into groups.
3. __FG__  (_FAQ Generator Subsytem_) : The third and final text box shows Frequently asked questions from the groups generated. 



''')

    # further instructions for Side Bar
    warning_text = st.sidebar.warning(
        'Go to "Project Report" to read more about the app')
    info_text = st.sidebar.info(
        'To see the Experimentation results of all during the project, go to "Experiments Section"')

    # display the developer information
    about()

    # initialize the list to store the images collected through out the session
    # if 'global_image_list' not in st.session_state:
    #     st.session_state.global_image_list=[]

    # define the fucntion to make the class label predictions
    def predict():

        # divider for the asthetics of the page
        st.write("-"*34)

        # compute the total number of images predicted by the user in the session and display it.
        # number_of_images = len(st.session_state.global_image_list)
        # st.write('#### Total products categorized : ', number_of_images)

        # make the predictions
        # st.write('-'*34)
        # if len(st.session_state.predictions) < number_of_images:

        #     #get the last image added to the list of images and pre-process it for prediction
        #     pred_image=st.session_state.global_image_list[-1]
        #     pred_image = pred_image.resize((128, 128))
        #     pred_image = np.expand_dims(pred_image, axis=0)

        #     #make the prediction
        #     pred = st.session_state.model.predict(pred_image,verbose=1)

        #     #make a list of the top 4 most likely categories for the product image
        #     pred_list=[]
        #     sorted_indices=np.argsort(pred[0])
        #     for h in range(4):
        #         pred_index=sorted_indices[-1-h]
        #         predicted_label=st.session_state.classes_list[pred_index]
        #         probability = np.round(pred[0][pred_index]*100,3)
        #         pred_list.append([predicted_label,probability])
        #     st.session_state.predictions.append(pred_list)

        # loop to display all the images categorized in the current session
        # for i in range(number_of_images):

        #     #Display the images in the reverse order so that the latest predictions are at the top
        #     try:
        #         image_ = st.session_state.global_image_list[-1-i]
        #         preds_ = st.session_state.predictions[-1-i]
        #     except Exception as e:
        #         pass

        #     #This code is diaplay the predictions in the tabular format with 3 columns
        #     col1,col2,col3 = st.columns([1.5,2,1])

        #     #display the image in the 1st column
        #     with col1:
        #         if i==0:
        #             st.write("### Image")
        #             st.write("-"*40)
        #         st.image(image_,width=180)

        #     #display the class labels in the 2nd column
        #     with col2:
        #         if i==0:
        #             st.write("### Product Category")
        #             st.write("-"*40)

        #         for g in range(4):
        #             st.write('* ',preds_[g][0].upper())

        #     #display the probability scores in the 3rd column
        #     with col3:
        #         if i==0:
        #             st.write("### Confidence")
        #             st.write("-"*40)

        #         for g in range(4):
        #             st.write('* ', preds_[g][1],' %')

        #     st.write('-'*34)

    # ask for user preference
    st.title('Email-2-FAQ :email:')
    st.caption('A web-app to generate FAQs from emails')
    col1, col2 = st.columns([1,6])
    with col1:
        st.write(' ')

    with col2:
        Main_image = st.image('images/email.jpg', width = 500)
    

    # for aesthetic purposes
    st.write("-"*34)


    # display the upload image interface
    uploaded_file = st.file_uploader(
        "Choose an file", type=["csv"])
    

    # check for errors in the file
    if uploaded_file is not None:
        try:
            # columns to get the tabular format with 2 columns
            col1, col2 = st.columns([1, 4])
            # display the image in the left column
            with col1:
                pass

            with col2:
                button = st.button("Submit")

        # display the error message of the format is not a csv format
        except Exception as e:
            st.error('Please upload the data in .csv format!')


    # Display the status while the model is predicting the class labels
    status = st.spinner(text="getting the predictions..")
    predict()

    # call the prediction method only if the image is uploaded or present
    # clear the status
    # status.empty()


# defining each page

# This will be the landing page: we will start with project report
def readme():

    success_text = st.sidebar.success('To continue, select "Run the app" ')
    info_text = st.sidebar.info(
        'To see the Experimentation results of all during the project, go to "Experiments Section"')    # This is load the markdown page for the entire home page
    # This is for the  home page
    st.title('Email-2-FAQ')
    st.caption('A web-app to generate FAQs from emails')
    Main_image = st.image('images/email.jpg')
    # readme_text = st.markdown(get_file_content_as_string(
    #     'report.md'), unsafe_allow_html=True)

    # success_text = st.sidebar.success('To continue, select "Run the app" ')
    # info_text = st.sidebar.info(
    #     'To see the Benchmarking results of all the models used, go to "Benchmarking Results"')
    # option = st.sidebar.selectbox('',('Project Report','Run the app', 'Benchmarking', 'Source code'))
    about()




def experiments():

    success_text = st.sidebar.success('To continue, select "Run the app" ')
    info_text = st.sidebar.info('Go to "Project Report" to read more about the app')
    # option = st.sidebar.selectbox('',('Project Report','Run the app', 'Benchmarking', 'Source code'))
    about()


# This is for the side menu for selecting the sections of the app
st.sidebar.markdown('# M E N U')


page_names_to_funcs = {
    "WebApp": run_app,
    "Project Report": readme,
    "Experiments": experiments
}


selected_page = st.sidebar.selectbox(
    "Choose the app mode", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()








# #condition, if the user chooses the home page
# if option == 'Show instructions':

#     #alert options for further instructions to proceed
#     success_text=st.sidebar.success('To continue, select "Run the app" ')
#     warning_text=st.sidebar.warning('To see the code, go to "Source code"')

#     #display the developer information
#     # about()

# #condition if the user wishes to see the source code
# if option == 'Source code':

#     #erase the main page contents first
#     Main_image.empty()
#     readme_text.empty()

#     #further instructions
#     success_text=st.sidebar.success('To continue, select "Run the app" ')
#     warning_text=st.sidebar.warning('Go to "Show instructions" to read more about the app')

#     #display the whole sode stored in the text file
#     text_file = open('app_code.txt',mode='r')
#     st.code(text_file.read())
#     text_file.close()

#     #display the developer information
#     about()

# condition if the user chooses to run the app
