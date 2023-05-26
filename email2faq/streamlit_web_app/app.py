from pathlib import Path
import pathlib
import base64
import re
import os
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import sys

temp = pathlib.Path(__file__).parent.resolve()
path1 = os.path.dirname(temp)

sys.path.insert(1, path1)

import fgen

path2 = os.path.join(temp, 'user_data')


# set page layout
st.set_page_config(
    page_title="Email-2-FAQ",
    page_icon=":email:",
    layout="centered",
    initial_sidebar_state="auto",
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

    st.title('Email-2-FAQ :email:')
    st.caption('A web-app to generate FAQs from emails')

    instruction = st.markdown('''
##### How to Use:question:

1. Upload the Emails(_in csv format_) to the upload section in the page.

##### Results (__FGen Framework__):email:

1. __QC__(_Query Classfier_)  : The First text box preprocesses the emails and outputs the queries from all the emails uploaded.
2. __FGG__ (_FAQ Group Generator_) : The second text box clusters similar quiries into groups.
3. __FG__  (_FAQ Generator Subsytem_) : The third and final text box shows Frequently asked questions from the groups generated. 

_Terms:_\n
`Threshold:` The minimum threshold to form clusters(threshild corresponds to the amount of similarity required to consider to 2 sentences in the same cluster) i.e if the threshold is high, it means that clusters will have only a few really similar sentences and this will result in more number of clusters.\n
`Frequency:` The minimum frequency of repetition for the question to be considered in as a Frequent question.\n

''')

    # further instructions for Side Bar
    warning_text = st.sidebar.info(
        'Go to "Project Report" to read more about the app')

    # display the developer information
    about()

       

    # define the fucntion predictions
    def predict(filepath,thres,freq):
        pred = fgen.generate_faq_fgen(filepath,thres,freq)
        # pred = {'valid_queries': ["what are you upto", "is everything ok", "what are different types of laptops available", "what are specifications of each type of laptop"], 'query_clusters': [['is everything ok', 'what will be good specs of the gaming laptop'], ['what are different types of laptops available', 'what is warranty period of laptops']], 'valid_faq': ['what is best gaming laptop?', 'what is the best laptop to buy?']}
        return pred
  

    # for aesthetic purposes
    st.write("-"*34)


    # display the upload image interface
    uploaded_file = st.file_uploader(
        "Choose an file", type=["csv"])
    

    # check for errors in the file
    if uploaded_file is not None:
        try:
            text = uploaded_file.read().decode("utf-8")
            # st.write(text)
            filename = 'input.csv'
            filepath = os.path.join(path2, filename)
            print(filepath, file=sys.stderr) 
            col1, col2 = st.columns([2,3])
            with col1:
                st.write(' ')

            with col2:
                gen_faq = st.button(
                label="Generate FAQ",
                type="primary")

            thres = st.slider("Threshold",min_value = 0.01, max_value = 0.99,value = 0.35)
            freq = st.slider("Frequency",min_value = 1,value = 1)
            if(gen_faq):
                with st.spinner(text=" Getting the predictions.."):
                    with open(filepath,'w') as f:
                        f.write(text)
                    pred = predict(filepath,thres,freq)
                qc_col, fgg_col, faq_col = st.columns(3)
                with qc_col:
                        st.markdown("#### Queries")
                        st.caption("Queries extracted from the emails")
                        st.markdown("---")
                        n = 1
                        for i in pred["valid_queries"]:
                            st.write(f"{n}) {i}")
                            n+=1
                        st.markdown("---")

                with fgg_col:
                    st.markdown("#### Query Clusters")
                    st.caption("Groups of similar queries")
                    st.markdown("---")
                    n = 1
                    for cluster in pred["query_clusters"]:
                        st.markdown(f"##### Cluster {n}")
                        for query in cluster:
                            st.markdown(i)
                        n+=1
                    st.markdown("---")
                
                with faq_col:       
                    st.markdown("### FAQ")
                    st.caption("Frequently asked questions")
                    st.markdown("---")
                    n = 1
                    for i in pred["valid_faq"]:
                        st.write(f"{n}) {i}")
                        n+=1
                    st.markdown("---")




        # display the error message of the format is not a csv format
        except Exception as e:
            print(e)
            st.error('Please upload the data in .csv format!')
       


# defining each page
# This will be the landing page: we will start with project report
def readme():

    success_text = st.sidebar.success('To continue, select "WebApp" ')
    # This is for the  home page
    st.title('Email-2-FAQ')
    st.caption('A web-app to generate FAQs from emails')
    Main_image = st.image('https://static.wixstatic.com/media/0fc399_1053e45158ab4724a1579d7790fdee5d~mv2.gif')
    readme_text = st.markdown(get_file_content_as_string(
        'report/report.md'), unsafe_allow_html=True)

    about()




# This is for the side menu for selecting the sections of the app
st.sidebar.markdown('# M E N U')
page_bg_img = '''
<style>
.stApp{
background-image: url("https://images.unsplash.com/photo-1497091071254-cc9b2ba7c48a?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1174&q=80");
background-size: cover;

}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

page_names_to_funcs = {
    "WebApp": run_app,
    "Project Report": readme,
}


selected_page = st.sidebar.selectbox(
    "Choose the app mode", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()


