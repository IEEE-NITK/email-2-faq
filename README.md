# Email-2-FAQ

<img src="https://img.shields.io/github/license/IEEE-NITK/email-2-faq"> <img src="https://img.shields.io/github/languages/top/IEEE-NITK/email-2-faq"> <img src="https://img.shields.io/github/issues/IEEE-NITK/email-2-faq"> <img src="https://img.shields.io/github/issues-pr/IEEE-NITK/email-2-faq"> <img src="https://img.shields.io/github/last-commit/IEEE-NITK/email-2-faq">

Email-2-FAQ is a web application that generates FAQs from emails. It uses a deep learning based end-to-end system (F-Gen) for automated email FAQ generation. The system is trained on a dataset of 1000 emails and their corresponding FAQs. The system is able to generate FAQs from emails with an accuracy of 0.85.

Emails have become essential information as businesses exchange enormous volumes of formal emails every day. The automatic creation of FAQs from email systems aids in identifying crucial information and may be helpful for future chatbots and intelligent email answering applications.

A few implementations use recently discovered deep learning techniques to retrieve FAQs from emails, even though there is research in the literature that concentrates on automatic FAQ development and automated email replying. With cutting-edge techniques, this solution seeks to make use of the unique framework F-Gen, an expert system that creates possible FAQs from emails.

We also aim to develop this product into a deployable service that companies and websites can use to generate FAQs from their thousands of emails. We also aim to use techniques such as sentiment analysis to generate automated feedback management and classification.

<br>

## Installation
To contribute and work on the repository, you need Python installed on your system. If you do not have Python installed, you can install it from [here](https://www.python.org/downloads/).

Fork and clone the repository from GitHub.
```bash
git clone https://github.com/<your-username-here>/email-2-faq.git
```

Traverse to the directory where the repository is cloned.
```bash
cd email-2-faq
```

To execute the script, you will need to install the dependencies. It is recommended to create a virtual environment to do the same
```bash
# Create a virtual environment (not necessary but recommended)
python3 -m venv <name-of-virtual-environment>
source <name-of-virtual-environment>/bin/activate

# Install the dependencies
pip install -r requirements.txt
```

<br>

### Web Application
Open a dedicated terminal to host the Flask server and run the following commands:
```bash
cd src
python web/app.py
```
The web application will be hosted on `http://localhost:5000/`

<br>

### References
Shiney Jeyaraj, Raghuveera T.,
*A deep learning based end-to-end system (F-Gen) for automated email FAQ generation*,
Expert Systems with Applications,
Volume 187,
2022,
115896,
ISSN 0957-4174,
https://doi.org/10.1016/j.eswa.2021.115896.
(https://www.sciencedirect.com/science/article/pii/S0957417421012525)
