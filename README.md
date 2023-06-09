# Welcome to Older or Newer

This is a web app using Flask and Python that allows users to apply a Logistic Regression model to datasets of web app features to see what words are most predictive of whether a feature is older or newer.

## How to use (Note: this requires Python 3, possibly 3.8 or higher)

1. Clone this repo
2. Create a virtual environment by running `python -m venv venv` in the command line (in the root directory of the repo)
3. Start the virtual environment
   1. In the command line, run `venv\Scripts\activate` to start the virtual environment (Windows) or `. venv/bin/activate` to start the virtual environment (Mac)
4. Install the required packages by running `pip install -r requirements.txt` in the venv terminal (this may take awhile)
5. Start the flask app by running `flask --app flaskapp run` in the venv terminal
   1. The first run might take a little longer, as NLTK will need to download some data for preprocessing of the text
6. Open the app in your browser by visiting the URL `http://127.0.0.1:5000/`

At any time, the flask app can be stopped by pressing `Ctrl+C` in the venv terminal.

The virtual environment can be stopped by running `deactivate` in the venv terminal.
