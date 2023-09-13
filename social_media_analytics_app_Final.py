import openai
import numpy as np
import pandas as pd
import time, os
import streamlit as st
import concurrent.futures
import json
from tqdm import tqdm_notebook
from tqdm import tqdm 
import re
import math
import plotly.express as px
from PIL import Image
# from streamlit import st.cache


# st.markdown("""
# <div style='text-align: center;'>
# <h2 style='font-size: 50px; font-family: Arial, sans-serif; 
#                    letter-spacing: 2px; text-decoration: none;'>
# <a href='https://affine.ai/' target='_blank' rel='noopener noreferrer'
#                style='background: linear-gradient(45deg, #ed4965, #c05aaf);
#                       -webkit-background-clip: text;
#                       -webkit-text-fill-color: transparent;
#                       text-shadow: none; text-decoration: none;'>
#                       Social Media Analytics
# </a>
# </h2>
# </div>
# """, unsafe_allow_html=True)



# image = Image.open(r"C:\Users\humant\Desktop\logo.png")
# st.image(image, use_column_width=None, width=200, clamp=True)
st.markdown("""
<div style='text-align: center;'>
<h2 style='font-size: 50px; font-family: Arial, sans-serif; 
                   letter-spacing: 2px; text-decoration: none;'>
<a href='https://affine.ai/' target='_blank' rel='noopener noreferrer'
               style='background: linear-gradient(45deg, #ed4965, #c05aaf);
                      -webkit-background-clip: text;
                      -webkit-text-fill-color: transparent;
                      text-shadow: none; text-decoration: none;'>
                      Social Media Analytics
</a>
</h2>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.title("Select The Data File")
    uploaded_file = st.file_uploader("Upload CSV File", type="csv")
    # button = st.button("Upload")


# Set your OpenAI API key here
# openai.api_key = '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'
os.environ["OPENAI_API_KEY"] = 'd2a50a1f0c964d9faf2dbd9dd7777ed7'
openai.api_type = "azure"

openai.api_base = "https://aipractices.openai.azure.com/"

openai.api_version = "2023-03-15-preview"

openai.api_key = os.getenv("OPENAI_API_KEY")

df = pd.DataFrame()

# Modify your analyze function to return a coroutine
def analyze(text, max_tokens=1024, stop=None):
    messages = [
       {"role": "user", "content": "Please extract similar aspects expressions which are present in the example, related segments, related sentiments and overall review sentiment from the following text, text segment should be tagged to any of the related or similar aspect there should not be the case where the text segment is not tagged to any of the aspect and format output in JSON."},
        {"role": "system", "content":"""example 1:
                                                {
                                                  "Review": "The game offers a lot of fun and keeps you engaged, which is a definite plus.Exploring the beautifully designed world is a treat, even though some areas feel lacking in detail. Character's progression adds depth to the experience...The story manages to hold your attention, although it might not be everyone's cup of tea.While it falls short in polish compared to its predecessors, it's still a decent game with potential....but the graphical issues can be frustrating."
                                                  "overall_satisfaction":  "NA",
                                                  "storytelling_and_narrative": 0,
                                                  "gameplay_mechanics_and_fun_factor":  1,
                                                  "open_world_and_exploration": 1,
                                                  "value_and_longevity": "NA",
                                                  "character_development": 1,
                                                  "technical_performance_and_bugs": -1,
                                                  "comparison_to_similar_games_or_prequels": 0,
                                                  "overall_satisfaction_segment": "NA",
                                                  "storytelling_and_narrative_segment": "The story manages to hold your attention, although it might not be everyone's cup of tea.",
                                                  "gameplay_mechanics_and_fun_factor_segment": "The game offers a lot of fun and keeps you engaged, which is a definite plus.",
                                                  "open_world_and_exploration_segment": "Exploring the beautifully designed world is a treat, even though some areas feel lacking in detail.",
                                                  "value_and_longevity_segment": "NA,
                                                  "character_development_segment": "Character's progression adds depth to the experience...",
                                                  "technical_performance_and_bugs_segment": "...but the graphical issues can be frustrating.",
                                                  "comparison_to_similar_games_or_prequels_segment": "While it falls short in polish compared to its predecessors, it's still a decent game with potential.",
                                                  "overall_review_sentiment": "positive"
                                                }

                                                example 2:

                                                {
                                                  "Review": "If youre an RPG fan you owe it to yourself to give The Witcher 3 a shot It rivals the best that Bethesda and Bioware have to offer and youll be hardpressed to find a better RPG this year",
                                                  "overall_satisfaction": "NA",
                                                  "storytelling_and_narrative": 1,
                                                  "gameplay_mechanics_and_fun_factor": "NA,
                                                  "open_world_and_exploration": 1,
                                                  "value_and_longevity": "NA",
                                                  "character_development": "NA",
                                                  "technical_performance_and_bugs": "NA",
                                                  "comparison_to_similar_games_or_prequels": 1,
                                                  "overall_satisfaction_segment": "NA",
                                                  "storytelling_and_narrative_segment": "If youre an RPG fan you owe it to yourself to give The Witcher 3 a shot",
                                                  "gameplay_mechanics_and_fun_factor_segment": "NA",
                                                  "open_world_and_exploration_segment": "It rivals the best that Bethesda and Bioware have to offer",
                                                  "value_and_longevity_segment": "NA",
                                                  "character_development_segment": "NA",
                                                  "technical_performance_and_bugs_segment": "NA",
                                                  "comparison_to_similar_games_or_prequels_segment": "youll be hardpressed to find a better RPG this year",
                                                  "overall_review_sentiment": "positive"
                                                }

                                                example 3:

                                                {
                                                  "Review": "I didnt like this game for a few reasons I didnt like the story and the gameplay was nothing new or special I would much rather play Assassins Creed 		or something The game had some glitches and it actually crashed my PS4",
                                                  "overall_satisfaction": -1,
                                                  "storytelling_and_narrative": -1,
                                                  "gameplay_mechanics_and_fun_factor": -1,
                                                  "open_world_and_exploration": "NA",
                                                  "value_and_longevity": "NA",
                                                  "character_development": "NA",
                                                  "technical_performance_and_bugs": -1,
                                                  "comparison_to_similar_games_or_prequels": -1,
                                                  "overall_satisfaction_segment": "I didnt like this game",
                                                  "storytelling_and_narrative_segment": "I didnt like the story",
                                                  "gameplay_mechanics_and_fun_factor_segment": "the gameplay was nothing new or special",
                                                  "open_world_and_exploration_segment": "NA",
                                                  "value_and_longevity_segment": "NA",
                                                  "character_development_segment": "NA",
                                                  "technical_performance_and_bugs_segment": "The game had some glitches and it actually crashed my PS4",
                                                  "comparison_to_similar_games_or_prequels_segment": "I would much rather play Assassins Creed or something",
                                                  "overall_review_sentiment": "negative"
                                                }
"""},
        {"role": "user", "content": text}
    ]
    # print("I'm here")
    response = openai.ChatCompletion.create(
        engine='chatgpt',
        messages=messages,
        max_tokens=max_tokens,
        stop=stop,
    )
    
    return response

#preprocessing on the reviews to remove unnecessary content like author name, any links
def tweet_preprocessing(review):
# Remove all special characters
    review = re.sub(r'[^a-zA-Z0-9\s]', '', review)

    # Remove user mentions (@username)
    review = re.sub(r'@[a-zA-Z]+', '', review)

    # Remove URLs
    review = re.sub(r'http\S+', '', review)

    # Remove any leading or trailing whitespaces
    review = review.strip()

    # Remove repeated spaces
    review = re.sub(r'\s+', ' ', review)

    return review



if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        df['cleaned_review']=df['text'].apply(tweet_preprocessing)
    
        print("Gykugdycusbc",df.shape)
analysis_results = []
extra_prompts = []


def process_review(index, text):
    
    try:
        res = analyze(
            text=text,
            max_tokens=1024,  # Adjust max_tokens based on your needs
        )
        raw_json = res["choices"][0].message['content'].strip()
    except Exception as e:
        print("Exception1:", e,index)

    try:
        global analysis_results 
        global extra_prompts
        json_data = json.loads(raw_json)
        json_data['index'] = index
        analysis_results.append(json_data)
        # print(analysis_results)
        # log.debug(f"JSON response: {pprint(json_data)}")
        extra_prompts.append(f"\n{text}\n{raw_json}")
        
    except Exception as e:
        # global analysis_results
        print("Exception 2",e,index)
        # log.error(f"Failed to parse '{raw_json}' -> {e}")
        analysis_results.append({'index':index})
        # analysis_results.append([])
# sentiment_data = None
def process_data(df):
    global sentiment_data
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor: 
        for i in range(0,len(df)):  # Adjust the range as needed
            text = df.loc[i, "cleaned_review"]
            executor.submit(process_review, i, text)

    df["analysis"] = analysis_results  
    # df.to_csv("aspects_for_social_media_app.csv", index=False) 
    
    data_list = df.analysis.tolist()
    sentiment_data = pd.DataFrame(data_list)
    sentiment_data.replace('NA', np.nan, inplace=True)
    sentiment_data = sentiment_data.fillna(-3)
    sentiment_data = sentiment_data.sort_values(by=['index'])
    print(sentiment_data)
    return sentiment_data

def create_or_load_final_dataframe():
    if 'final_data' not in st.session_state:
        st.session_state.final_data = process_data(df)
    return st.session_state.final_data

# def create_or_load_final_dataframe():
#     if 'final_data' not in st.session_state:
#         st.session_state.final_data = process_data(df)
#     return st.session_state.final_data
# def create_or_load_final_dataframe():
#     if uploaded_file:
# 
#         # Update the session state variable with the new data
#         st.session_state.final_data = process_data(df)
    
#     return st.session_state.final_data


    
if df.empty:
    centered_html = """
    <div style="display: flex; justify-content: center; align-items: center; height: 10vh;">
        <p style="text-align: center;">Upload csv File!</p>
    </div>
    """

    # Display the centered text
    st.markdown(centered_html, unsafe_allow_html=True)
    # st.markdown('Upload csv File!')
else:
    st.markdown("""
            <style>
                .st-dq {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                }
            </style>
            """, unsafe_allow_html=True)
    print("this runs 1")

    with st.spinner('Wait for it...'):
    
    # with st.form("df_form"):
        sentiment_data = create_or_load_final_dataframe()


        print("this runs 2")
        positive_percentage_overall_satisfaction = math.ceil((sentiment_data['overall_satisfaction'] == 1).sum() / len(sentiment_data) * 100)
        positive_percentage_storytelling_and_narrative = math.ceil((sentiment_data['storytelling_and_narrative'] == 1).sum() / len(sentiment_data) * 100)
        positive_percentage_gameplay_mechanics_and_fun_factor = math.ceil((sentiment_data['gameplay_mechanics_and_fun_factor'] == 1).sum() / len(sentiment_data) * 100)
        positive_percentage_open_world_and_exploration = math.ceil((sentiment_data['open_world_and_exploration'] == 1).sum() / len(sentiment_data) * 100)
        positive_percentage_value_and_longevity = math.ceil((sentiment_data['value_and_longevity'] == 1).sum() / len(sentiment_data) * 100)
        positive_percentage_character_development = math.ceil((sentiment_data['character_development'] == 1).sum() / len(sentiment_data) * 100)
        positive_percentage_technical_performance_and_bugs = math.ceil((sentiment_data['technical_performance_and_bugs'] == 1).sum() / len(sentiment_data) * 100)
        positive_percentage_comparison_to_similar_games_or_prequels = math.ceil((sentiment_data['comparison_to_similar_games_or_prequels'] == 1).sum() / len(sentiment_data) * 100)

        # }
        data = {
                    "Aspect": ["Overall satisfaction", "Storytelling and narrative", "Gameplay mechanics and fun factor", "Open world and exploration", "Value and longevity", "Character development", "Technical performance and bugs", "Comparison to similar games or prequels"],
                    "Positive Percentage": [positive_percentage_overall_satisfaction, positive_percentage_storytelling_and_narrative, positive_percentage_gameplay_mechanics_and_fun_factor, positive_percentage_open_world_and_exploration, positive_percentage_value_and_longevity, positive_percentage_character_development, positive_percentage_technical_performance_and_bugs, positive_percentage_comparison_to_similar_games_or_prequels],
                }


        percentage_data = pd.DataFrame(data).set_index("Aspect").T
        percentage_data.index.name = 'Aspects'
        percentage_data.reset_index(inplace=True)
        
        st.title("Game Review Analysis")
        st.write("Aspects Positive Percentage Ratings:")
        # submitted = st.form_submit_button("Submit")
    # Check if the form has been submitted
        # if submitted:

        st.dataframe(percentage_data.iloc[:,1:], height=50, width=1500,hide_index=True)
        # 
        # st.table(sentiment_data.head())
            # Create a selectbox to choose a review
        
    
    with st.form("review_form"):
        # selected_review = st.selectbox("Select a Review", list(sentiment_data["Review"][:20]))
        selected_review = st.selectbox("Select a Review", list(sentiment_data[(sentiment_data['Review'] != -3)]['Review']))
        
        # Create a form submit button
        submitted = st.form_submit_button("Submit")
    # Check if the form has been submitted
        if submitted:
            # Find the row corresponding to the selected review
            selected_aspect_values = sentiment_data[sentiment_data["Review"] == selected_review]

            # Display aspect values for the selected review
            st.write("Aspect Values for the Selected Review:")
            variables = ['storytelling_and_narrative','gameplay_mechanics_and_fun_factor','open_world_and_exploration','value_and_longevity','character_development','technical_performance_and_bugs','comparison_to_similar_games_or_prequels','overall_review_sentiment']
            for i in variables:
                if i != 'overall_review_sentiment':
                    selected_aspect_values[i] = selected_aspect_values[i].astype(int)
            # selected_aspect_values = selected_aspect_values.loc[:,int_variables.columns].T.reset_index()
            
            # selected_aspect_values = selected_aspect_values.loc[:, selected_aspect_values.columns != 'overall_review_sentiment'].apply(pd.to_numeric)
            selected_aspect_values = selected_aspect_values.loc[:,variables].T.reset_index()
            print(selected_aspect_values)
            
            selected_aspect_values = pd.DataFrame(selected_aspect_values.set_axis(['Aspects', 'Aspects Sentiments'], axis='columns'))
            selected_aspect_values = selected_aspect_values[(selected_aspect_values['Aspects Sentiments'] != -3)]
            # selected_aspect_values = selected_aspect_values['Aspects Sentiments']
            st.dataframe(selected_aspect_values,use_container_width=True,hide_index=True)
            # st.dataframe(selected_aspect_values.iloc[:1,:9].set_index('Review').T,use_container_width=True)
            
            legend = """
               -1 : Negative\n
                0 : Neutral\n
                1 : Positive\n
            """
            # Define CSS style for the legend box
            legend_style = """
                background-color: black;
                color: white;
                padding: 10px;
                border-radius: 5px;
            """

            # Display the legend box with the text
            st.write('<div style="{}">-1 : Negative\n 0 : Neutral\n 1 : Positive\n</div>'.format(legend_style), unsafe_allow_html=True)

            
            # st.markdown(legend)
            
        st.success('Task Completed!')
