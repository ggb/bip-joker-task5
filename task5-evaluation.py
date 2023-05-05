import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Task 5: Human performance on JOKER wordplay classification",
                    page_icon=":black_joker:",
                    layout="wide"
)

@st.cache_data
def get_excel_data():
    df = pd.read_excel(
        io= 'data/results-survey883364-wordplays.xlsx',
        engine= 'openpyxl',
        sheet_name= 'Sheet1'
    )
    return df

df = get_excel_data()
count_all_entries = len(df)

# --- sidebar ---
st.sidebar.header("Filter here:")

is_complete = st.sidebar.checkbox("Complete responses only")

st.sidebar.markdown("""
---

You can filter the survey participants by the following attributes:
""")

# Age Slider
age_from, age_to = st.sidebar.slider(
    "Age:",
    int(df["PAGE"].min()), int(df["PAGE"].max()), 
    (int(df["PAGE"].min()), int(df["PAGE"].max()))
)

# Experience Slider
exp_from, exp_to = st.sidebar.slider(
    "English experience in years",
    int(df["PENGEXP"].min()), int(df["PENGEXP"].max()), 
    (int(df["PENGEXP"].min()), int(df["PENGEXP"].max()))
)

# First language Multiselect
first_lang = st.sidebar.multiselect(
    "First language",
    options=df["PLANG"].unique(),
    default=df["PLANG"].unique(),
)


querystring = "`lastpage` == 12" if is_complete else "`lastpage` != 0"

df = df.query(
    querystring + " & " + "`PLANG` == @first_lang & `PAGE` >= @age_from & `PAGE` <= @age_to & `PENGEXP` >= @exp_from & `PENGEXP` <= @exp_to"
)

df_resp = df.drop_duplicates(subset=["id"])
count_resp = len(df_resp)

df_users = df_resp.drop_duplicates(subset=["CODE"])
count_users = len(df_users)

# --- Mainpage ---
st.title(":black_joker: Task 5: Human performance on JOKER wordplay classification")
st.markdown("##")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Classified wordplayes", count_all_entries)
col2.metric("Number of survey responses", count_resp)
col3.metric("Number of participants", count_users)
col4.metric("English language experience in years", int(df_users["PENGEXP"].sum()))


st.markdown("""
## I. INTRODUCTION

Why is the survey and it's results interesting?

## II. RELATED WORK

@Ismael: What do we know about wordplays from a linguistic point of view? What *is* a wordplay? 

## III. METHODS

Description of the survey, its questions, runtime, target audience etc.

## IV. RESULTS

Neutral description of the results without interpretation.

### 4.1 Descriptive Results

""")

# will be removed later on
st.dataframe(df_users)

col1, col2, col3 = st.columns(3)

# First language
vals = pd.DataFrame(list(df_users["PLANG"].value_counts().to_dict().items()),
                    columns=["First Language", "counts"])

c = alt.Chart(vals).mark_arc().encode(
    color = alt.X("First Language:N"),
    theta = "counts:Q",
)

col1.altair_chart(c, use_container_width=True)

# Origin
vals = pd.DataFrame(list(df_users["PORG"].value_counts().to_dict().items()),
                    columns=["Origin Country", "counts"])

c = alt.Chart(vals).mark_arc().encode(
    color = alt.X("Origin Country:N"),
    theta = "counts:Q",
)

col2.altair_chart(c, use_container_width=True)

# Age Distribution
vals = pd.DataFrame(list(df_users["PAGE"].value_counts().to_dict().items()),
                    columns=["Age", "counts"])

c = alt.Chart(vals).mark_bar().encode(
    x = alt.X("Age:N"),
    y = "counts:Q",
)

col3.altair_chart(c, use_container_width=True)

# English Experience Distribution

#TODO

# Classifications per wordplay

#TODO

# Understanding, preknowledge, funnieness, offensivness and life-usage as pies

#TODO

st.markdown("""
### Perfomance evaluation

Calculation of performance metrics for the human classification and comparison to machine learning algorithm.
""")

# F1, Recall and Precision on the data for classification

# F1, Recall and Precision on the data for word location

# Inter rater reliability

# Intra rater reliability


st.markdown("""
## Discussion

What do we learn from the results? What is interesting and deserves further investigation? Were are the limitations of our work?
""")

