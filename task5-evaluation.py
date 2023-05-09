import pandas as pd
import streamlit as st
import altair as alt
import math
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

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
    task5_selection = pd.read_excel(
        io = 'data/task5_survey_selection.xlsx',
        engine= 'openpyxl',
        sheet_name= 'Sheet1'
    )
    return df, task5_selection

df, task5_selection = get_excel_data()
count_all_entries = len(df)

# for native speakers...
def exp_helper(row):
    if row["PLANG"] == "English":
        return row["PAGE"]
    elif math.isnan(row["PENGEXP"]):
        return 0
    else:
        return row["PENGEXP"]

df["PENGEXP"] = df.apply(exp_helper, axis=1)

# add gold standard to each row
def lookup_helper(wp_id, target):
    val = task5_selection[task5_selection["id"] == wp_id]
    return val.iloc[0][target]

df["class"] = df["WP1"].apply(lambda x: lookup_helper(x, "wordplay"))
df["location"] = df["WP1"].apply(lambda x: lookup_helper(x, "location"))

# more prep
df["WCLASS"] = df["WCLASS"].str.lower()
df["WLOC"] = df["WLOC"].str.lower()

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

To determine human performance on the dataset, a survey was created and distributed to universities in different European countries: Gdansk (Poland), Kiel (Germany), Brest (France) and Cadiz (Spain). The survey was available for three weeks from 13.04.2023 to 04.05.2023. 

In order to achieve a good result with the expected number of participants, 100 random entries were selected from the training dataset. The [sample](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html) function of pandas with random state 42 was used for this. Each user of the survey was given another random selection of 10 entries to rate. If an entry was identified as a pun, additional questions were asked conditionally about the character of the pun. A complete list of questions can be found below. 

Users were able to participate in the survey more than once. To do this, respondents were first asked to generate a pseudonymous code that preserved the anonymity of the user but allowed identification across multiple surveys.  Besides the code, only a few questions were asked about the person: Age, mother tongue, country of origin and as a degree of English proficiency, the years the person has been speaking/learning English. These questions can be found in full detail in the list below:

* Self-constructed code for identification between different survey runs (Short free text)
* What's your first language? (List with options and comment)
* If English is not your first language, how many years have you been speaking/learning English? (Numerical input)
* Where do you come from? (List with options and comment)
* How old are you? (Numerical input)
* [For each of 10 randomly select entries: ]
    *  Is this a wordplay? (Yes/No)
    * [In case participant answered 'Yes': ]
        * Which word is most important for the wordplay? (Short free text)
        * Do you understand the wordplay? (Yes/No)
        * Have you heard this wordplay before? (Yes/No)
        * Is the wordplay funny? (Yes/No)
        * Is the wordplay offensive? (Yes/No)
        * Would you use this pun/wordplay in your everyday life? (Yes/No)
        * Please translate the sentence(s)/wordplay into your first language. (Long free text)
        * Do you have other comments on this sentence(s)? (Long free text)

The survey was created with the software LimeSurvey and designed to be GDPR compliant (no recording of IP, data sparing, privacy statement etc.). 

When evaluating the survey data, incomplete answers were also taken into account, provided that at least one entry was classified. The evaluation of the questions was done with Python ([pandas](https://pandas.pydata.org/docs/index.html), [scikit-learn](https://scikit-learn.org)). Only standard metrics (Precision, Recall, F1) were used for the evaluation. In the case of inter-rater reliability, Krippendorff's alpha was used for the evaluation because it allows calculation over more than two raters and is also suitable for binary classifications (Krippendorff 1970, 2008).

## IV. RESULTS

Neutral description of the results without interpretation.

### 4.1 Descriptive Results

The descriptive statistics presented below are intended to provide a characterisation of the survey participants. It should be noted in particular that there are only few English native speakers in the survey, but there is a self-reported English language proficiency of 16.7 on average. 

The largest proportions for country of origin and first language are Spain (19) and Germany (17) and Spanish (19) and German (17) respectively. Poland and France account for 10 and 9 participants respectively in terms of country of origin and first language. A few more participants come from the United States, Turkey, Russia, Trinidad and Tobago, and Austria. 

The age distribution shows a wide range, with survey participants between 16 and 89 years old. Almost half (34 out of 73) of the participants are 23 or younger.

""")

tab1, tab2 = st.tabs(["Visualizations", "Raw Data"])

with tab1:
    col1, col2 = st.columns(2)

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

    col1, col2 = st.columns(2)

    col1.altair_chart(c, use_container_width=True)

    # English Experience Distribution

    vals = pd.DataFrame(list(df_users["PENGEXP"].value_counts().to_dict().items()),
                        columns=["English experience", "counts"])

    c = alt.Chart(vals).mark_bar().encode(
        x = alt.X("English experience:N"),
        y = "counts:Q",
    )

    col2.altair_chart(c, use_container_width=True)


with tab2:
    # will be removed later on
    st.dataframe(df_users)



# Understanding, preknowledge, funnieness, offensivness and life-usage as pies
col1, col2, col3, col4, col5 = st.columns(5)

def create_pie(element,name,coln):
    vals = pd.DataFrame(list(df[element].value_counts().to_dict().items()),
                        columns=[name, "counts"])

    c = alt.Chart(vals).mark_arc().encode(
        color = alt.X(name+":N"),
        theta = "counts:Q",
    )

    coln.altair_chart(c, use_container_width=True)

create_pie("WUNDER","understanding",col1)
create_pie("WKNOWN","preknowledge",col2)
create_pie("WFUNNY","funnieness",col3)
create_pie("WOFFENS","offensivness",col4)
create_pie("WLIFE","life-usage",col5)

st.markdown("""
### Perfomance evaluation

The human raters achieve the following performance when classifying the entries: 
""")

# F1, Recall, Precision, Accuracy on the data for classification
col1, col2, col3, col4 = st.columns(4)

col1.metric("F1 Score", round(f1_score(df["class"], df["WCLASS"], average="binary", pos_label="yes"), 2))
col2.metric("Precision", round(precision_score(df["class"], df["WCLASS"], average="binary", pos_label="yes"), 2))
col3.metric("Recall", round(recall_score(df["class"], df["WCLASS"], average="binary", pos_label="yes"), 2))
col4.metric("Accuracy", round(accuracy_score(df["class"], df["WCLASS"]), 2))

st.markdown("""

""")

# F1, Recall and Precision on the data for word location
col1, col2, col3, col4 = st.columns(4)

df["location"] = df["location"].str.lower()
col1.metric("F1 Score", round(f1_score(df["location"].astype(str), df["WLOC"].astype(str), average="macro"), 2))
col2.metric("Precision", round(precision_score(df["location"].astype(str), df["WLOC"].astype(str), average="macro"), 2))
col3.metric("Recall", round(recall_score(df["location"].astype(str), df["WLOC"].astype(str), average="macro"), 2))
col4.metric("Accuracy", round(accuracy_score(df["location"].astype(str), df["WLOC"].astype(str).astype(str)), 2))

# Inter rater reliability


# Intra rater reliability


st.markdown("""
## Discussion

What do we learn from the results? What is interesting and deserves further investigation? Were are the limitations of our work?
""")

