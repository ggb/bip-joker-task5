import pandas as pd
import streamlit as st
import altair as alt
import math
import simpledorff
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
st.subheader("by TheLangVerse (Gregor Große-Bölting, Anna Ledworowska and Ismael Cross Moreno)")
st.markdown("##")


col1, col2, col3, col4 = st.columns(4)

col1.metric("Classified wordplayes", count_all_entries)
col2.metric("Number of survey responses", count_resp)
col3.metric("Number of participants", count_users)
col4.metric("English language experience in years", int(df_users["PENGEXP"].sum()))


st.markdown("""
## I. INTRODUCTION

> Humour remains one of the most thorny aspects of intercultural communication. Understanding humour often requires recognition of implicit cultural references or, especially in the case of wordplay, knowledge of word formation processes and discernment of double meanings. These issues raise the question not only of how to translate humour across cultures and languages, but also how to even recognise it in the first place. Such tasks are challenging for humans and computers alike. (Ermakova et al., 2023)

The fact that humour understanding is one of the most difficult application fields of automatic natural language processing is the starting point of the CLEF JOKER track. In the course of dealing with the data sets and trying to solve the tasks, we asked ourselves the question: How good would humans be at these tasks? Because as Ermakova (2023) already states, humour is already a thorny aspect of intercultural communication in general. And if humans already have problems with it, what performance can we expect from machine learning models?

We try to explore these questions in the following. To do this, we created a survey with a random selection of the JOKER training dataset and distributed it to various European universities. In addition to the classification of the puns, we also asked the respondents for the localisation of the pun words and asked further questions in the context of the puns, e.g. about their relevance to the real life. The results once again confirm our expectations: Humour and puns are difficult, not only for machines, but also for humans. This is particularly evident in the low inter-rater reliability of the data set we collected.

## II. THEORETICAL BACKGROUND

In order to contextualise the results, we would first like to discuss research findings related to this work, which will enable us to frame the results below. We will then briefly discuss the nature of puns and wordplays, what makes them special and especially difficult in the machine learning environment — a matter we will return to in the discussion. 

### Related work

In her PhD thesis, Medelyan (2009) investigated how human performance in extracting keyphrases from human scientific texts compares with machine processes, including the Maui alogrithm she developed. Among other things, she finds that human inter-rater reliability varies between 18.5\% and 37.8\%, depending on the rater's expertise and language knowledge in the application domain. They were able to show that the Maui algorithm achieves comparable or (slightly) better performance than human raters in many cases. Building on these results, Große-Bölting et al. (2015) were able to show that the results can be further improved in some cases; Galke et al. (2017) were subsequently able to achieve further improvements through the use of neural networks that are far above the usual performance of human raters in the same task domain.

Blohm et al. (2020) compare the performance of automated machine learning tools (AutoML) with that of human raters for thirteen different publicly available datasets covering a range of different text classification tasks, such as sentiment analysis or identification of fake news. The authors conclude that in most (9 out of 13) cases AutoML is not able to beat human ratings. However, there are cases where this is already possible and the authors see the differences narrowing as machine learning develops. 

Dodge and Karam (2017) discuss another interesting use case for comparing machine and human classification, the evaluation or content detection of images under conditions of visual distortion. The authors note that image recognition by deep neural networks (DNNs) has reached a very advanced stage of development and accordingly achieves performance comparable to humans in most cases. However, the training data is mostly of high quality and has little bias or error. Therefore, the researchers had 15 human raters evaluate corresponding images and compared the results with different DNNs: The results show impressively that the performance without disturbances is the same for humans and DNNs, but with increasing disturbances (noise and blur) the humans show a significantly higher recognition accuracy.

### On puns and wordplays

Puns are figures of speech that use similar-sounding words or phrases with multiple meanings to create a rhetorical effect, be it humorous or serious ([Merriam-Webster](https://www.merriam-webster.com/dictionary/pun), 2023). This can involve causing a word, sentence, or discourse to involve two or more different meanings. Ambiguity, or the presence of more than one possible interpretation or meaning, is central to the concept of puns. There are different types of ambiguity involved in puns, such as lexical ambiguity (when a word has more than one meaning) and syntactic ambiguity (when a sentence can have more than one meaning due to its structure) (Luu, 2015).

From a linguistic point of view, signs consist of two parts: the signifier (the form the word takes) and the signified (the concept it represents). Homonyms occur when a single signifier has multiple signifieds, such as "bat" referring to both a small flying mammal and a piece of sports equipment. In the case of puns, a single signifier can represent multiple signifieds simultaneously, which can be a challenge for the mind to process (Igasheva, 2019).

Puns are not limited to casual conversation or advertising; they can also be found in literature, particularly in poetry throughout history. For example, the first recorded pun in Western literature occurs in the ninth book of the Odyssey, in which the character Polyphemus mistakes the name "Nobody" for the name of the person who has blinded him (Luu, 2015).


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

When evaluating the survey data, incomplete answers were also taken into account, provided that at least one entry was classified. The evaluation of the questions was done with Python ([pandas](https://pandas.pydata.org/docs/index.html), [scikit-learn](https://scikit-learn.org)). Only standard metrics (Precision, Recall, F1) were used for the evaluation. In the case of inter-rater reliability, Krippendorff's alpha (implemented in the python library [simpledorff](https://github.com/LightTag/simpledorff)) was used for the evaluation because it allows calculation over more than two raters and is also suitable for binary classifications (Krippendorff 1970, 2008). Krippendorff's alpha indicates the agreement of the codings with a value between 0 and 1, where 0 means no or a random match, while a value of 1 represents a perfect match (Hayes and Krippendorf, 2007). There are no generally accepted threshold values for a match to be considered good (Krippendorf, 2004). While some authors interpreted values from 0.61 as sufficient or "substantial agreement" (Landis and Koch, 1977), Krippendorff himself calls for values of 0.80 or better and allows values from 0.67 only "tentative conclusions" (Krippendorf 2004).

## IV. RESULTS

Neutral description of the results without interpretation.

### 4.1 Descriptive Results

The descriptive statistics presented below are intended to provide a characterisation of the survey participants. It should be noted in particular that there are only few English native speakers in the survey, but there is a self-reported English language proficiency of 16.7 on average. 

The largest proportions for country of origin and first language are Spain (19) and Germany (17) and Spanish (19) and German (17) respectively. Poland and France account for 10 and 9 participants respectively in terms of country of origin and first language. A few more participants come from the United States, Turkey, Russia, Trinidad and Tobago, and Austria. 

The age distribution shows a wide range, with survey participants between 16 and 89 years old. Almost half (34 out of 73) of the participants are 23 or younger.

""")

tab1, tab2 = st.tabs(["Visualizations", "Raw User Data"])

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

"""
If respondents identified an entry as a pun, this was followed by several more questions aimed at further characterising the pun. The visualizations below summarize the results of these more in-depth questions: 

Only about 14% of respondents had problems understanding an identified pun, although most were previously unknown (only 12% were previously known). Regarding the funniness, opinions are divided: Slightly more than half (52%) found the puns funny. Only a small proportion (5%) were perceived as offensive or objectionable. 26% of the puns were rated in a way that the respondents could imagine using them in real life.
"""

tab1, tab2 = st.tabs(["Visualizations", "Raw Wordplay Data"])

with tab1:
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

with tab2: 
    st.dataframe(df)

st.markdown("""

### 4.2 Human performance

In the following, human performance in the evaluation and localisation of pun words in word games is considered. In addition, the inter-rater reliability with regard to classification is presented.

#### Classification of wordplays

For the classification of the puns, the human raters were shown a random entry from the dataset of previously 100 randomly selected entries from the training dataset and asked the simple question: Is this a wordplay? Only 'yes' and 'no' were available as answer options; unlike all other binary questions, no option was given not to answer this question. The performance of the human raters is as follows: 
""")

# F1, Recall, Precision, Accuracy on the data for classification
col1, col2, col3, col4 = st.columns(4)

col1.metric("F1 Score", round(f1_score(df["class"], df["WCLASS"], average="binary", pos_label="yes"), 2))
col2.metric("Precision", round(precision_score(df["class"], df["WCLASS"], average="binary", pos_label="yes"), 2))
col3.metric("Recall", round(recall_score(df["class"], df["WCLASS"], average="binary", pos_label="yes"), 2))
col4.metric("Accuracy", round(accuracy_score(df["class"], df["WCLASS"]), 2))

st.markdown("""
#### Localizing pun words

The localisation of pun words is a difficult linguistic task. For the following analysis, no further cleaning was done on the data, i.e. an exact match between the test data and the human ratings was required. 

""")

# F1, Recall and Precision on the data for word location
col1, col2, col3, col4 = st.columns(4)

df["location"] = df["location"].str.lower()
col1.metric("F1 Score", round(f1_score(df["location"].astype(str), df["WLOC"].astype(str), average="macro"), 2))
col2.metric("Precision", round(precision_score(df["location"].astype(str), df["WLOC"].astype(str), average="macro"), 2))
col3.metric("Recall", round(recall_score(df["location"].astype(str), df["WLOC"].astype(str), average="macro"), 2))
col4.metric("Accuracy", round(accuracy_score(df["location"].astype(str), df["WLOC"].astype(str).astype(str)), 2))

# Inter rater reliability
irr = simpledorff.calculate_krippendorffs_alpha_for_df(df,experiment_col='WP1',
                                                 annotator_col='CODE',
                                                 class_col='WCLASS')
print(irr)

st.markdown("""
#### Inter-rater reliability

The inter-rater reliability of the human classifiers across the entire data set is 0.19964 (Krippendorff's alpha). This value is far below the values that are commonly accepted as limits of good agreement (see methods section). 

Thus, this value indicates only very low agreement among the human classifiers in the evaluation of wordplays.
""")

st.metric("IRR", round(irr, 3))

st.markdown("""
#### Intra-rater reliability

Only 27 respondents rated the same entry twice, no entry was rated more than twice by a user; the small number does not allow any statement about how large the intra-rater reliability is, so that no calculation was made here. 
""")
#wp_per_user = df.pivot_table(columns=["CODE", "WP1"], aggfunc="size")
#st.dataframe(wp_per_user)

st.markdown("""
## V. DISCUSSION

The evaluation of the human classification of puns shows some interesting results: The F1 score of 0.74 and an accuracy of 0.69 is less high than one would initially expect, suggesting that humans also have problems to some extent in assessing the entries in the dataset. At the same time, the level of agreement among raters is very low. However, given Medelyan's (2009) observations above, it is consistent with expectations regarding untrained, non-native human raters. Of course, it should also be noted that humour is in the eye of the beholder and depends very much on cultural and linguistic circumstances and prior experience. Since people from several European and non-European countries took part in the survey, a variety of assessments can be expected accordingly. Another indication of this is provided by the assessments of the puns: Only a few were known beforehand and opinions regarding their funniness vary widely; moreover, the puns do not seem to be convincing enough for the respondents to adopt them into their own linguistic vocabulary. 

The situation is even more difficult for the localisation task. The low F1 score achieved by human raters can be beaten by a simple machine strategy: choosing the last word of the pun achieves an F1 score of 0.35. A system that erroneously learns to identify as pun word the last word of a pun is thus better at this task than a human. However, this may be a hasty conclusion: a look at the data shows that a not insignificant proportion of respondents completely omitted the pun word when classifying an entry as a pun. So it is probably due in no small part to the nature of surveys that the results do not turn out so well. While a human being has the choice not to give an answer, a machine guesser is always obliged to do so. 

So, overall, what can be expected from machines in terms of recognising puns and humour in general? Not much, or to put it another way: Just as little as from humans. The fact that humour is in the eye of the beholder ensures that it is a chronically difficult field in which even the processing of natural language by algorithms reaches its limits. A model that performs better than humans at this task would possibly and depending on the field of application be useless in actual use. When Alpha Go beat South Korean Lee Sedol, one of the world's top professional Go players, its gameplay was described by [some commentators](https://www.alphagomovie.com/) as alien-like. While such behaviour might be okay in a closed environment like a board game, it would lead to significant problems in social contexts. A socially interacting AI that was too good at understanding and expressing humour could lead to a similar effect, an "[uncanny valley](https://en.wikipedia.org/wiki/Uncanny_valley)" of social interaction.

### Limitations

Although participants from a number of different countries were recruited for this survey and a total of over 500 entries were classified, the greatest limitation of the analysis is the small data base. For further analyses and more reliable statements, a new and more extensive survey would be necessary. Furthermore, not only 100 randomly selected questions should be distributed, but if possible significantly more. 

The questionnaire should be further developed on the basis of the questions collected. For example, it was expressed in personal contact that the translation of the phrases, an aspect that was not used for the above evaluation, was often difficult due to the distance between input and entry on the screen page. Although the intention was to design a questionnaire that could be completed quickly, in reality it proved to take longer than expected to answer: The evaluation of 10 entries and the answering of further questions in the case of identifying puns, proved to be more extensive than hoped.

## VI. CONCLUSION

Participants from various European and non-European countries took part in our survey on the evaluation of puns. The results show once again that the evaluation of humour and puns is not easy — not only for algorithms, but also for humans. Our analysis thus provides another reference point for classifying and evaluating the future development of algorithms in this environment.

## LITERATURE

* Blohm, Matthias, Marc Hanussek, und Maximilien Kintz. 2020. „Leveraging Automated Machine Learning for Text Classification: Evaluation of AutoML Tools and Comparison with Human Performance“.
* Dodge, Samuel, und Lina Karam. 2017. „A Study and Comparison of Human and Deep Learning Recognition Performance under Visual Distortions“. In 2017 26th International Conference on Computer Communication and Networks (ICCCN). IEEE. https://doi.org/10.1109/icccn.2017.8038465.
* Ermakova, Liana, Tristan Miller, Anne-Gwenn Bosser, Victor Manuel Palma Preciado, Grigori Sidorov, und Adam Jatowt. 2023. „Science For Fun: The CLEF 2023 JOKER Track On Automatic Wordplay Analysis“. In Advances in Information Retrieval: 45th European Conference on Information Retrieval, ECIR 2023, Dublin, Ireland, April 2–6, 2023, Proceedings, Part III, 546–56. Berlin, Heidelberg: Springer-Verlag. https://doi.org/10.1007/978-3-031-28241-6_63.
* Galke, Lukas, Florian Mai, Alan Schelten, Dennis Brunsch, und Ansgar Scherp. 2017. „Using Titles vs. Full-Text as Source for Automated Semantic Document Annotation“. In Proceedings of the Knowledge Capture Conference. K-CAP 2017. New York, NY, USA: Association for Computing Machinery. https://doi.org/10.1145/3148011.3148039.
* Große-Bölting, Gregor, Chifumi Nishioka, und Ansgar Scherp. 2015. „A Comparison of Different Strategies for Automated Semantic Document Annotation“. In Proceedings of the 8th International Conference on Knowledge Capture. K-CAP 2015. New York, NY, USA: Association for Computing Machinery. https://doi.org/10.1145/2815833.2815838.
* Hayes, Andrew F, und Klaus Krippendorff. 2007. „Answering the call for a standard reliability measure for coding data“. Communication methods and measures. Taylor & Francis.
* Krippendorff, Klaus. 1970. „Estimating the Reliability, Systematic Error and Random Error of Interval Data“. Educational and Psychological Measurement. https://doi.org/10.1177/001316447003000105.
* ———. 2004. „Reliability in content analysis: Some common misconceptions and recommendations“. Human communication research. Wiley Online Library.
* ———. 2008. „Systematic and Random Disagreement and the Reliability of Nominal Data“. Communication Methods and Measures. https://doi.org/10.1080/19312450802467134.
* Landis, J Richard, und Gary G Koch. 1977. „The measurement of observer agreement for categorical data“. biometrics. JSTOR.
* Luu, Chi. 2015. „Linguistic Anarchy! It’s all Pun and Games Until Somebody Loses a Sign“. JSTOR Daily. https://daily.jstor.org/linguistic-anarchy-pun-games-somebody-loses-sign/.
* Medelyan, Olena. 2009. „Human-competitive automatic topic indexing“. The University of Waikato.
* Sergeevna Igasheva, Anastasiia. 2019. „LINGUISTIC PECULIARITIES OF PUN, ITS TYPOLOGY AND CLASSIFICATION“. In Education, innovation, research as a resource for community development. Publishing house Sreda. https://doi.org/10.31483/r-32974.

""")

