# English Experience Distribution

vals = pd.DataFrame(list(df_users["PENGEXP"].value_counts().to_dict().items()),
                    columns=["English experience", "counts"])

c = alt.Chart(vals).mark_bar().encode(
    x = alt.X("English experience:N"),
    y = "counts:Q",
)

col1.altair_chart(c, use_container_width=True)

# Understanding, preknowledge, funnieness, offensivness and life-usage as pies
col1, col2, col3, col4, col5 = st.columns(5)

def create_pie(element,name,coln):
    vals = pd.DataFrame(list(df_users[element].value_counts().to_dict().items()),
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

# F1, Recall and Precision on the data for word location

df["location"] = df["location"].str.lower()
col1.metric("F1 Score", round(f1_score(df["location"].astype(str), df["WLOC"].astype(str), average="macro"), 2))
col2.metric("Precision", round(precision_score(df["location"].astype(str), df["WLOC"].astype(str), average="macro"), 2))
col3.metric("Recall", round(recall_score(df["location"].astype(str), df["WLOC"].astype(str), average="macro"), 2))
col4.metric("Accuracy", round(accuracy_score(df["location"].astype(str), df["WLOC"].astype(str).astype(str)), 2))