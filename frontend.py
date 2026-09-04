import streamlit as st;
from model import predictSpam,classifyEmail,get_stats,get_confusion_matrix_data

st.set_page_config(page_title="Spam Filter", layout="wide")
st.sidebar.title("Project Navigation")
page=st.sidebar.radio("Go to:",["Spam Classifier","Model Statistics"])

if page=="Spam Classifier":
    st.title("Spam Classifier")
    tab1, tab2 = st.tabs(["SMS Classifier", "Email Classifier"])
    with tab1:
        st.subheader("Analyze SMS")
        user_input=st.text_area("Enter your message below",placeholder="Type here... (e.g., You won the lottery )",height=150)
        if st.button("Check"):
            if(user_input.strip()==""):
                st.warning("Enter some message!")
            else:
                with st.spinner("Analyzing...."):
                    result=predictSpam(user_input)
                st.subheader("Result:")
                if result["label"] == "SPAM":
                    st.error(f"🚨 This looks like {result['label']}! (Confidence: {result['confidence']})")
                else:
                    st.success(f"✅ This looks like {result['label']}!")
    with tab2:
        st.subheader("Analyze Email")
        sender = st.text_input("Sender Email Address", placeholder="Type here...")
        subject = st.text_input("Email Subject", placeholder="Type here....")
        body = st.text_area("Email Body", placeholder="Paste the full email content here...", height=250)
        if st.button("Check Email"):
            if body.strip() == "":
                st.warning("Please provide the email body!")
            else:
                st.info("Analyzing")
                result=predictSpam(body)
                st.subheader("Result:")
                if result["label"] == "SPAM":
                    st.error(f"🚨 This looks like {result['label']}! (Confidence: {result['confidence']})")
                else:
                    st.success(f"✅ This looks like {result['label']}!")
elif page=="Model Statistics":
    st.title("Model Statistics")
    stats = get_stats()
    st.subheader("Current Performance Metrics")
    st.metric("Overall Accuracy", round(stats["accuracy"]*100,2))
    st.metric("Precision", round(stats["precision"]*100,2))
    st.metric("Recall", round(stats["recall"]*100,2))

    data = get_confusion_matrix_data() # [TN, FP, FN, TP]
    labels = ['True Negative (Correct Ham)', 'False Positive (Ham as Spam)', 
              'False Negative (Spam missed)', 'True Positive (Correct Spam)']
    
    
    