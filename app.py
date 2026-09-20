import streamlit as st
import joblib
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity

support_documents = [
    """
    Billing - Duplicate Charge

    If a customer reports being charged twice for the same transaction,
    verify the transaction history and payment records.

    If a duplicate charge is confirmed, the billing team should initiate
    a refund for the duplicate transaction.

    Refunds may take 5-7 business days to appear in the customer's account.
    """,

    """
    Billing - Failed Payment

    If a customer's payment fails, ask them to verify their payment
    information and ensure sufficient funds are available.

    The customer may retry the payment or use a different payment method.

    If the problem continues, the ticket should be routed to the
    Billing & Payments team.
    """,

    """
    Technical Support - Login Issue

    If a customer cannot log in, first ask them to verify their email
    address and password.

    The customer should try resetting their password using the
    Forgot Password option.

    If the issue continues after a password reset, the ticket should
    be escalated to Technical Support.
    """,

    """
    Technical Support - Website Outage

    If the production website is unavailable for multiple customers,
    treat the issue as a high-priority technical incident.

    The technical team should investigate service availability,
    server health, and recent deployments.

    Widespread outages should be escalated immediately.
    """,

    """
    Refund - Product Return

    Customers requesting a product return should provide their order
    number and reason for the return.

    Eligible products can be returned within 30 days of purchase.

    After the returned product is received and inspected, the refund
    will be processed to the original payment method.
    """,

    """
    Refund - Refund Status

    Customers asking about an existing refund should provide their
    order number or refund reference.

    Approved refunds normally take 5-7 business days to appear in
    the original payment method.

    If the refund has not appeared after this period, the case should
    be reviewed by the Returns & Exchanges team.
    """
]

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()

document_embeddings = embedding_model.encode(
    support_documents
)

def retrieve_top_documents(ticket_text, top_k=2):

    # Convert the customer ticket into an embedding
    ticket_embedding = embedding_model.encode([ticket_text])

    # Compare ticket with all support documents
    similarities = cosine_similarity(
        ticket_embedding,
        document_embeddings
    )[0]

    # Get indices of the most relevant documents
    top_indices = similarities.argsort()[::-1][:top_k]

    results = []

    for index in top_indices:
        results.append({
            "index": int(index),
            "score": float(similarities[index]),
            "document": support_documents[index]
        })

    return results

@st.cache_resource
def load_ml_models():

    category_vectorizer = joblib.load(
        "category_vectorizer.pkl"
    )

    category_model = joblib.load(
        "category_model.pkl"
    )

    urgency_vectorizer = joblib.load(
        "urgency_vectorizer.pkl"
    )

    urgency_model = joblib.load(
        "urgency_model.pkl"
    )

    return (
        category_vectorizer,
        category_model,
        urgency_vectorizer,
        urgency_model
    )


(
    category_vectorizer,
    category_model,
    urgency_vectorizer,
    urgency_model
) = load_ml_models()

def predict_ticket(ticket_text):

    # Category prediction
    category_features = category_vectorizer.transform(
        [ticket_text]
    )

    category = category_model.predict(
        category_features
    )[0]

    # Urgency prediction
    urgency_features = urgency_vectorizer.transform(
        [ticket_text]
    )

    urgency = urgency_model.predict(
        urgency_features
    )[0]

    return {
        "category": category,
        "urgency": urgency
    }

st.set_page_config(
    page_title="AI Support Ticket Copilot - RAG",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Support Ticket Copilot")

st.subheader("ML + RAG Version")

st.write(
    "This version combines ticket classification, "
    "urgency prediction, semantic retrieval, and "
    "LLM-generated support responses."
)

st.success("RAG application environment loaded successfully!")

ticket_text = st.text_area(
    "Enter a support ticket",
    placeholder="Example: Our website is unavailable and customers cannot log in."
)

if st.button("Analyze Ticket"):

    if ticket_text.strip():

        prediction = predict_ticket(ticket_text)

        retrieved_docs = retrieve_top_documents(
            ticket_text,
            top_k=2
        )

        st.subheader("ML Predictions")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Category",
                prediction["category"].title()
            )

        with col2:
            st.metric(
                "Urgency",
                prediction["urgency"].title()
            )

        st.subheader("Retrieved Knowledge")

        for i, doc in enumerate(retrieved_docs, start=1):

            with st.expander(
                f"Document {i} — Similarity: {doc['score']:.3f}"
            ):
                st.write(doc["document"])

    else:
        st.warning("Please enter a support ticket.")
