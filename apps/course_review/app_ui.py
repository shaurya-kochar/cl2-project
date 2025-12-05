import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from review_analyzer import analyze_review

st.set_page_config(page_title="IIIT CourseReview Analyzer", layout="wide")

st.title("🎓 IIIT CourseReview Analyzer")
st.markdown("**Classical ML-powered sentiment & emotion analysis for course feedback**")

with st.sidebar:
    st.header("About")
    st.info("""
    This app uses:
    - Logistic Regression
    - Naive Bayes
    - TF-IDF features
    - Multi-label emotion detection
    
    Extension of P20 project
    """)

review_text = st.text_area(
    "Enter course review:",
    height=150,
    placeholder="E.g., The course was excellent! Prof. Smith explained concepts very clearly."
)

if st.button("Analyze Review", type="primary"):
    if review_text.strip():
        with st.spinner("Analyzing..."):
            result = analyze_review(review_text)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Sentiment Analysis")
            sentiment = result['sentiment']['ensemble']
            confidence = result['sentiment']['confidence']
            
            if 'Positive' in sentiment:
                st.success(f"**{sentiment}** ({confidence:.1%} confidence)")
            elif 'Negative' in sentiment:
                st.error(f"**{sentiment}** ({confidence:.1%} confidence)")
            else:
                st.info(f"**{sentiment}** ({confidence:.1%} confidence)")
            
            st.metric("Model Agreement", f"{result['model_comparison']['agreement']:.0f}%")
            
            st.markdown("**Model Predictions:**")
            st.write(f"- LR: {result['sentiment']['lr']}")
            st.write(f"- NB: {result['sentiment']['nb']}")
            st.write(f"- Ensemble: {result['sentiment']['ensemble']}")
        
        with col2:
            st.subheader("😊 Emotion Detection")
            if result['emotions']:
                for emotion in result['emotions']:
                    st.markdown(f"- **{emotion}**")
            else:
                st.write("No strong emotions detected")
            
            st.markdown("---")
            st.subheader("🔍 Model Confidence")
            st.progress(result['model_comparison']['lr_confidence'], text=f"LR: {result['model_comparison']['lr_confidence']:.1%}")
            st.progress(result['model_comparison']['nb_confidence'], text=f"NB: {result['model_comparison']['nb_confidence']:.1%}")
        
        st.markdown("---")
        st.subheader("💡 Interpretability")
        st.write("**Top contributing features** (indices):")
        st.write(f"Positive: {result['feature_importance']['top_positive']}")
        st.write(f"Negative: {result['feature_importance']['top_negative']}")
        
    else:
        st.warning("Please enter a review")

st.markdown("---")
st.caption("Built with Streamlit | Classical ML Text Classification")
