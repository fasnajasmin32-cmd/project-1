import streamlit as st
import joblib
st.title("Nearest Earth Objects")
st.header("Data Analysis")
est_diameter_min=st.number_input("Enter your est_diameter_min")
est_diameter_max=st.number_input("Enter your est_diameter_max:")
relative_velocity=st.number_input("Enter your relative_velocity:")
miss_distance=st.number_input("Enter your miss_distance:")
orbiting_body=st.selectbox("orbiting_body:",['Earth'])
sentry_object=st.selectbox("sentry_object:",['False'])
absolute_magnitude=st.number_input("Enter your absolute_magnitude:")

classi=joblib.load(r'C:\Users\ASUS\machine learning\NASA-Nearest Earth Objects.pkl')
label1=joblib.load(r'C:\Users\ASUS\machine learning\le1.pkl')
stand=joblib.load(r'C:\Users\ASUS\machine learning\sd.pkl')

orbiting_body=0
sentry_object=0

if st.button("Predict"):
     result=classi.predict(stand.transform([[est_diameter_min, est_diameter_max,relative_velocity,
                                            miss_distance,orbiting_body,sentry_object, absolute_magnitude]]))[0]


     if result==0:
        st.success('False'.format(result))
     else:
        st.success('True'.format(result))                           