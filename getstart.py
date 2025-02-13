import streamlit as st
up_file_boot=st.file_uploader("Upload image of boot space", type=["png","jpeg"])
	photo=st.camera_input("Take a photo")
	if up_file_boot is None and  photo is None:
		print("HI")
	elif photo is None:
		st.image(up_file_boot)
	else:
		st.image(photo)



		options=st.sidebar.radio('Pages', options=['Home', 'Get Started', 'Visualisation'])




	df=pd.read_csv("car.csv")
	x_val=st.selectbox("Select make", options=df['Make'])
	y_val=st.selectbox("Select model", options=df['Model'])