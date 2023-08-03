import io
import librosa
import numpy as np
import streamlit as st
import audiomentations
from matplotlib import pyplot as plt
import librosa.display
from scipy.io import wavfile
import pydub
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


plt.rcParams["figure.figsize"] = (10, 7)




def create_audio_player(audio_data, sample_rate):
    virtualfile = io.BytesIO()
    wavfile.write(virtualfile, rate=sample_rate, data=audio_data)

    return virtualfile


@st.cache
def handle_uploaded_audio_file(uploaded_file):
    a = pydub.AudioSegment.from_file(
        file=uploaded_file, format=uploaded_file.name.split(".")[-1]
    )

    channel_sounds = a.split_to_mono()
    samples = [s.get_array_of_samples() for s in channel_sounds]

    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples[0].typecode).max

    return fp_arr[:, 0], a.frame_rate


def plot_wave(y, sr):
    fig, ax = plt.subplots()

    img = librosa.display.waveshow(y, sr=sr, x_axis="time", ax=ax)

    return plt.gcf()


def plot_transformation(y, sr, transformation_name):
    D = librosa.stft(y)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
    ax.set(title=transformation_name)
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    return plt.gcf()


def spacing():
    st.markdown("<br></br>", unsafe_allow_html=True)


def plot_audio_transformations(y, sr, pipeline: audiomentations.Compose):
    cols = [1, 1, 1]

    col1, col2, col3 = st.columns(cols)
    with col1:
        st.markdown(
            f"<h4 style='text-align: center; color: black;'>Original</h5>",
            unsafe_allow_html=True,
        )
        st.pyplot(plot_transformation(y, sr, "Original"))
    with col2:
        st.markdown(
            f"<h4 style='text-align: center; color: black;'>Wave plot </h5>",
            unsafe_allow_html=True,
        )
        st.pyplot(plot_wave(y, sr))
    with col3:
        st.markdown(
            f"<h4 style='text-align: center; color: black;'>Audio</h5>",
            unsafe_allow_html=True,
        )
        spacing()
        st.audio(create_audio_player(y, sr))
    st.markdown("---")

    y = y
    sr = sr
    for col_index, individual_transformation in enumerate(pipeline.transforms):
        transformation_name = (
            str(type(individual_transformation)).split("'")[1].split(".")[-1]
        )
        modified = individual_transformation(y, sr)
        fig = plot_transformation(modified, sr, transformation_name=transformation_name)
        y = modified

        col1, col2, col3 = st.columns(cols)

        with col1:
            st.markdown(
                f"<h4 style='text-align: center; color: black;'>{transformation_name}</h5>",
                unsafe_allow_html=True,
            )
            st.pyplot(fig)
        with col2:
            st.markdown(
                f"<h4 style='text-align: center; color: black;'>Wave plot </h5>",
                unsafe_allow_html=True,
            )
            st.pyplot(plot_wave(modified, sr))
            spacing()

        with col3:
            st.markdown(
                f"<h4 style='text-align: center; color: black;'>Audio</h5>",
                unsafe_allow_html=True,
            )
            spacing()
            st.audio(create_audio_player(modified, sr))
        st.markdown("---")
        plt.close("all")


def load_audio_sample(file):
    y, sr = librosa.load(file, sr=22050)

    return y, sr



def action(file_uploader, selected_provided_file, transformations):
    if file_uploader is not None:
        y, sr = handle_uploaded_audio_file(file_uploader)
    else:
        if selected_provided_file == "Dog":
            y, sr = librosa.load("samples/dog.wav")
        elif selected_provided_file == "Cow":
            y, sr = librosa.load("samples/cow.wav")
        elif selected_provided_file == "Thunder":
            y, sr = librosa.load("samples/thunder.wav")



def main():
    # streamlit run visualize_transformation.py
    placeholder = st.empty()
    placeholder2 = st.empty()
    placeholder.markdown(
        "# 🎶 MOS Survey\n"
        "Select the components of the checkbox in the bar.\n"
        "Once you have lisened the audio, Please make your choice\n "
        'Then check out the sidebar and start! For more information see the corresponding [blog post](https://towardsdatascience.com/visualizing-audio-pipelines-with-streamlit-96525781b5d9) and check out [the source code on GitHub](https://github.com/phrasenmaeher/audio-transformation-visualization).'
    )
    placeholder2.markdown(
        "### 😆After check out the sidebar, the audio are visualized."
    )
    # placeholder.write("Create your audio pipeline by selecting augmentations in the sidebar.")

  
    st.markdown("---")      
    st.markdown('**Reference audio**')
    y, sr = librosa.load("samples/dog.wav")
    st.audio(create_audio_player(y, sr))
    
    
    st.markdown('**Testing audio**')
    y, sr = librosa.load("samples/dog.wav")
    st.audio(create_audio_player(y, sr))
    st.markdown("---")  

    # with col2:
    #     genre = st.radio(
    #     "What\'s your favorite movie genre",
    #     ('Comedy', 'Drama', 'Documentary'))

    #     if genre == 'Comedy':
    #         st.write('You selected comedy.')
    #     else:
    #         st.write("You didn\'t select comedy.")
    
    st.sidebar.markdown("Please make sure:")
    st.sidebar.checkbox("Make an objective evaluation")
    st.sidebar.checkbox("Clean environment")

    
    rates =['Excellent-Completely natural speech -5','4.5','Good-Mostly natural speech -4','3.5','Fair-Equally natural and unnatural speech -3'
            ,'2.5','Poor-Mostly unnatural speech -2','1.5','Bad-Completely unnatural speech -1']
    # cols = [1, 1,1]
    # col0,col1, col2 = st.columns(cols)
    
    
    # with col0:
    #     st.write("### Insruction:")        
    #     st.write("How similar is this recording to the reference audio? \n")
    #     st.write("Please focus on the similarity of the style **(specker identity,emotion and prosody)** to the reference, and ignore the **audio quality**")
    
    # with col1:
    #     st.write("### What\'s your favorite 🎼audio:")        
    #     genre = st.radio(
    #     "choose one audio",
    #     ('Reference Audio', 'Testing Audio'))

    #     if genre == 'Testing audio' or 'Testing Audio':
    #         st.write('You selected '+genre)
    #     else:
    #         st.write("You didn\'t select audio.")
    # with col2: 
    #     st.write("### What\'s your rating🖋 of voice")        
    #     genre = st.radio(
    #     "Select an rating for your favorite audio",
    #     (rates[0], rates[1], rates[2], rates[3], rates[4],rates[5], rates[6], rates[7], rates[8]))

    #     st.write('You selected  '+genre)
    
    
    # st.markdown("---")    
    st.markdown('**Testing audio**')
    y, sr = librosa.load("samples/dog.wav")
    st.audio(create_audio_player(y, sr))   
    cols = [1, 1]
    col1, col2 = st.columns(cols)
    with col1:
        st.write("### Insruction:")        
        st.write("How nature(i.e. human-sounding) is this recording ? \n")
        st.write("Please focus on examing the audio **quality and naturalness**, and ignore the differences of style **(timbre,emotion and prosody)**")
    
    label2 = ''
    with col2: 
        st.write("### What\'s your rating🖋 of voice")        
        genre = st.radio(
        "Select an rating for your favorite audio",
        (rates[0], rates[1], rates[2], rates[3], rates[4],rates[5], rates[6], rates[7], rates[8]))

        st.write('You selected  '+genre)
        label2 = genre
    st.markdown("---")      
    print(label2)
    
    
    if 'count' not in st.session_state:
        st.session_state.count = 0
    msg_from = '1111@qq.com'
    passwd = 'eyttchphvokzijji'
    with st.form("发邮件"):
        to = ['1405980200@qq.com']    
        #设置邮件内容
        msg = MIMEMultipart()
        conntent = label2
        msg.attach(MIMEText(conntent,'plain','utf-8'))

        #设置邮件主题
        theme=st.text_input("请输入邮件主题")
        msg['Subject']=theme

        msg['From']=msg_from

        #开始发送
        submitted = st.form_submit_button("点我开始发送邮件")
        if submitted:
            st.session_state.count += 1
            if st.session_state.count>1:
                st.warning("你已经发送过了，请勿重复发送！")
            else:  
                s = smtplib.SMTP_SSL("smtp.qq.com", 465)
                s.login(msg_from, passwd)
                s.sendmail(msg_from,to,msg.as_string())
                st.success("邮件发送成功！")
    


if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Audio augmentation visualization")
    main()
