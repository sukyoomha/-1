#분류 결과 + 이미지 + 텍스트와 함께 분류 결과에 따라 다른 출력 보여주기
#파일 이름 streamlit_app.py
import streamlit as st
from fastai.vision.all import *
from PIL import Image
import gdown

# Google Drive 파일 ID
file_id = '1_Nq-wFgjn1fi-eoor0HGF7_pXXL1b92F'

# Google Drive에서 파일 다운로드 함수
#@st.cache(allow_output_mutation=True)
@st.cache_data
def load_model_from_drive(file_id):
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'model.pkl'
    gdown.download(url, output, quiet=False)

    # Fastai 모델 로드
    learner = load_learner(output)
    return learner

def display_left_content(image, prediction, probs, labels):
    st.write("### 왼쪽: 기존 출력 결과")
    if image is not None:
        st.image(image, caption="업로드된 이미지", use_container_width=True)
    st.write(f"예측된 클래스: {prediction}")
    st.markdown("<h4>클래스별 확률:</h4>", unsafe_allow_html=True)
    for label, prob in zip(labels, probs):
        st.markdown(f"""
            <div style="background-color: #f0f0f0; border-radius: 5px; padding: 5px; margin: 5px 0;">
                <strong style="color: #333;">{label}:</strong>
                <div style="background-color: #d3d3d3; border-radius: 5px; width: 100%; padding: 2px;">
                    <div style="background-color: #4CAF50; width: {prob*100}%; padding: 5px 0; border-radius: 5px; text-align: center; color: white;">
                        {prob:.4f}
                    </div>
                </div>
        """, unsafe_allow_html=True)

def display_right_content(prediction, data):
    st.write("### 오른쪽: 동적 분류 결과")
    cols = st.columns(3)

    # 1st Row - Images
    for i in range(3):
        with cols[i]:
            st.image(data['images'][i], caption=f"이미지: {prediction}", use_container_width=True)

    # 3rd Row - Text
    for i in range(3):
        with cols[i]:
            st.write(data['texts'][i])

# 모델 로드
st.write("모델을 로드 중입니다. 잠시만 기다려주세요...")
learner = load_model_from_drive(file_id)
st.success("모델이 성공적으로 로드되었습니다!")

labels = learner.dls.vocab

# 스타일링을 통해 페이지 마진 줄이기
st.markdown("""
    <style>
    .reportview-container .main .block-container {
        max-width: 90%;
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# 분류에 따라 다른 콘텐츠 관리
content_data = {
    labels[0]: {
        'images': [
            "https://www.whosaeng.com/imgdata/whosaeng_com/202012/2020123019521698.jpg",
            "https://blog.kakaocdn.net/dn/tavtk/btq97mNuhyW/0CSseiaexjbl78815PtGNK/img.png",
            "https://barkiri.cdn.ntruss.com/product/p79.webp"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ"
        ],
        'texts': [
            "이지엔 입니다.",
            "이부프로펜 200mg가 함유된 진통제입니다.",
            "과량 복용시에 위 또는 장관의 출혈, 궤양 및 천공 등 위장관계의 심각한 부작용의 위험을 증가시킬 수 있습니다."
        ]
    },
    labels[1]: {
        'images': [
            "https://pimg.mk.co.kr/meet/neds/2021/06/image_readtop_2021_575286_16236787304681117.jpg",
            "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQgnxsjgcKqc36eAxoHgbOT0csPGpyeQeD0CQ&s",
            "https://cdn.hitnews.co.kr/news/photo/202205/39424_47499_1333.jpg"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "타이레놀 입니다.",
            "아세트아미노펜 500mg가 함유된 진통제입니다.",
            "과량 복용시에 매우 심각한 간 손상을 일으킬 수 있으며, 췌장, 신 장에도 심각한 손상을 유발할 수 있습니다."
        ]
    },
    labels[2]: {
        'images': [
            "https://mblogthumb-phinf.pstatic.net/MjAyMzA2MjJfNjkg/MDAxNjg3NDA4MTQ4MTg3.F3OlXNky5aTwhWk9C_7fcg3cttx64Nh6NUwGz7u2zfwg.NNQrslZhs_KeAH9PEqREv9wfEI2K9eXUetAD1-UsJyIg.JPEG.jongha3846/1.jpg?type=w800",
            "https://nedrug.mfds.go.kr/pbp/cmn/itemImageDownload/1MtU6vRYsHt",
            "https://cdn.hitnews.co.kr/news/photo/202401/51883_68783_1458.png"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ"
        ],
        'texts': [
            "탁센입니다.",
            "나프록센 250mg가 함유된 진통제입니다.",
            "과량 복용시에 위장관 장애 및 간과 신장에 부담을 줄 수 있습니다."
        ]
    }
}

# 레이아웃 설정
left_column, right_column = st.columns([1, 2])  # 왼쪽과 오른쪽의 비율 조정

# 파일 업로드 컴포넌트 (jpg, png, jpeg, webp, tiff 지원)
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg", "webp", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = PILImage.create(uploaded_file)
    prediction, _, probs = learner.predict(img)

    with left_column:
        display_left_content(image, prediction, probs, labels)

    with right_column:
        # 분류 결과에 따른 콘텐츠 선택
        data = content_data.get(prediction, {
            'images': ["https://via.placeholder.com/300"] * 3,
            'videos': ["https://www.youtube.com/watch?v=3JZ_D3ELwOQ"] * 3,
            'texts': ["기본 텍스트"] * 3
        })
        display_right_content(prediction, data)

