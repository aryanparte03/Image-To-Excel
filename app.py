import streamlit as st
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import easyocr
import io
from streamlit_pasteimage import paste_image

# --- Language selection ---
user_langs = st.multiselect(
    "OCR language(s) for this session",
    options=['en', 'hi', 'fr', 'de', 'zh', 'es', 'ru', 'ja', 'ko', 'ar'],
    default=["en"]
)
reader = easyocr.Reader(user_langs, gpu=False)

st.title("Paste or Upload Table Image (Ctrl+V)")
st.caption("Tip: Just copy a table screenshot (e.g., with Snipping Tool) and press Ctrl+V here!")

# --- Clipboard or Upload image ---
img = paste_image("Paste table screenshot here (Ctrl+V below)")
if img is not None:
    image = Image.fromarray(img)
else:
    uploaded_file = st.file_uploader(
        "Or upload a table screenshot", type=['png','jpg','jpeg','bmp','tiff'])
    image = Image.open(uploaded_file) if uploaded_file else None

if image is not None:
    st.image(image, caption="Input Table Image"), st.write("---")

    # --- 1. Pre-processing: grayscale, adaptive threshold, morph ops ---
    img_cv = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (2,2), 0)
    th = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                              cv2.THRESH_BINARY_INV, 17, 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    morph = cv2.dilate(cv2.erode(th, kernel, iterations=1), kernel, iterations=1)

    # --- 2. Find and filter contours (likely cell boxes) ---
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c)>80]
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))  # top-to-bottom, then left-to-right

    # --- 3. Cluster into rows based on vertical proximity ---
    def group_boxes(boxes, vtol=20):
        rows = []
        cur_row, last_y = [], None
        for box in boxes:
            x,y,w,h = box; cy = y+h//2
            if last_y is not None and abs(cy-last_y)>vtol:
                rows.append(cur_row); cur_row=[]
            cur_row.append(box); last_y=cy
        if cur_row: rows.append(cur_row)
        return [sorted(r, key=lambda b:b[0]) for r in rows if r]

    grouped = group_boxes(boxes)
    max_cols = max(len(r) for r in grouped) if grouped else 1

    # --- 4. OCR every detected cell; filter low-confidence ---
    data = []
    for row in grouped:
        line = []
        for x, y, w, h in row:
            cimg = img_cv[y:y+h,x:x+w]
            result = reader.readtext(cimg, detail=1, paragraph=True)
            celltxt = ""
            if result:  # Only use text above a confidence threshold
                strings = [out[1] for out in result if out[2]>0.5]
                celltxt = " ".join(strings)
            line.append(celltxt)
        # pad to rectangle
        line += [""]*(max_cols-len(line))
        data.append(line)

    df = pd.DataFrame(data)
    st.subheader("Editable Table Preview")
    df_edit = st.data_editor(df, num_rows="dynamic")
    st.subheader("Export to Excel")
    outbuf = io.BytesIO()
    df_edit.to_excel(outbuf, index=False, header=False)
    st.download_button(
        "Download as Excel",
        outbuf.getvalue(),
        file_name="table.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("Paste a screenshot or upload a table image.")

st.caption("Robust, noise-resistant: advanced preprocessing, classic+OCR synergy, Ctrl+V paste, instant Excel output.")
