import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="Cancer Classification",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Cancer Classification App")
st.write("Upload an image to predict the cancer type.")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


num_classes = 4  # Change according to your dataset

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)

MODEL_PATH = "model.pth"

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


class_names = [
    "Benign",
    "Malignant",
    "Normal",
    "Other"
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict_image(image):
    image = image.convert("RGB")

    img_tensor = transform(image)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item() * 100

    return predicted_class, confidence_score, probabilities.cpu().numpy()[0]


uploaded_file = st.file_uploader(
    "Upload Cancer Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):

        with st.spinner("Predicting..."):

            prediction, confidence, probs = predict_image(image)

        st.success(f"Prediction: {prediction}")

        st.info(f"Confidence: {confidence:.2f}%")

        st.subheader("Prediction Probabilities")

        for i, class_name in enumerate(class_names):
            st.write(f"{class_name}: {probs[i] * 100:.2f}%")
