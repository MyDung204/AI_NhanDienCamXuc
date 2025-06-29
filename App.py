import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import io
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, render_template, Response, request, jsonify, send_file
import base64
import datetime
import os

app = Flask(__name__)

# Load mô hình đã huấn luyện
model = load_model("emotion_model.h5", compile=False)

# Load file nhận diện khuôn mặt từ OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Danh sách cảm xúc
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Ánh xạ cảm xúc sang emoji (cần giống với frontend)
EMOJI_MAP = {
    "Angry": "😡",
    "Disgust": "🤢",
    "Fear": "😨",
    "Happy": "😃",
    "Neutral": "😐",
    "Sad": "😢",
    "Surprise": "😲"
}

# Cố gắng tải font cho emoji
emotion_font = None
text_font = None

font_paths = [
    "C:/Windows/Fonts/seguiemj.ttf",  # Windows Segoe UI Emoji
    "/System/Library/Fonts/Apple Color Emoji.ttc",  # macOS Apple Color Emoji
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    # Linux Noto Color Emoji (install via apt-get install fonts-noto-color-emoji)
    "NotoColorEmoji.ttf",  # Fallback for local project folder
    "arial.ttf",  # General fallback (might not show color emoji but will render text)
    "DejaVuSans.ttf"  # Another common Linux font
]

for path in font_paths:
    try:
        if os.path.exists(path):
            emotion_font = ImageFont.truetype(path, 55)
            text_font = ImageFont.truetype(path, 35)
            print(f"Loaded font from: {path}")
            break
    except IOError as e:
        print(f"IOError loading font {path}: {e}")
    except Exception as e:
        print(f"Error loading font {path}: {e}")

if not emotion_font:
    print("Warning: Could not load any specific emoji font. Using default PIL font (may not display color emoji).")
    emotion_font = ImageFont.load_default()
    text_font = ImageFont.load_default()

# Biến toàn cục để quản lý camera và dữ liệu cảm xúc
cap = None
latest_webcam_emotion_data = {"emotion": "Neutral", "confidence": 0.0,
                              "all_confidences": {label: 0.0 for label in emotion_labels}}
latest_webcam_frame_for_snapshot = None  # Store frame for snapshot
emotion_history = deque(maxlen=10)  # Used for smoothing webcam emotions


def get_camera():
    global cap
    if cap is None or not cap.isOpened():
        print("Attempting to open webcam...")
        cap = cv2.VideoCapture(0)  # 0 is default camera
        if not cap.isOpened():
            print("Error: Could not open webcam. Make sure it's connected and not in use.")
            cap = None  # Set to None if opening failed
            return None
        print("Webcam opened successfully.")
    return cap


def release_camera():
    global cap
    if cap is not None and cap.isOpened():
        print("Releasing webcam...")
        cap.release()
        cap = None
        print("Webcam released.")


def process_frame_for_display(frame):
    global latest_webcam_emotion_data, latest_webcam_frame_for_snapshot

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    current_frame_emotion = "Neutral"
    current_frame_confidence = 0.0
    current_frame_all_confidences = {label: 0.0 for label in emotion_labels}

    display_frame = frame.copy()  # Use a copy for drawing to preserve original for last_processed_frame

    if len(faces) > 0:
        (x, y, w, h) = faces[0]  # Focus on the first face for dominant emotion
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (64, 64))
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)

        preds = model.predict(face)[0]  # cite: 5

        for i, label in enumerate(emotion_labels):
            current_frame_all_confidences[label] = float(preds[i])

        predicted_emotion = emotion_labels[np.argmax(preds)]  # cite: 5
        confidence = float(np.max(preds))  # cite: 5

        # Apply smoothing for the displayed label
        emotion_history.append(predicted_emotion)
        most_common_emotion = max(set(emotion_history), key=emotion_history.count)

        current_frame_emotion = most_common_emotion
        current_frame_confidence = confidence  # Keep actual confidence for display/data

        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        text_to_display = f"{current_frame_emotion} ({current_frame_confidence * 100:.2f}%)"

        # Adjust text position carefully if it goes out of frame
        text_size = cv2.getTextSize(text_to_display, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = x
        text_y = y - 10
        if text_y < text_size[1] + 10:  # If too close to top
            text_y = y + h + text_size[1] + 10
        if text_x + text_size[0] > display_frame.shape[1]:  # If goes out of right boundary
            text_x = display_frame.shape[1] - text_size[0] - 5

        cv2.putText(display_frame, text_to_display, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    latest_webcam_emotion_data = {
        "emotion": current_frame_emotion,
        "confidence": current_frame_confidence,
        "all_confidences": current_frame_all_confidences
    }
    # Always store a copy of the original frame (without drawings) for clean snapshots
    latest_webcam_frame_for_snapshot = frame.copy()

    return display_frame


def generate_webcam_frames():
    camera = get_camera()
    if camera is None:
        print("Webcam is not available, stopping frame generation.")
        # Yield an empty or placeholder frame if camera fails to open to prevent client hanging
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', np.zeros((480, 640, 3), np.uint8))[
                   1].tobytes() + b'\r\n')
        return

    while camera.isOpened():
        success, frame = camera.read()
        if not success:
            print("Failed to read frame from camera. Releasing camera.")
            break

        processed_frame = process_frame_for_display(frame)

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # Ensure camera is released if loop breaks (e.g., camera disconnected)
    release_camera()


def add_emotion_overlay(image_np, dominant_emotion, all_confidences_dict):
    # Ensure image_np is BGR for OpenCV functions, convert to RGB for PIL
    img_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    if emotion_font and text_font:
        emotion_emoji_char = EMOJI_MAP.get(dominant_emotion, "🤔")  # Use thinking emoji as fallback
        emotion_text_label = dominant_emotion  # Just the label for simplicity on overlay

        width_pil, height_pil = img_pil.size
        # TĂNG MARGIN CỦA OVERLAY TỪ CÁC CẠNH ĐỂ TRÁNH BỊ CẮT VÀ DỊCH XUỐNG DƯỚI
        margin_x = 55  # Tăng từ 50 -> 55
        margin_y = 60  # Tăng từ 55 -> 60

        # Get text bounding box for accurate positioning
        emoji_bbox = draw.textbbox((0, 0), emotion_emoji_char, font=emotion_font)
        emoji_width = emoji_bbox[2] - emoji_bbox[0]
        emoji_height = emoji_bbox[3] - emoji_bbox[1]

        text_bbox = draw.textbbox((0, 0), emotion_text_label, font=text_font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Combine width of emoji and text for background calculation
        content_width = max(emoji_width, text_width)
        content_height = emoji_height + text_height + 15  # 15px gap between emoji and text

        bg_padding_x = 25
        bg_padding_y = 20

        # Position for the whole overlay block
        block_x1 = width_pil - content_width - margin_x - bg_padding_x
        block_y1 = margin_y - bg_padding_y
        block_x2 = width_pil - margin_x + bg_padding_x
        block_y2 = block_y1 + content_height + 2 * bg_padding_y

        # Draw a translucent background rectangle
        draw.rounded_rectangle((block_x1, block_y1, block_x2, block_y2), radius=18,
                               fill=(0, 0, 0, 200))  # Darker, more rounded background

        # Calculate text/emoji drawing positions within the background block
        emoji_draw_x = block_x1 + (block_x2 - block_x1 - emoji_width) / 2
        emoji_draw_y = block_y1 + bg_padding_y + 5

        text_draw_x = block_x1 + (block_x2 - block_x1 - text_width) / 2
        text_draw_y = emoji_draw_y + emoji_height + 10

        draw.text((emoji_draw_x, emoji_draw_y), emotion_emoji_char, font=emotion_font,
                  fill=(255, 255, 255))  # White for emoji
        draw.text((text_draw_x, text_draw_y), emotion_text_label, font=text_font,
                  fill=(255, 255, 255))  # White for text
    else:
        print("Warning: Fonts not loaded, skipping text/emoji drawing on output image.")

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_BGR2RGB)  # Convert back to OpenCV format (BGR)


@app.route('/')
def index():
    return render_template('index.html')  # cite: 1


@app.route('/video_feed')
def video_feed():
    print("Serving video feed...")
    return Response(generate_webcam_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')  # cite: 1


@app.route('/webcam_emotion_data')
def webcam_emotion_data():
    return jsonify(latest_webcam_emotion_data)


@app.route('/start_webcam')
def start_webcam_api():
    camera_started = get_camera()
    if camera_started:
        print("Webcam start API called: Success.")
        return jsonify({"status": "webcam started"}), 200
    print("Webcam start API called: Failed.")
    return jsonify({"status": "failed to start webcam", "error": "Camera not available"}), 500


@app.route('/stop_webcam')
def stop_webcam_api():
    release_camera()
    print("Webcam stop API called: Success.")
    return jsonify({"status": "webcam stopped"}), 200


@app.route('/capture_webcam_frame', methods=['GET'])
def capture_webcam_frame():
    global latest_webcam_frame_for_snapshot
    if latest_webcam_frame_for_snapshot is not None:
        # Add emotion overlay to the snapshot frame
        processed_snapshot_frame = add_emotion_overlay(
            latest_webcam_frame_for_snapshot.copy(),  # Pass a copy to avoid modifying the stored frame
            latest_webcam_emotion_data["emotion"],
            latest_webcam_emotion_data["all_confidences"]  # Pass all confidences if needed for future enhancements
        )

        ret, buffer = cv2.imencode('.jpg', processed_snapshot_frame)
        if ret:
            img_io = io.BytesIO(buffer)
            img_io.seek(0)
            filename = datetime.datetime.now().strftime("emotion_snapshot_%Y%m%d_%H%M%S.jpg")
            return send_file(img_io, mimetype='image/jpeg', as_attachment=True, download_name=filename)
    print("Capture webcam frame: No frame available.")
    return "No frame available", 404


@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            in_memory_file = io.BytesIO()
            file.save(in_memory_file)
            in_memory_file.seek(0)
            img_np = np.frombuffer(in_memory_file.read(), np.uint8)
            frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

            if frame is None:
                return jsonify({"error": "Không thể giải mã ảnh. File ảnh không hợp lệ hoặc bị hỏng."}), 400

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                print("No face detected in uploaded image.")
                return jsonify({"error": "Không tìm thấy khuôn mặt nào trong ảnh. Vui lòng tải ảnh rõ mặt hơn."}), 400

            (x, y, w, h) = faces[0]  # Focus on the first face for analysis
            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (64, 64))
            face_roi = face_roi.astype('float32') / 255.0
            face_roi = np.expand_dims(face_roi, axis=0)
            face_roi = np.expand_dims(face_roi, axis=-1)

            preds = model.predict(face_roi)[0]  # cite: 5

            all_confidences = {label: float(preds[i]) for i, label in enumerate(emotion_labels)}
            dominant_emotion = emotion_labels[np.argmax(preds)]  # cite: 5
            dominant_confidence = float(np.max(preds))  # cite: 5

            # Draw rectangle and dominant emotion label on the frame (OpenCV)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # Position text carefully to avoid going off-screen
            text_size = cv2.getTextSize(dominant_emotion, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[
                0]  # Giảm kích thước font một chút
            text_x = x
            text_y = y - 10
            if text_y < text_size[1] + 10:  # If too close to top
                text_y = y + h + text_size[1] + 10
            if text_x + text_size[0] > frame.shape[1]:  # If goes out of right boundary
                text_x = frame.shape[1] - text_size[0] - 5

            cv2.putText(frame, dominant_emotion, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Add emotion overlay using PIL (text and emoji)
            processed_analyzed_image = add_emotion_overlay(frame, dominant_emotion, all_confidences)

            byte_arr = io.BytesIO()
            # Convert numpy array (BGR from OpenCV) to PIL Image (RGB) for saving
            Image.fromarray(cv2.cvtColor(processed_analyzed_image, cv2.COLOR_BGR2RGB)).save(byte_arr, format='JPEG')
            byte_arr.seek(0)
            encoded_img = base64.b64encode(byte_arr.getvalue()).decode('utf-8')  # cite: 1

            return jsonify({
                "success": True,
                "emotion": dominant_emotion,
                "confidence": dominant_confidence,
                "all_confidences": all_confidences,
                "image_data": encoded_img
            })

        except Exception as e:
            print(f"Error during image analysis: {e}")
            error_message = f"Lỗi xử lý ảnh: {str(e)}. Vui lòng thử lại."
            if "Unsupported image type" in str(e) or "not a JPEG file" in str(e) or "image file is truncated" in str(e):
                error_message = "File không phải là định dạng ảnh hợp lệ (JPG, PNG) hoặc bị hỏng. Vui lòng kiểm tra lại."
            return jsonify({"error": error_message}), 500

    return jsonify({"error": "Unknown error"}), 500


if __name__ == "__main__":
    # Ensure camera is released when app starts (if previous run crashed)
    release_camera()
    app.run(debug=True, threaded=True)  # Use threaded=True for better concurrent handling of requests