HƯỚNG DẪN SỬ DỤNG

1. Huấn luyện AI
Yêu cầu: Chuẩn bị ảnh cần thiết để huấn luyện

a. Nhập ảnh
Trong file "AI_NhanDienCamXuc", truy cập vào file /train.
Tại đây sẽ có các thư mục con gồm "Angry", "Disgust", "Fear", "Happy", "Netral", "Sad", "Surprise" tương ứng với "Tức giận", "Ghê tởm", "Sợ hãi", "Vui vẻ", "Bình thường", "Buồn bã", "Bất ngờ".
Trong mỗi thư mục còn này là nơi chứa các ảnh biểu cảm tương ứng với tên thư mục.
Nhập các ảnh biểu cảm tương ứng vào mỗi thư mục con.

b. Tiến hành huấn luyện
Trong file "AI_NhanDienCamXuc", truy cập vào file "TrainAI.py".
Hệ thống sẽ tự động truy cập đến file "Train" và phân loại các ảnh trong thu mục con thành từng biểu cảm.
Hệ thống tiến hành tách các điểm mốc khuôn mặt (Các phần của khuông mặt được hệ thống đánh dấu: Lông mày, mắt, miệng).
Tự động tối ưu hóa và phân loại các ảnh, bỏ qua các ảnh không nhận diện được hoặc không phù hợp. Tự động lấy ảnh tốt nhất để phân tích.
Lưu dữ liệu biểu cảm đã được huấn luyện thành một file H5 là "Emotion_model.h5". Đây là file đã được hệ thống phân tích từ ảnh thành mã máy, chứa dữ liệu biểu cảm đã huấn luyện.


2. Sự dụng hệ thống phân tích biểu cảm

a.Khởi động hệ thống

Trong file "AI_NhanDienCamXuc", truy cập vào file "App.py".
Hệ thống sẽ khởi chạy và thông báo một đường dẫn "http://127.0.0.1:5000/"
Truy cập vào đường dẫn đã được thông báo.
Tại giao diện web sẽ có hai lựa chọn để phân tích biểu cảm: Nhận diện biểu cảm theo thời gian thực từ Webcam và Nhận diện biểu cảm từ ảnh.

b.Nhận diện biểu cảm theo thời gian thực từ Webcam
Chọn chức năng "Nhận diện biểu cảm theo thời gian thực từ webcam".
Hệ thống sẽ yêu cầu cấp quyền truy cập webcam.
Khi webcam được bật hệ thống sẽ bắt đầu phân tích khuôn mặt trong webcam.
Thể hiện các biểu cảm khác nhau và hệ thống sẽ lấy dữ liệu từ webcam so sánh với dữ liệu đã được huấn luyện, phân tích biểu cảm và dự đoán.
Hiện thị kết quả lên giao diện web.

c. Nhận diện biểu cảm từ ảnh
Chọn chức năng "Nhận diện biểu cảm từ ảnh"
Hệ thống sẽ yêu cầu nhập ảnh cần phân tích.
Nhập ảnh muốn phân tích vào.
Tương tự với nhận diện biểu cảm theo thời gian thực từ webcame hệ thống sẽ tách các điểm mốc của khuôn mặt và so sánh với các dữ liệu đã được huấn luyện.
Phân tích và dự đoán các biểu cảm.
Hiện thị kết quả lên web.