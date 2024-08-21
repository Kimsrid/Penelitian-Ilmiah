import argparse
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv  # Pastikan ini adalah modul yang benar dari Supervisely
from collections import defaultdict, deque

SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])

TARGET_WIDTH = 25
TARGET_HEIGHT = 250


TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)

class ViewTransformer: 
    def __init__(self, source: np.ndarray, target: np.ndarray): 
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Deskripsi dari program")
    parser.add_argument(
        "--source_video_path",
        type=str,
        default=r"vehicles.mp4",  # Menggunakan raw string
        help="Path ke video sumber"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    print(f"Source video path: {args.source_video_path}")
    
    # Dapatkan informasi video
    video_info = sv.VideoInfo.from_video_path(args.source_video_path)
    
    # Inisialisasi model YOLO (ganti dengan model yang tersedia)
    model = YOLO("yolov8x.pt")

    byte_track = sv.ByteTrack(frame_rate=video_info.fps)

    # Hitung ketebalan garis dinamis
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
  
    # Inisialisasi annotator bounding box dan label
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(text_thickness=thickness)

    # Dapatkan frame dari video
    frame_generator = sv.get_video_frames_generator(args.source_video_path)

    polygon_zone = sv.PolygonZone(SOURCE, frame_resolution_wh=video_info.resolution_wh)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

    # Iterasi melalui setiap frame
    for frame in frame_generator:
        # Dapatkan hasil deteksi dari model YOLO
        result = model(frame)[0]
        
        # Konversi hasil deteksi ke format Detections
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[polygon_zone.trigger(detections)]
        detections = byte_track.update_with_detections(detections=detections)

        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        points = view_transformer.transform_points(points=points).astype(int)

        labels = []
        for tracker_id, [_, y] in zip(detections.tracker_id, points):
            coordinates[tracker_id].append(y)
            if len(coordinates[tracker_id]) < video_info.fps / 2:
                labels.append(f"#{tracker_id}")
            else:
                coordinate_start = coordinates[tracker_id][-1]
                coordinate_end = coordinates[tracker_id][0]
                distance = abs(coordinate_start - coordinate_end)
                time = len(coordinates[tracker_id]) / video_info.fps
                speed = distance / time * 3.6
                labels.append(f"#{tracker_id} {int(speed)} km/h")
                print(f"Tracker ID: {tracker_id}, Speed: {int(speed)} km/h")

        # Salin frame untuk anotasi
        annotated_frame = frame.copy()
        
        # Anotasi frame dengan bounding box
        annotated_frame = bounding_box_annotator.annotate(
            scene=annotated_frame, detections=detections)
        
        # Anotasi frame dengan label
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels)

        # Menggambar poligon pada frame
        cv2.polylines(annotated_frame, [SOURCE], isClosed=True, color=(0, 0, 255), thickness=thickness)

        # Skala ulang frame jika terlalu besar
        scale_percent = 50  # Skala frame menjadi 50% dari ukuran aslinya
        width = int(annotated_frame.shape[1] * scale_percent / 100)
        height = int(annotated_frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_frame = cv2.resize(annotated_frame, dim, interpolation=cv2.INTER_AREA)
        

        # Tampilkan frame yang telah dianotasi dan diskala ulang
        cv2.imshow("annotated_frame", resized_frame)
        
        # Tunggu input pengguna untuk keluar (tekan 'q' untuk keluar)
        if cv2.waitKey(1) == ord("q"):
            break
    
    # Tutup semua jendela OpenCV
    cv2.destroyAllWindows()
