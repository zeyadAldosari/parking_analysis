import streamlit as st
import os
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

def load_api_key():
    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        st.error("No API KEY")
    return api_key

class ParkingAnalyzer:
    def __init__(self, is_local=False, api_key=None, model_path=None):
        self.is_local = is_local
        if is_local:
            self.model = YOLO(model_path)
            self.class_names = {0: "car", 1: "free"}
        else:
            self.client = InferenceHTTPClient(
                api_url="https://detect.roboflow.com",
                api_key=api_key
            )
        
        self.colors = {'free': (0, 255, 0), 'car': (0, 0, 255)}
        self.zones = {}

    def define_zones(self, detections):
        if not detections:
            return {}
            
        spaces = [(int(d['x']), int(d['y'])) for d in detections]
        spaces.sort(key=lambda x: x[0])
        
        leftmost_x = min(x for x, _ in spaces)
        rightmost_x = max(x for x, _ in spaces)
        zone_width = (rightmost_x - leftmost_x) / 3
        
        return {
            'A': {'min_x': leftmost_x, 'max_x': leftmost_x + zone_width},
            'B': {'min_x': leftmost_x + zone_width, 'max_x': leftmost_x + 2*zone_width},
            'C': {'min_x': leftmost_x + 2*zone_width, 'max_x': rightmost_x}
        }

    def draw_detections(self, image, detections):
        visualization = image.copy()
        zone_stats = {zone: {'free': 0, 'occupied': 0} for zone in self.zones.keys()}
        cv2.rectangle(visualization, (10, 10), (250, 130), (0, 0, 0), -1)
        cv2.putText(visualization, "Parking Analysis:", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        for det in detections:
            x, y = int(det['x']), int(det['y'])
            w, h = int(det['width']), int(det['height'])
            x1, y1 = int(x - w/2), int(y - h/2)
            x2, y2 = int(x + w/2), int(y + h/2)
            
            current_zone = None
            for zone_name, bounds in self.zones.items():
                if bounds['min_x'] <= x <= bounds['max_x']:
                    current_zone = zone_name
                    if det['class'] == 'free':
                        zone_stats[zone_name]['free'] += 1
                    else:
                        zone_stats[zone_name]['occupied'] += 1
                    break
            
            color = self.colors[det['class']]
            cv2.rectangle(visualization, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class']} - Zone {current_zone}" if current_zone else det['class']
            cv2.putText(visualization, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        y_offset = 50
        for zone, stats in zone_stats.items():
            total = stats['free'] + stats['occupied']
            if total > 0:
                occupancy = (stats['occupied'] / total) * 100
                text = f"Zone {zone}: {stats['free']} free, {occupancy:.1f}% full"
                cv2.putText(visualization, text, (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 20
                
        return visualization, zone_stats

    def get_recommendation(self, zone_stats):
        best_zone = max(zone_stats.items(), 
                       key=lambda x: x[1]['free'])
        return (f"Recommended parking area: Zone {best_zone[0]} "
                f"({best_zone[1]['free']} free spaces)") if best_zone[1]['free'] > 0 else "No free spaces available"

    def convert_yolo_detections(self, results):
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                width = x2 - x1
                height = y2 - y1
                
                detections.append({
                    'x': float(x1 + width/2),
                    'y': float(y1 + height/2),
                    'width': float(width),
                    'height': float(height),
                    'class': self.class_names[cls]
                })
        return detections

    def analyze_image(self, image):
        if self.is_local:
            results = self.model(image)
            detections = self.convert_yolo_detections(results)
        else:
            result = self.client.infer(image, model_id="deteksiparkirkosong/6")
            detections = result.get('predictions', [])
            
        self.zones = self.define_zones(detections)
        annotated_image, zone_stats = self.draw_detections(
            cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), 
            detections
        )
        
        return {
            'annotated_image': annotated_image,
            'zone_stats': zone_stats,
            'recommendation': self.get_recommendation(zone_stats)
        }

def main():
    st.title("Parking Analysis")
    
    model_option = st.radio("Select Detection Model",
                           ["Roboflow API", "Local Model"])
    
    analyzer = None
    if model_option == "Roboflow API":
        api_key = load_api_key()
        if api_key:
            analyzer = ParkingAnalyzer(is_local=False, api_key=api_key)
    else:
        model_path = "best.pt"
        if os.path.exists(model_path):
            analyzer = ParkingAnalyzer(is_local=True, model_path=model_path)
        else:
            st.error(f"Local model not found at {model_path}")
    
    if analyzer and (uploaded_file := st.file_uploader(
            "Upload parking lot image", 
            type=['jpg', 'jpeg', 'png'])):
        try:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(image)
            
            results = analyzer.analyze_image(image)
            
            with col2:
                st.subheader("Analysis Result")
                st.image(cv2.cvtColor(results['annotated_image'], 
                                    cv2.COLOR_BGR2RGB))
            
            st.subheader("Zone Analysis")
            cols = st.columns(len(results['zone_stats']))
            for i, (zone, stats) in enumerate(results['zone_stats'].items()):
                total = stats['free'] + stats['occupied']
                if total > 0:
                    occupancy = (stats['occupied'] / total) * 100
                    cols[i].metric(f"Zone {zone}",
                                 f"{stats['free']} free spaces",
                                 f"{occupancy:.1f}% occupied")
            
            st.success(results['recommendation'])
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()