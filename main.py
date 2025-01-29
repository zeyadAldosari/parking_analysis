import streamlit as st
import os
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
import cv2
import numpy as np
from PIL import Image

def load_api_key():
    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        st.error("No API KEY")
    return api_key

class EnhancedParkingAnalyzer:
    def __init__(self, api_key):
        self.client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=api_key
        )
        self.colors = {
            'free': (0, 255, 0),
            'car': (0, 0, 255)  
            }
        self.zones = {}
        
    def preprocess_image(self, uploaded_file):
        pil_image = Image.open(uploaded_file)
        cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return pil_image, cv2_image

    def define_zones(self, detections):
        if not detections:
            return {}
        spaces = [(int(d['x']), int(d['y'])) for d in detections]
        spaces.sort(key=lambda x: x[0])
        total_spaces = len(spaces)
        zones = {}
        
        if total_spaces > 0:
            leftmost_x = min(x for x, _ in spaces)
            rightmost_x = max(x for x, _ in spaces)
            zone_width = (rightmost_x - leftmost_x) / 3
            zones = {
                'A (Entrance)': {'min_x': leftmost_x, 'max_x': leftmost_x + zone_width},
                'B (Middle)': {'min_x': leftmost_x + zone_width, 'max_x': leftmost_x + 2*zone_width},
                'C (End)': {'min_x': leftmost_x + 2*zone_width, 'max_x': rightmost_x}
            }
        
        return zones

    def draw_detections(self, cv2_image, detections):
        visualization = cv2_image.copy()
        
        zone_stats = {zone: {'free': 0, 'occupied': 0} for zone in self.zones.keys()}
        
        cv2.rectangle(visualization, (10, 10), (250, 130), (0, 0, 0), -1)
        cv2.putText(visualization, "Parking Analysis:", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        for detection in detections:
            x = int(detection['x'])
            y = int(detection['y'])
            width = int(detection['width'])
            height = int(detection['height'])
            
            x1 = int(x - width/2)
            y1 = int(y - height/2)
            x2 = int(x + width/2)
            y2 = int(y + height/2)
            
            current_zone = None
            for zone_name, zone_bounds in self.zones.items():
                if zone_bounds['min_x'] <= x <= zone_bounds['max_x']:
                    current_zone = zone_name
                    if detection['class'] == 'free':
                        zone_stats[zone_name]['free'] += 1
                    else:
                        zone_stats[zone_name]['occupied'] += 1
                    break
            
            color = self.colors[detection['class']]
            cv2.rectangle(visualization, (x1, y1), (x2, y2), color, 2)
            
            label = f"{detection['class']} - Zone {current_zone[0]}" if current_zone else detection['class']
            cv2.putText(visualization, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        y_offset = 50
        for zone_name, stats in zone_stats.items():
            total = stats['free'] + stats['occupied']
            if total > 0:
                occupancy_rate = (stats['occupied'] / total) * 100
                text = f"Zone {zone_name[0]}: {stats['free']} free, {occupancy_rate:.1f}% full"
                cv2.putText(visualization, text, (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 20
        
        return visualization, zone_stats

    def get_recommendation(self, zone_stats):
        best_zone = None
        max_free_spaces = -1
        
        for zone_name, stats in zone_stats.items():
            if stats['free'] > max_free_spaces:
                max_free_spaces = stats['free']
                best_zone = zone_name
        
        if best_zone and max_free_spaces > 0:
            return f"Recommended parking area: Zone {best_zone} ({max_free_spaces} free spaces)"
        else:
            return "No free spaces available at the moment"

    def analyze_image(self, pil_image, cv2_image):
        try:
            result = self.client.infer(pil_image, model_id="deteksiparkirkosong/6")
            detections = result.get('predictions', [])
            self.zones = self.define_zones(detections)
            annotated_image, zone_stats = self.draw_detections(cv2_image, detections)
            recommendation = self.get_recommendation(zone_stats)
            
            return {
                'annotated_image': annotated_image,
                'zone_stats': zone_stats,
                'recommendation': recommendation
            }
            
        except Exception as e:
            st.error(f"Error analyzing image: {str(e)}")
            raise

def main():
    st.title("Parking Analysis")
    
    api_key = load_api_key()
    if not api_key:
        return
        
    analyzer = EnhancedParkingAnalyzer(api_key)
    
    uploaded_file = st.file_uploader("Upload parking lot image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        try:
            pil_image, cv2_image = analyzer.preprocess_image(uploaded_file)
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(pil_image)
            
            results = analyzer.analyze_image(pil_image, cv2_image)
            
            with col2:
                st.subheader("Analysis Result")
                st.image(cv2.cvtColor(results['annotated_image'], cv2.COLOR_BGR2RGB))
            
            st.subheader("Zone Analysis")
            cols = st.columns(len(results['zone_stats']))
            
            for i, (zone_name, stats) in enumerate(results['zone_stats'].items()):
                total = stats['free'] + stats['occupied']
                if total > 0:
                    occupancy_rate = (stats['occupied'] / total) * 100
                    cols[i].metric(
                        f"Zone {zone_name}",
                        f"{stats['free']} free spaces",
                        f"{occupancy_rate:.1f}% occupied"
                    )
            
            st.success(results['recommendation'])
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.write("Full error:", e)

if __name__ == "__main__":
    main()