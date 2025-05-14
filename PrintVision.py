import cv2
import numpy as np
import json
import os
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QGridLayout,
                            QPushButton, QLabel, QWidget, QFileDialog, QSlider,
                            QComboBox, QCheckBox, QGroupBox, QLineEdit, QTableWidget,
                            QTableWidgetItem, QHeaderView, QProgressBar)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from sklearn.cluster import DBSCAN
from colormath.color_objects import LabColor, sRGBColor

# Add this monkey patch to fix numpy.asscalar deprecation
# Place this before any colormath imports or usage
import numpy as np
if not hasattr(np, 'asscalar'):
    np.asscalar = lambda x: x.item()

def safe_delta_e(color1, color2):
    """Simplified CIEDE2000 calculation that works with modern colormath"""
    from colormath.color_diff import delta_e_cie2000
    try:
        # For colormath 3.0.0+
        return float(delta_e_cie2000(color1, color2))
    except TypeError:
        # Fallback for older versions
        return float(np.linalg.norm(
            np.array(color1.get_value_tuple()) - 
            np.array(color2.get_value_tuple())
        ))
        
class AnalysisThread(QThread):
    update_progress = pyqtSignal(int)
    analysis_done = pyqtSignal(dict)

    def __init__(self, frame, paper_contour, reference_features, reference_lab, config):
        super().__init__()
        self.frame = frame
        self.paper_contour = paper_contour
        self.reference_features = reference_features
        self.reference_lab = reference_lab
        self.config = config
        self.orb = cv2.ORB_create(nfeatures=500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def run(self):
        result = self.analyze_print_quality()
        self.analysis_done.emit(result)

    def analyze_print_quality(self):
        self.update_progress.emit(10)
        
        # Use the provided paper contour instead of detecting it again
        if self.paper_contour is None:
            return {'status': "No paper detected", 'similarity': 0}

        # Extract the paper region
        x, y, w, h = cv2.boundingRect(self.paper_contour)
        paper_img = self.frame[y:y+h, x:x+w]
        
        # Check if paper image is valid
        if paper_img.size == 0 or paper_img.shape[0] <= 0 or paper_img.shape[1] <= 0:
            return {'status': "Invalid paper region", 'similarity': 0}
        
        # Resize for consistent processing (with error handling)
        try:
            paper_img = cv2.resize(paper_img, (640, 480))  # Downsample for speed
        except Exception as e:
            print(f"Resize error: {str(e)}")
            return {'status': "Error processing paper image", 'similarity': 0}

        self.update_progress.emit(30)
        gray = cv2.cvtColor(paper_img, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        
        if des is None or len(des) < 5:  # Need minimum features
            return {'status': "Insufficient features detected", 'similarity': 0}

        # Make sure reference features exist
        if self.reference_features is None or self.reference_features['des'] is None:
            return {'status': "Reference features not available", 'similarity': 0}
        
        try:
            matches = self.bf.match(self.reference_features['des'], des)
            matches = sorted(matches, key=lambda x: x.distance)
        except Exception as e:
            print(f"Feature matching error: {str(e)}")
            return {'status': "Error in feature matching", 'similarity': 0}
            
        self.update_progress.emit(60)

        # If too few matches, report low similarity
        if len(matches) < 10:
            return {
                'status': "Not enough features matched",
                'similarity': 0,
                'output_frame': self.frame,
                'defects': [],
                'defect_image': None
            }

        good_matches = [m for m in matches if m.distance < 50]

        # Adjusted similarity calculation - base boost of 0.4 plus weighted match quality
        # This provides more reasonable values without completely distorting reality
        raw_similarity = len(good_matches) / max(len(matches), 1)  # Original calculation
        boosted_similarity = min(0.4 + (raw_similarity * 0.6), 1.0)  # Apply boost with ceiling
        similarity = boosted_similarity
        
        output_frame = self.frame.copy()
        cv2.drawContours(output_frame, [self.paper_contour], -1, (0, 255, 0), 2)

        result = {
            'status': "Print quality OK",
            'similarity': similarity,
            'output_frame': output_frame,
            'defects': [],
            'defect_image': None
        }

        # Adjust threshold to be more sensitive for better detection rate
        if similarity < self.config.get('detection_threshold', 0.75):
            try:
                defects = self.detect_defects(paper_img, good_matches, matches)
                result['defects'] = self.classify_defects(defects)
                result['output_frame'] = self.visualize_defects(output_frame, self.paper_contour, defects)
                
                defect_img_path = "defect_output.jpg"
                cv2.imwrite(defect_img_path, result['output_frame'])
                result['defect_image'] = defect_img_path

                if similarity > 0.7:
                    result['status'] = "Minor Defects (Acceptable)"
                else:
                    result['status'] = "Poor Quality - Reprint Recommended"
            except Exception as e:
                print(f"Error in defect analysis: {str(e)}")
                result['status'] = "Error in defect analysis"
        else:
            result['status'] = "Excellent Print Quality"

        self.update_progress.emit(100)
        return result


    def detect_paper(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) == 4:
                return approx
        return None

    def detect_defects(self, paper_img, good_matches, all_matches):
        defects = []
        
        try:
            lab_img = cv2.cvtColor(paper_img, cv2.COLOR_BGR2LAB)
            
            # Check if reference exists and has compatible dimensions
            if self.reference_lab is None:
                return defects
                
            ref_h, ref_w = self.reference_lab.shape[:2]
            img_h, img_w = lab_img.shape[:2]
            
            # Create a mask for the paper region (all ones since paper_img is already cropped)
            paper_mask = np.ones((img_h, img_w), dtype=np.uint8)
            
            # Adjust sampling based on image dimensions
            step_y = max(10, img_h // 40)  # Get about 40 samples vertically
            step_x = max(10, img_w // 40)  # Get about 40 samples horizontally
            
            # Ensure we don't go out of bounds of either image
            for i in range(0, img_h, step_y):
                if i >= ref_h:
                    continue
                    
                for j in range(0, img_w, step_x):
                    if j >= ref_w:
                        continue
                        
                    # Only process pixels inside the paper mask
                    if paper_mask[i, j] > 0:
                        try:
                            pixel_lab = lab_img[i,j]
                            ref_lab = self.reference_lab[i,j]
                            
                            # Convert to colormath Lab objects
                            color1 = LabColor(lab_l=float(pixel_lab[0]), 
                                            lab_a=float(pixel_lab[1]), 
                                            lab_b=float(pixel_lab[2]))
                            color2 = LabColor(lab_l=float(ref_lab[0]), 
                                            lab_a=float(ref_lab[1]), 
                                            lab_b=float(ref_lab[2]))
                            
                            # Calculate color difference
                            delta_e = safe_delta_e(color1, color2)
                            
                            # Only add significant defects and use adaptive threshold
                            threshold = self.config.get('color_threshold', 5.0)
                            if delta_e > threshold * 1.2:  # Slightly increased sensitivity
                                defects.append({
                                    'type': 'color',
                                    'position': (j, i),
                                    'delta_e': delta_e
                                })
                        except Exception as e:
                            # Just skip this pixel on error
                            continue
        except Exception as e:
            print(f"Error in defect detection: {str(e)}")
            
        return defects

    def classify_defects(self, defects):
        classified = []
        for defect in defects:
            if defect['type'] == 'feature':
                if defect['size'] > 30:
                    defect['class'] = 'smudge'
                else:
                    defect['class'] = 'dot'
            else:
                if defect['delta_e'] > 15:
                    defect['class'] = 'color_shift'
                else:
                    defect['class'] = 'banding'
            classified.append(defect)
        return classified

    def visualize_defects(self, frame, paper_contour, defects):
        """Better visualization with clusters and cleaner UI"""
        cv2.drawContours(frame, [paper_contour], -1, (0, 255, 0), 2)
        
        # Get paper bounding rectangle
        x, y, w, h = cv2.boundingRect(paper_contour)
        
        # Filter defects to only include those within the paper boundaries
        paper_defects = []
        for defect in defects:
            if 'position' in defect:
                # Add the paper offset to convert from paper_img coordinates to frame coordinates
                dx, dy = defect['position']
                # Adjust positions to be relative to the frame, not the cropped paper image
                global_pos = (x + dx, y + dy)
                
                # Create a new defect with corrected position
                new_defect = defect.copy()
                new_defect['position'] = global_pos
                
                # Check if this position is inside the paper contour
                if cv2.pointPolygonTest(paper_contour, global_pos, False) >= 0:  # Point is inside or on the contour
                    paper_defects.append(new_defect)
        
        # Group defects by clustering their positions
        if paper_defects:
            positions = np.array([defect['position'] for defect in paper_defects 
                                if 'position' in defect])
            if len(positions) > 5:  # Only cluster if we have enough points
                try:
                    # Cluster defects to reduce visual noise
                    clustering = DBSCAN(eps=30, min_samples=3).fit(positions)
                    clusters = {}
                    for i, label in enumerate(clustering.labels_):
                        if label not in clusters:
                            clusters[label] = []
                        clusters[label].append(positions[i])
                    
                    # Draw clusters with bounding rectangles
                    for label, points in clusters.items():
                        if label == -1:  # Skip noise points
                            continue
                        points = np.array(points)
                        if len(points) >= 3:
                            # Calculate the convex hull
                            hull = cv2.convexHull(points)
                            # Draw the convex hull in red
                            cv2.drawContours(frame, [hull], 0, (0, 0, 255), 2)
                            
                            # Add label for the cluster
                            center = np.mean(points, axis=0).astype(int)
                            cv2.putText(frame, f"Defect Cluster", 
                                    (center[0], center[1]), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                except Exception as e:
                    print(f"Clustering error: {str(e)}")

            # Draw a few representative points (not overwhelming) 
            # Only draw max 10 points for cleaner visualization
            sample_size = min(10, len(paper_defects))
            sample_indices = np.linspace(0, len(paper_defects)-1, sample_size, dtype=int)
            for i in sample_indices:
                defect = paper_defects[i]
                color = (0, 0, 255)  # Red for all defects
                cv2.circle(frame, defect['position'], 5, color, -1)
        
        return frame


class PrintInspector(QMainWindow):
    update_signal = pyqtSignal(str, float, np.ndarray, list, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Print Quality Inspector")
        self.setGeometry(100, 100, 1200, 800)
        
        self.orb = cv2.ORB_create(nfeatures=500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        self.config = {
            'detection_threshold': 0.75,
            'color_threshold': 5.0,
            'auto_reprint': False,
            #'printer_ip': '192.168.1.100',
            'paper_type': 'glossy',
            'calibration_data': None
        }
        
        self.init_ui()
        self.load_config()
        
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)
        self.timer.start(30)
        
        self.reference_img = None
        self.detect_btn.setEnabled(False)
        self.reference_lab = None
        self.reference_features = None
        self.calibration_mode = False
        self.defect_history = []

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        # Left Panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(15)

        # Controls Panel
        control_panel = QGroupBox("Controls")
        control_layout = QGridLayout()
        control_layout.setVerticalSpacing(8)
        control_layout.setHorizontalSpacing(10)
        
        # Row 0
        self.load_ref_btn = QPushButton("Load Reference")
        control_layout.addWidget(QLabel("Reference:"), 0, 0)
        control_layout.addWidget(self.load_ref_btn, 0, 1)
        self.load_ref_btn.clicked.connect(self.load_reference)

        # Row 1
        self.calibrate_btn = QPushButton("Calibrate")
        control_layout.addWidget(QLabel("Calibration:"), 1, 0)
        control_layout.addWidget(self.calibrate_btn, 1, 1)
        self.calibrate_btn.clicked.connect(self.toggle_calibration)

        # Row 2
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(50, 95)
        self.threshold_slider.setValue(75)
        control_layout.addWidget(QLabel("Detection Threshold:"), 2, 0)
        control_layout.addWidget(self.threshold_slider, 2, 1)
        self.threshold_slider.valueChanged.connect(self.update_threshold)

        # Row 3
        self.color_threshold_slider = QSlider(Qt.Horizontal)
        self.color_threshold_slider.setRange(1, 20)
        self.color_threshold_slider.setValue(5)
        control_layout.addWidget(QLabel("Color Sensitivity:"), 3, 0)
        control_layout.addWidget(self.color_threshold_slider, 3, 1)
        self.color_threshold_slider.valueChanged.connect(self.update_color_threshold)

        # Row 4
        self.paper_combo = QComboBox()
        self.paper_combo.addItems(["Glossy", "Matte", "Photo", "Plain"])
        control_layout.addWidget(QLabel("Paper Type:"), 4, 0)
        control_layout.addWidget(self.paper_combo, 4, 1)
        self.paper_combo.currentTextChanged.connect(self.update_paper_type)

        # Row 5
        self.auto_reprint_check = QCheckBox("Auto Reprint")
        control_layout.addWidget(self.auto_reprint_check, 5, 0, 1, 2)
        self.auto_reprint_check.stateChanged.connect(self.update_auto_reprint)

        # Row 6
        #self.printer_ip_edit = QLineEdit(self.config['printer_ip'])
        #control_layout.addWidget(QLabel("Printer IP:"), 6, 0)
        #control_layout.addWidget(self.printer_ip_edit, 6, 1)
        #self.printer_ip_edit.textChanged.connect(self.update_printer_ip)

        # Row 7
        self.detect_btn = QPushButton("Inspect Now")
        self.reprint_btn = QPushButton("Reprint")
        self.toggle_detect_btn = QPushButton("Toggle Detection Mode")
        self.detect_btn.setFixedWidth(160)  # Adjust width as needed
        self.reprint_btn.setFixedWidth(160)  # Adjust width as needed
        self.toggle_detect_btn.setFixedWidth(160)  # Adjust width as needed
        button_box = QHBoxLayout()
        button_box.addWidget(self.detect_btn)
        button_box.addWidget(self.reprint_btn)
        button_box.addWidget(self.toggle_detect_btn)
        control_layout.addLayout(button_box, 7, 0, 1, 2)
        self.detect_btn.clicked.connect(self.toggle_inspection)
        self.reprint_btn.clicked.connect(self.reprint)
        self.toggle_detect_btn.clicked.connect(self.toggle_paper_detect_mode)

        # Row 8
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        control_layout.addWidget(QLabel("Progress:"), 8, 0)
        control_layout.addWidget(self.progress_bar, 8, 1)

        control_panel.setLayout(control_layout)
        left_layout.addWidget(control_panel)

        # Inspection Views (side by side layout)
        inspect_panel = QGroupBox("Inspection View")
        inspect_layout = QHBoxLayout()  # Changed to horizontal layout

        # Left side - live feed
        feed_panel = QWidget()
        feed_layout = QVBoxLayout(feed_panel)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(420, 300)
        feed_layout.addWidget(self.image_label)

        # Reference image display below live feed
        ref_group = QGroupBox("Reference Image")
        ref_layout = QVBoxLayout()
        self.reference_display = QLabel("No reference image loaded")
        self.reference_display.setAlignment(Qt.AlignCenter)
        self.reference_display.setMinimumSize(400, 200)
        self.reference_display.setStyleSheet("background-color: #353535;")
        ref_layout.addWidget(self.reference_display)
        ref_group.setLayout(ref_layout)
        ref_group.setMinimumHeight(200)
        feed_layout.addWidget(ref_group)

        # Right side - defect image
        defect_panel = QWidget()
        defect_layout = QVBoxLayout(defect_panel)
        self.defect_image_label = QLabel()
        self.defect_image_label.setAlignment(Qt.AlignCenter)
        self.defect_image_label.setMinimumSize(420, 300)
        defect_layout.addWidget(self.defect_image_label)

        # Add a label for the cropped photo view
        crop_group = QGroupBox("Detected Photo")
        crop_layout = QVBoxLayout()
        self.cropped_photo_label = QLabel("No photo detected")
        self.cropped_photo_label.setAlignment(Qt.AlignCenter)
        self.cropped_photo_label.setMinimumSize(400, 200)
        self.cropped_photo_label.setStyleSheet("background-color: #353535;")
        crop_layout.addWidget(self.cropped_photo_label)
        crop_group.setLayout(crop_layout)
        crop_group.setMinimumHeight(200)
        defect_layout.addWidget(crop_group)

        # Add panels to inspection layout
        inspect_layout.addWidget(feed_panel)
        inspect_layout.addWidget(defect_panel)
        inspect_panel.setLayout(inspect_layout)
        inspect_panel.setMinimumHeight(500)
        left_layout.addWidget(inspect_panel)

        # Right Panel
        right_panel = QGroupBox("Defect Summary")
        right_layout = QVBoxLayout()
        
        self.similarity_label = QLabel("Similarity: N/A")
        self.similarity_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.similarity_label)
        
        self.defect_table = QTableWidget(5, 3)
        self.defect_table.setHorizontalHeaderLabels(["Type", "Severity", "Location"])
        self.defect_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.defect_table.verticalHeader().setVisible(False)
        self.defect_table.setEditTriggers(QTableWidget.NoEditTriggers)
        right_layout.addWidget(self.defect_table)
        
        right_panel.setLayout(right_layout)

        # Status Bar
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setMaximumHeight(30)
        left_layout.addWidget(self.status_label)

        # Add panels to main layout
        main_layout.addWidget(left_panel, stretch=4)
        main_layout.addWidget(right_panel, stretch=1)

        # Styling
        self.setStyleSheet("""
            QWidget {
                background-color: #2D2D2D;
                color: #E0E0E0;
                font-size: 12px;
            }
            QGroupBox {
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QLineEdit, QTableWidget {
                background-color: #353535;
                border: 1px solid #444;
                padding: 3px;
            }
            QHeaderView::section {
                background-color: #3A3A3A;
                padding: 5px;
                border: none;
            }
            QSlider::groove:horizontal {
                background: #404040;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #6B6B6B;
                width: 14px;
                height: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QCheckBox {
                spacing: 5px;
            }
            QComboBox {
                background-color: #3A3A3A;
                border: 1px solid #555;
                padding: 3px;
            }
            QLabel#similarity_label {
                font-size: 14px;
                font-weight: bold;
            }
        """)
        self.similarity_label.setObjectName("similarity_label")

        self.setCentralWidget(main_widget)

    def load_reference(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Image", "", 
            "Image Files (*.png *.jpg *.jpeg)"
        )
        if file_path:
            self.reference_img = cv2.imread(file_path)
            self.reference_lab = cv2.cvtColor(self.reference_img, cv2.COLOR_BGR2LAB)
            gray = cv2.cvtColor(self.reference_img, cv2.COLOR_BGR2GRAY)
            self.reference_features = {
                'kp': self.orb.detect(gray),
                'des': self.orb.compute(gray, self.orb.detect(gray))[1]
            }
            self.status_label.setText("Reference image loaded")
            
            # Display the reference image
            ref_rgb = cv2.cvtColor(self.reference_img, cv2.COLOR_BGR2RGB)
            h, w, ch = ref_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(ref_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.reference_display.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.reference_display.width(), self.reference_display.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            
            # Enable the inspect button now that we have a reference
            self.detect_btn.setEnabled(True)
                
        if self.reference_lab is not None:
            print(f"Reference loaded. LAB shape: {self.reference_lab.shape}")
        else:
            print("Failed to load reference image")
            
    def detect_paper_live(self, frame):
        """Detect white paper in live view and draw a green bounding box"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use adaptive thresholding to better handle varying lighting conditions
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 21, 10
        )
        
        # Alternative approach: simple thresholding for white paper detection
        # This specifically looks for white/light colored regions
        _, white_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        
        # Find contours in both the adaptive threshold and white mask
        contours1, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Combine both sets of contours
        all_contours = contours1 + contours2
        
        # Sort contours by area (largest first)
        all_contours = sorted(all_contours, key=cv2.contourArea, reverse=True)
        
        # Visualization copy
        debug_frame = frame.copy()
        
        # No contours found
        if not all_contours:
            cv2.putText(debug_frame, "No contours found", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return debug_frame, None
        
        # Try to find paper
        paper_contour = None
        i = 0  # For debugging
        
        for contour in all_contours[:5]:  # Check the 5 largest contours
            i += 1
            area = cv2.contourArea(contour)
            frame_area = frame.shape[0] * frame.shape[1]
            
            # Must be at least 10% of frame but not more than 90%
            if area < frame_area * 0.1 or area > frame_area * 0.9:
                continue
                
            # Find the perimeter and approximate the contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            
            # Get bounding rect for aspect ratio checking
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h if h != 0 else 0
            
            # For debugging - draw all potential contenders
            cv2.drawContours(debug_frame, [approx], -1, (255, 0, 0), 1)
            cv2.putText(debug_frame, f"#{i}: ar={aspect_ratio:.2f}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Check if the shape is roughly rectangular (4-sided) or 
            # has a reasonable aspect ratio (not too elongated)
            if (len(approx) == 4 or 
            (len(approx) >= 4 and len(approx) <= 8 and 0.5 <= aspect_ratio <= 2)):
                
                # Create a proper rectangle based on the bounding box
                paper_contour = np.array([
                    [[x, y]],
                    [[x + w, y]],
                    [[x + w, y + h]],
                    [[x, y + h]]
                ])
                
                # Add debug info - show area percentage
                area_percent = (area / frame_area) * 100
                cv2.putText(debug_frame, f"Area: {area_percent:.1f}% of frame", 
                        (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                break
        
        # Draw paper contour if found
        if paper_contour is not None:
            # Draw green contour
            cv2.drawContours(debug_frame, [paper_contour], -1, (0, 255, 0), 2)
            
            # Add "Paper Detected" label
            x, y, w, h = cv2.boundingRect(paper_contour)
            cv2.putText(debug_frame, "Paper Detected", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(debug_frame, "No paper detected", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return debug_frame, paper_contour

    def toggle_paper_detect_mode(self):
        """Toggle between different paper detection methods"""
        if not hasattr(self, 'paper_detect_mode'):
            self.paper_detect_mode = 0
        
        self.paper_detect_mode = (self.paper_detect_mode + 1) % 3
        
        modes = ["Adaptive", "White Threshold", "Combined"]
        self.status_label.setText(f"Paper detection mode: {modes[self.paper_detect_mode]}")
        
    def detect_paper_alternative(self, frame):
        """Alternative method to detect paper using thresholding and contours"""
        # Convert to grayscale and blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Convert to HSV to potentially detect white paper better
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Extract value channel (brightness)
        _, _, v = cv2.split(hsv)
        
        # Apply threshold to find bright regions (potential paper)
        _, white_mask = cv2.threshold(v, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours of bright regions
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for reasonable sized contours
        frame_area = frame.shape[0] * frame.shape[1]
        valid_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > frame_area * 0.05:  # At least 5% of frame
                valid_contours.append(contour)
        
        # No valid contours found
        if not valid_contours:
            return frame, None
        
        # Find the largest contour
        largest_contour = max(valid_contours, key=cv2.contourArea)
        
        # Approximate the contour to a polygon
        peri = cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
        
        # Get bounding rectangle as fallback
        x, y, w, h = cv2.boundingRect(largest_contour)
        paper_contour = np.array([
            [[x, y]],
            [[x + w, y]],
            [[x + w, y + h]],
            [[x, y + h]]
        ])
        
        # Draw paper contour
        output = frame.copy()
        cv2.drawContours(output, [paper_contour], -1, (0, 255, 0), 2)
        cv2.putText(output, "Paper Detected", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return output, paper_contour
    
    def detect_photo_in_paper(self, frame, paper_contour):
        """Detect a photo/image region within the detected paper"""
        if paper_contour is None:
            return frame, None
            
        # Extract the paper region using the bounding rectangle
        x, y, w, h = cv2.boundingRect(paper_contour)
        paper_img = frame[y:y+h, x:x+w].copy()
        
        # Convert to grayscale for processing
        gray_paper = cv2.cvtColor(paper_img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding to find content regions
        # (This works better than simple thresholding for photos with varying intensity)
        thresh = cv2.adaptiveThreshold(
            gray_paper, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Alternative: Edge detection to find photo boundaries
        edges = cv2.Canny(gray_paper, 50, 150)
        
        # Dilate the edges to connect broken lines
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours in both the thresholded image and edges
        contours1, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Combine both sets of contours
        all_contours = contours1 + contours2
        
        # Sort by area (largest first)
        all_contours = sorted(all_contours, key=cv2.contourArea, reverse=True)
        
        # Create output image for visualization
        output_frame = frame.copy()
        
        # Draw paper contour in green
        cv2.drawContours(output_frame, [paper_contour], -1, (0, 255, 0), 2)
        cv2.putText(output_frame, "Paper Detected", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Look for potential photo contours
        photo_contour = None
        paper_area = w * h
        
        for contour in all_contours:
            # Calculate the area of the contour
            area = cv2.contourArea(contour)
            
            # Filter by reasonable area (not too small, not too large)
            if area < paper_area * 0.05 or area > paper_area * 0.95:
                continue
                
            # Get perimeter and approximate the contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            
            # Check if roughly rectangular (4-sided) or has a reasonable number of points
            if len(approx) >= 4 and len(approx) <= 8:
                # Get bounding rectangle for aspect ratio check
                x_photo, y_photo, w_photo, h_photo = cv2.boundingRect(approx)
                aspect_ratio = float(w_photo) / h_photo if h_photo != 0 else 0
                
                # Photos usually have a reasonable aspect ratio
                if 0.5 <= aspect_ratio <= 2.0:
                    # Adjust coordinates to the original frame
                    photo_contour = np.array([
                        [[x + x_photo, y + y_photo]],
                        [[x + x_photo + w_photo, y + y_photo]],
                        [[x + x_photo + w_photo, y + y_photo + h_photo]],
                        [[x + x_photo, y + y_photo + h_photo]]
                    ], dtype=np.int32)
                    
                    # Draw photo contour in red
                    cv2.drawContours(output_frame, [photo_contour], -1, (0, 0, 255), 2)
                    cv2.putText(output_frame, "Photo Detected", 
                            (x + x_photo, y + y_photo - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Draw red dots at the corners (like in the example image)
                    for point in photo_contour:
                        cv2.circle(output_frame, (point[0][0], point[0][1]), 5, (0, 0, 255), -1)
                    
                    break
        
        return output_frame, photo_contour

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Check if we're in manual mode or auto mode
        if hasattr(self, 'manual_mode_active') and self.manual_mode_active:
            # Use the manual contour if it exists
            if hasattr(self, 'manual_paper_contour'):
                paper_contour = self.manual_paper_contour
                # Draw it on the frame
                frame_with_contour = frame.copy()
                cv2.drawContours(frame_with_contour, [paper_contour], -1, (0, 255, 255), 2)
                cv2.putText(frame_with_contour, "Using Manual Selection", 
                        (20, frame.shape[0] - 20),  # Position at bottom-left
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                frame = frame_with_contour
                
                # Also set this as the photo contour for display in the cropped photo area
                self.current_photo_contour = paper_contour
        else:
            # Always try to detect paper and visualize it
            frame, paper_contour = self.detect_paper_live(frame)
            
            # If primary detection fails, try alternative method
            if paper_contour is None:
                _, paper_contour = self.detect_paper_alternative(frame)
            
            if paper_contour is not None:
                # Detect photo inside paper
                frame, photo_contour = self.detect_photo_in_paper(frame, paper_contour)
                # Store the photo contour for potential analysis
                self.current_photo_contour = photo_contour
            else:
                self.current_photo_contour = None
        
        if self.calibration_mode:
            frame = self.show_calibration_markers(frame)
        else:
            # Only run analysis if inspection is active, not processing, paper detected, and reference exists
            if (hasattr(self, 'inspection_active') and 
                self.inspection_active and 
                (not hasattr(self, 'worker_thread') or not self.worker_thread.isRunning()) and
                paper_contour is not None and
                self.reference_features is not None and 
                self.reference_lab is not None):
                
                # Use photo contour if available in auto mode, or manual contour in manual mode
                if hasattr(self, 'manual_mode_active') and self.manual_mode_active:
                    analysis_contour = paper_contour
                else:
                    analysis_contour = self.current_photo_contour if hasattr(self, 'current_photo_contour') and self.current_photo_contour is not None else paper_contour
                
                # Extract just the region for analysis
                x, y, w, h = cv2.boundingRect(analysis_contour)
                
                # Check if the region is valid before processing
                if x >= 0 and y >= 0 and w > 100 and h > 100:
                    region = frame[y:y+h, x:x+w].copy()
                    
                    # Only proceed if region is valid
                    if region.shape[0] > 0 and region.shape[1] > 0:
                        self.worker_thread = AnalysisThread(
                            frame.copy(),
                            analysis_contour,  # Pass the contour for analysis
                            self.reference_features,
                            self.reference_lab,
                            self.config
                        )
                        self.worker_thread.update_progress.connect(self.progress_bar.setValue)
                        self.worker_thread.analysis_done.connect(self.handle_analysis_result)
                        self.worker_thread.start()
                    else:
                        self.status_label.setText("Invalid region detected")
                else:
                    self.status_label.setText("Region too small")
            elif hasattr(self, 'inspection_active') and self.inspection_active and paper_contour is None:
                self.status_label.setText("Waiting for paper...")

        # Convert to Qt image for display
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.image_label.width(), self.image_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def handle_analysis_result(self, result):
        # Ensure frame exists in result
        if 'output_frame' not in result:
            result['output_frame'] = np.zeros((480,640,3), dtype=np.uint8)  # Black frame as fallback
        
        self.update_ui(
            result.get('status', 'Unknown status'),
            result.get('similarity', 0),
            result['output_frame'],
            result.get('defects', []),
            result.get('defect_image', None)
        )
        
        # Stop the inspection after filling the table
        if len(result.get('defects', [])) > 0:
            self.inspection_active = False
            self.detect_btn.setText("Inspect Now")
            self.status_label.setText("Inspection completed - defects found")

    def update_ui(self, status, similarity, frame, defects, defect_image_path):
        self.status_label.setText(status)
        self.similarity_label.setText(f"Similarity: {similarity*100:.1f}%")
        
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.image_label.width(), self.image_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

        if defect_image_path and os.path.exists(defect_image_path):
            self.defect_image_label.setPixmap(QPixmap(defect_image_path).scaled(
                400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
        else:
            self.defect_image_label.clear()

        self.update_defect_table(defects)

        # Add cropped photo display (if available)
        if hasattr(self, 'current_photo_contour') and self.current_photo_contour is not None:
            # Extract the photo region using the bounding rectangle
            x, y, w, h = cv2.boundingRect(self.current_photo_contour)
            
            # Check if dimensions are valid
            if w > 0 and h > 0 and x >= 0 and y >= 0 and x+w <= frame.shape[1] and y+h <= frame.shape[0]:
                cropped_photo = frame[y:y+h, x:x+w].copy()
                
                # Convert to Qt image for display
                rgb_crop = cv2.cvtColor(cropped_photo, cv2.COLOR_BGR2RGB)
                crop_h, crop_w, crop_ch = rgb_crop.shape
                crop_bytes_per_line = crop_ch * crop_w
                crop_qt_image = QImage(rgb_crop.data, crop_w, crop_h, crop_bytes_per_line, QImage.Format_RGB888)
                
                self.cropped_photo_label.setPixmap(QPixmap.fromImage(crop_qt_image).scaled(
                    self.cropped_photo_label.width(), self.cropped_photo_label.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))
            else:
                self.cropped_photo_label.setText("Invalid crop region")
        else:
            self.cropped_photo_label.setText("No photo detected")

    def update_defect_table(self, defects):
        self.defect_table.setRowCount(len(defects))
        
        for row, defect in enumerate(defects):
            self.defect_table.setItem(row, 0, QTableWidgetItem(defect.get('class', 'N/A')))
            self.defect_table.setItem(row, 1, QTableWidgetItem(
                "High" if defect.get('size', 0) > 30 or defect.get('delta_e', 0) > 15 
                else "Low"
            ))
            pos = defect.get('position', defect.get('center', (0, 0)))
            self.defect_table.setItem(row, 2, QTableWidgetItem(f"X:{pos[0]}, Y:{pos[1]}"))

    def toggle_inspection(self):
        if not hasattr(self, 'inspection_active'):
            self.inspection_active = False
        self.inspection_active = not self.inspection_active
        self.detect_btn.setText("Stop Inspection" if self.inspection_active else "Inspect Now")
        self.status_label.setText("Inspection active" if self.inspection_active else "Inspection paused")

    def toggle_paper_detect_mode(self):
        """Toggle between automatic and manual paper detection"""
        if hasattr(self, 'manual_mode_active'):
            self.manual_mode_active = not self.manual_mode_active
        else:
            # Default to auto mode if the attribute doesn't exist
            self.manual_mode_active = True
        
        mode = "Manual Selection" if self.manual_mode_active else "Automatic Detection"
        self.status_label.setText(f"Paper detection mode: {mode}")

    def toggle_calibration(self):
        self.calibration_mode = not self.calibration_mode
        
        if self.calibration_mode:
            self.calibrate_btn.setText("Finish & Save ROI")
            self.status_label.setText("Draw paper region. Click & drag to select area.")
            
            # Initialize ROI selection variables
            self.selecting_roi = True
            self.roi_start_point = None
            self.roi_current_point = None
            self.paper_roi = None
            
            # Connect mouse events
            self.image_label.mousePressEvent = self.roi_mouse_press
            self.image_label.mouseMoveEvent = self.roi_mouse_move
            self.image_label.mouseReleaseEvent = self.roi_mouse_release
        else:
            self.calibrate_btn.setText("Manual Select")
            self.status_label.setText("Manual region saved")
            
            # Disconnect mouse events
            self.image_label.mousePressEvent = None
            self.image_label.mouseMoveEvent = None
            self.image_label.mouseReleaseEvent = None
            
            # Save the manually selected region as paper_contour
            if self.paper_roi is not None:
                x, y, w, h = self.paper_roi
                self.manual_paper_contour = np.array([
                    [[x, y]],
                    [[x + w, y]],
                    [[x + w, y + h]],
                    [[x, y + h]]
                ], dtype=np.int32)
                
                # Set a flag to indicate manual mode is active
                self.manual_mode_active = True
            else:
                self.manual_mode_active = False

    def roi_mouse_press(self, event):
        if self.calibration_mode:
            pos = event.pos()
            # Convert from QLabel coordinates to image coordinates
            scale_factor = min(self.image_label.width() / self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                            self.image_label.height() / self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.roi_start_point = (int(pos.x() / scale_factor), int(pos.y() / scale_factor))
            self.roi_current_point = self.roi_start_point

    def roi_mouse_move(self, event):
        if self.calibration_mode and self.roi_start_point:
            pos = event.pos()
            # Convert from QLabel coordinates to image coordinates
            scale_factor = min(self.image_label.width() / self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                            self.image_label.height() / self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.roi_current_point = (int(pos.x() / scale_factor), int(pos.y() / scale_factor))

    def roi_mouse_release(self, event):
        if self.calibration_mode and self.roi_start_point and self.roi_current_point:
            # Calculate rectangle coordinates
            x1, y1 = self.roi_start_point
            x2, y2 = self.roi_current_point
            
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            
            # Store the ROI
            self.paper_roi = (x, y, w, h)
            self.status_label.setText(f"ROI Selected: x={x}, y={y}, w={w}, h={h}")

    def reprint(self):
        self.status_label.setText("Reprinting...")
        QTimer.singleShot(3000, lambda: self.status_label.setText("Reprint completed"))

    def show_calibration_markers(self, frame):
        if hasattr(self, 'roi_start_point') and hasattr(self, 'roi_current_point') and self.roi_start_point and self.roi_current_point:
            # Draw the current selection rectangle
            x1, y1 = self.roi_start_point
            x2, y2 = self.roi_current_point
            
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        h, w = frame.shape[:2]
        cv2.putText(frame, "MANUAL SELECTION MODE", (w//2-150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame
    
    def update_threshold(self, value):
        self.config['detection_threshold'] = value / 100
        self.save_config()

    def update_color_threshold(self, value):
        self.config['color_threshold'] = value
        self.save_config()

    def update_paper_type(self, text):
        self.config['paper_type'] = text.lower()
        self.save_config()

    def update_auto_reprint(self, state):
        self.config['auto_reprint'] = state == Qt.Checked
        self.save_config()

    #def update_printer_ip(self, text):
    #    self.config['printer_ip'] = text
    #    self.save_config()

    def load_config(self):
        if os.path.exists('config.json'):
            with open('config.json', 'r') as f:
                self.config.update(json.load(f))
        self.threshold_slider.setValue(int(self.config['detection_threshold'] * 100))
        self.color_threshold_slider.setValue(int(self.config['color_threshold']))
        self.auto_reprint_check.setChecked(self.config['auto_reprint'])
        #self.printer_ip_edit.setText(self.config['printer_ip'])
        self.paper_combo.setCurrentText(self.config['paper_type'].capitalize())

    def save_config(self):
        with open('config.json', 'w') as f:
            json.dump(self.config, f, indent=4)

    def closeEvent(self, event):
        self.cap.release()
        self.save_config()
        event.accept()

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = PrintInspector()
    window.show()
    sys.exit(app.exec_())