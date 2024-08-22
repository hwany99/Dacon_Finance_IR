import numpy as np
import os
import io
import os
import io
import cv2 as cv
import numpy as np
import fitz  # PyMuPDF
from PIL import Image, ImageDraw

def count_unique_colors(idx, image):
    image_array = np.array(image)
    unique_colors = set(tuple(pixel) for row in image_array for pixel in row)
    # print(f"idx{idx}: unique colors: {len(unique_colors)}")
    return len(unique_colors)

########################## pdf extract calculate func ##########################

# 이진화된 이미지에서 재귀적으로 문서내의 각 섹션의 경계를 찾아냄
# 주로 텍스트나 그래픽이 포함된 블록을 식별
def find_section_borders(img_binary:np.array, 
                    min_w:float=0.0, min_h:float=0.0, 
                    min_dx:float=15.0, min_dy:float=15.0): 

    def cut_sections(arr:np.array, top_left:tuple, res:list, 
                    min_w:float, min_h:float, min_dx:float, min_dy:float):
        x0, y0 = top_left
        h, w = arr.shape

        projection = np.count_nonzero(arr==255, axis=1)
        pos_y = identify_gaps(projection, min_w, min_dy)
        if not pos_y: return        

        arr_y0, arr_y1 = pos_y
        for r0, r1 in zip(arr_y0, arr_y1):
            x_arr = arr[r0:r1, 0:w]
            projection = np.count_nonzero(x_arr==255, axis=0)
            pos_x = identify_gaps(projection, min_h, min_dx)
            if not pos_x: continue

            arr_x0, arr_x1 = pos_x
            if len(arr_x0)==1:
                res.append((x0+arr_x0[0], y0+r0, x0+arr_x1[0], y0+r1))
                continue
            
            for c0, c1 in zip(arr_x0, arr_x1):
                y_arr = arr[r0:r1, c0:c1]
                top_left = (x0+c0, y0+r0)
                cut_sections(y_arr, top_left, res, min_w, min_h, min_dx, min_dy)

    res = []
    cut_sections(arr=img_binary, top_left=(0, 0), res=res, 
            min_w=min_w, min_h=min_h, min_dx=min_dx, min_dy=min_dy)
    return res

# 최소값과 갭을 기준으로 객체의 시작과 끝을 찾음, 객체의 위치를 찾는데 사용
def identify_gaps(arr_values:np.array, min_value:float, min_gap:float):

    idx = np.where(arr_values>min_value)[0]
    if not len(idx): return

    gaps = idx[1:] - idx[0:-1]
    gap_idx = np.where(gaps>min_gap)[0]
    segment_start = idx[gap_idx]
    segment_end = idx[gap_idx+1]

    starts = np.insert(segment_end, 0, idx[0])
    ends = np.append(segment_start, idx[-1])
    ends += 1 

    return starts, ends

class daconCustomExtractor:
    def __init__(self, page: fitz.Page) -> None:
        self._page = page

    # 경계 상자에 따라 페이지의 특정 영역을 pixmap 으로 변환
    # pixmap = 문서의 특정 부분을 이미지로 렌더링
    def clip_page_to_pixmap(self, bbox: fitz.Rect = None, zoom: float = 3.0):
        if bbox is None:
            clip_bbox = self._page.rect
        elif self._page.rotation:
            clip_bbox = bbox * self._page.rotation_matrix
        else:
            clip_bbox = bbox
        clip_bbox = self._page.rect & clip_bbox
        matrix = fitz.Matrix(zoom, zoom)
        pix = self._page.get_pixmap(clip=clip_bbox, matrix=matrix)
        return pix

    # svg 윤곽선을 감지하여 각 이미지의 외부 및 내부 경계 분석 
    # 각 페이지 => 흑백 이미지로 변환후 이진화를 통해 경계를 찾아냄
    # 그후 클립하여 저장 
    def detect_svg_contours(self, page_num, output_dir, min_svg_gap_dx: float, min_svg_gap_dy: float, min_w: float, min_h: float):
        
        pixmap = self.clip_page_to_pixmap(zoom=1.0)
        src = self._pixmap_to_cv_image(pixmap)
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        _, binary = cv.threshold(gray, 253, 255, cv.THRESH_BINARY_INV)
        external_bboxes = find_section_borders(binary, min_dx=min_svg_gap_dx, min_dy=min_svg_gap_dy)

        # 화면 디버깅용 코드 => 화면에 경계를 표시해서 보여줌 
        # Convert to Pillow image for visualization
        # pil_image = Image.fromarray(cv.cvtColor(src, cv.COLOR_BGR2RGB))
        # draw = ImageDraw.Draw(pil_image)
        # for bbox in external_bboxes:
        #     x0, y0, x1, y1 = bbox
        #     draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
        # pil_image.show()

        # 클립된 영역 이미지 저장 
        os.makedirs(output_dir, exist_ok=True)
        for i, bbox in enumerate(external_bboxes):
            pixmap = self.clip_page_to_pixmap(bbox=bbox, zoom=3.0)
            image = Image.open(io.BytesIO(pixmap.tobytes()))
            if count_unique_colors(i, image) > 300:
                image.save(f"{output_dir}/{page_num}_{i}.png")
        return external_bboxes
    
    @staticmethod
    def _pixmap_to_cv_image(pixmap: fitz.Pixmap):
        import cv2 as cv
        import numpy as np
        img_byte = pixmap.tobytes()
        return cv.imdecode(np.frombuffer(img_byte, np.uint8), cv.IMREAD_COLOR)